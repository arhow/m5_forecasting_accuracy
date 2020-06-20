from keras.models import Model, load_model
from keras.layers import Input
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional
import tensorflow as tf
import numpy as np
import time
import os

def gru(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_sz = batch_sz
        # self.expand_dims = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[n_timeseries])
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        # x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, input_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(input_dim)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        #         print('Decoder.call', 'x', x.shape, 'enc_output', enc_output.shape, 'hidden', hidden.shape)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


class AttentionSeq2Seq:

    BUFFER_SIZE = 2048

    def __init__(self, input_dim, units, batch_size, buffer_size=2048, checkpoint_dir=None):
        self.input_dim = input_dim
        self.units = units
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.encoder = Encoder(units, batch_size)
        self.decoder = Decoder(input_dim, units, batch_size)
        self.optimizer = tf.train.AdamOptimizer()
        if type(checkpoint_dir) == type(None):
            self.checkpoint_dir = './training_checkpoints'
        else:
            self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        return

    def fit(self, encoder_inputs, decoder_inputs, decoder_outputs, epochs, verbose=1):

        dataset = tf.data.Dataset.from_tensor_slices((encoder_inputs, decoder_inputs, decoder_outputs)).shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        his = []
        for epoch in range(epochs):
            start = time.time()
            hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (en_input, de_input, de_output)) in enumerate(dataset):
                loss = 0

                with tf.GradientTape() as tape:
                    enc_output, enc_hidden = self.encoder(en_input, hidden)
                    dec_hidden = enc_hidden
                    # dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)

                    # Teacher forcing - feeding the target as the next input
                    for t in range(1, de_output.shape[1]):
                        # passing enc_output to the decoder
                        predictions, dec_hidden, _ = self.decoder(de_input, dec_hidden, enc_output)
                        loss += self.loss_function(de_output[:, t], predictions)
                        # using teacher forcing
                        de_input = tf.expand_dims(de_output[:, t], 1)

                batch_loss = (loss / int(de_output.shape[1]))
                total_loss += batch_loss
                variables = self.encoder.variables + self.decoder.variables
                gradients = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables))

                if batch % 100 == 0:
                    if verbose > 1:
                        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            his.append({'epoch': epoch + 1, 'loss': total_loss / (batch+1), 'sec': time.time() - start})
            if verbose > 0:
                print('Epoch {} Loss {:.4f} {}sec'.format(epoch + 1, total_loss / (batch+1), time.time() - start))
        return

    def predict(self):
        return

    def loss(self, real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)






# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp,
                                                max_length_targ)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()

def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.idx2word[predicted_id] + ' '

        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
translate(u'hace mucho frio aqui.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)


