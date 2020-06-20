import tensorflow as tf
import numpy as np


class Encoder(tf.keras.Model):
    def __init__(self, rnn_size, vocab_size=None, embedding_size=None):
        super(Encoder, self).__init__()
        self.rnn_size = rnn_size
        #         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        #         embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(sequence, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.rnn_size]), tf.zeros([batch_size, self.rnn_size]))


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size, attention_func):
        super(LuongAttention, self).__init__()
        self.attention_func = attention_func

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError('Unknown attention score function! Must be either dot, general or concat.')

        if attention_func == 'general':
            # General score function
            self.wa = tf.keras.layers.Dense(rnn_size)
        elif attention_func == 'concat':
            # Concat score function
            self.wa = tf.keras.layers.Dense(rnn_size, activation='tanh')
            self.va = tf.keras.layers.Dense(1)

    def call(self, decoder_output, encoder_output):
        if self.attention_func == 'dot':
            # Dot score function: decoder_output (dot) encoder_output
            # decoder_output has shape: (batch_size, 1, rnn_size)
            # encoder_output has shape: (batch_size, max_len, rnn_size)
            # => score has shape: (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
        elif self.attention_func == 'general':
            # General score function: decoder_output (dot) (Wa (dot) encoder_output)
            # decoder_output has shape: (batch_size, 1, rnn_size)
            # encoder_output has shape: (batch_size, max_len, rnn_size)
            # => score has shape: (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, self.wa(
                encoder_output), transpose_b=True)
        elif self.attention_func == 'concat':
            # Concat score function: va (dot) tanh(Wa (dot) concat(decoder_output + encoder_output))
            # Decoder output must be broadcasted to encoder output's shape first
            decoder_output = tf.tile(
                decoder_output, [1, encoder_output.shape[1], 1])

            # Concat => Wa => va
            # (batch_size, max_len, 2 * rnn_size) => (batch_size, max_len, rnn_size) => (batch_size, max_len, 1)
            score = self.va(
                self.wa(tf.concat((decoder_output, encoder_output), axis=-1)))

            # Transpose score vector to have the same shape as other two above
            # (batch_size, max_len, 1) => (batch_size, 1, max_len)
            score = tf.transpose(score, [0, 2, 1])

        # alignment a_t = softmax(score)
        alignment = tf.nn.softmax(score, axis=2)

        # context vector c_t is the weighted average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, rnn_size, attention_func, embedding_size=None):
        super(Decoder, self).__init__()
        self.attention = LuongAttention(rnn_size, attention_func)
        self.rnn_size = rnn_size
        #         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True)
        self.wc = tf.keras.layers.Dense(rnn_size, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state, encoder_output):
        # Remember that the input to the decoder
        # is now a batch of one-word sequences,
        # which means that its shape is (batch_size, 1)
        #         embed = self.embedding(sequence)

        # Therefore, the lstm_out has shape (batch_size, 1, rnn_size)
        lstm_out, state_h, state_c = self.lstm(sequence, initial_state=state)

        # Use self.attention to compute the context and alignment vectors
        # context vector's shape: (batch_size, 1, rnn_size)
        # alignment vector's shape: (batch_size, 1, source_length)
        context, alignment = self.attention(lstm_out, encoder_output)

        # Combine the context vector and the LSTM output
        # Before combined, both have shape of (batch_size, 1, rnn_size),
        # so let's squeeze the axis 1 first
        # After combined, it will have shape of (batch_size, 2 * rnn_size)
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)

        # lstm_out now has shape (batch_size, rnn_size)
        lstm_out = self.wc(lstm_out)

        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.ws(lstm_out)

        return logits, state_h, state_c, alignment


class Seq2SeqWithLuong:

    def __init__(self, input_dims, output_dims, rnn_size=512, attention_func='concat'):

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rnn_size = rnn_size
        self.attention_func = attention_func

        self.encoder = Encoder(rnn_size)
        self.decoder = Decoder(output_dims, rnn_size, attention_func)
        self.optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)
        return

    def loss_func(self, targets, logits):
        mse = tf.keras.losses.mean_squared_error(targets, logits)
        return mse

    @tf.function
    def train_step(self, source_seq, target_seq_in, target_seq_out, en_initial_states):
        loss = 0
        with tf.GradientTape() as tape:
            en_outputs = self.encoder(source_seq, en_initial_states)
            en_states = en_outputs[1:]
            de_state_h, de_state_c = en_states
            # We need to create a loop to iterate through the target sequences
            for i in range(target_seq_out.shape[1]):
                # Input to the decoder must have shape of (batch_size, length)
                # so we need to expand one dimension
                decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
                logit, de_state_h, de_state_c, _ = self.decoder(decoder_in, (de_state_h, de_state_c), en_outputs[0])
                # The loss is now accumulated through the whole batch
                loss += self.loss_func(target_seq_out[:, i], logit)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss / target_seq_out.shape[1]

    @tf.function
    def eval_step(self, source_seq, target_seq_in, target_seq_out, en_initial_states):
        loss = 0
        with tf.GradientTape() as tape:
            en_outputs = self.encoder(source_seq, en_initial_states)
            en_states = en_outputs[1:]
            de_state_h, de_state_c = en_states
            # We need to create a loop to iterate through the target sequences
            for i in range(target_seq_out.shape[1]):
                # Input to the decoder must have shape of (batch_size, length)
                # so we need to expand one dimension
                decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
                logit, de_state_h, de_state_c, _ = self.decoder(decoder_in, (de_state_h, de_state_c), en_outputs[0])
                # The loss is now accumulated through the whole batch
                loss += self.loss_func(target_seq_out[:, i], logit)
        return loss / target_seq_out.shape[1]

    def predict(self, X):
        source_seq, target_seq_in = X, X[:, -1:, :]
        en_initial_states = self.encoder.init_states(1)
        preds = []
        for i in np.arange(source_seq.shape[0]):
            en_outputs = self.encoder(source_seq[i:i + 1], en_initial_states)
            de_state_h, de_state_c = en_outputs[1:]
            de_output, de_state_h, de_state_c, alignment = self.decoder(target_seq_in[i:i + 1],
                                                                        (de_state_h, de_state_c), en_outputs[0])
            preds.append(de_output.numpy())
        return np.array(preds)

    def train(self, X, y, validation_data=None, epochs=10, batch_size=32, verbose=0):
        X1 = X
        X2 = X[:, -1:, :]
        dataset_trn = tf.data.Dataset.from_tensor_slices((X1, X2, y)).shuffle(batch_size)
        dataset_trn = dataset_trn.batch(batch_size, drop_remainder=True)
        his = []
        for e in range(epochs):
            en_initial_states = self.encoder.init_states(batch_size)
            #         encoder.save_weights('../cache/baseline-lstm-with-keras-0-7/encoder_{}.h5'.format(e + 1))
            #         decoder.save_weights('../cache/baseline-lstm-with-keras-0-7/decoder_{}.h5'.format(e + 1))
            trn_loss = 0
            for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset_trn.take(-1)):
                loss = self.train_step(source_seq, target_seq_in, target_seq_out, en_initial_states)
                trn_loss += np.mean(loss.numpy())
                if verbose > 0:
                    if batch % verbose == 0:
                        print(f'Epoch {e + 1} Batch {batch} Loss {np.mean(loss.numpy()):.4f}')
            trn_loss /= batch
            his_i = {'epoch': e + 1, 'loss': trn_loss}

            if not validation_data is None:
                X_val = validation_data[0]
                y_val = validation_data[1]
                X1_val = X_val
                X2_val = X_val[:, -1:, :]
                dataset_val = tf.data.Dataset.from_tensor_slices((X1_val, X2_val, y_val)).shuffle(batch_size)
                dataset_val = dataset_val.batch(batch_size, drop_remainder=True)
                en_initial_states = self.encoder.init_states(batch_size)
                val_loss = 0
                for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset_val.take(-1)):
                    loss = self.eval_step(source_seq, target_seq_in, target_seq_out, en_initial_states)
                    val_loss += np.mean(loss.numpy())
                    # Update val metrics
                    # val_acc_metric.update_state(y_batch_val, val_logits)
                # val_acc = val_acc_metric.result()
                his_i = {**his_i, 'val_loss:': val_loss}
            his.append(his_i)
        return his