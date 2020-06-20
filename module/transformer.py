import numpy as np
import tensorflow as tf
import time

BATCH_SIZE = 5
dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(20).batch(BATCH_SIZE)

def positional_embedding(pos, model_size):
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE

max_length = max(len(data_en[0]), len(data_fr_in[0]))
MODEL_SIZE = 128

pes = []
for i in range(max_length):
    pes.append(positional_embedding(i, MODEL_SIZE))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value, mask=None):
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)

            # Here we scale the score as described in the paper
            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, query_len, value_len)

            # mask must be broadcastable to (batch, query_len, value_len)
            if mask is not None:
                score *= mask

                # asign masked positions to -1e9
                # so that their values after softmax are zeros
                score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)

            alignment = tf.nn.softmax(score, axis=2)
            # alignment has shape (batch, query_len, value_len)

            head = tf.matmul(alignment, self.wv[i](value))
            # head has shape (batch, decoder_len, value_size)
            heads.append(head)

        # Concatenate all the attention heads
        # so that the last dimension summed up to model_size
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        # heads has shape (batch, query_len, model_size)
        return heads


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]

        self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(512, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

    def call(self, sequence, padding_mask=None):
        embed_out = self.embedding(sequence)
        embed_out += pes[:sequence.shape[1], :]

        sub_in = embed_out

        for i in range(self.num_layers):
            sub_out = self.attention[i](sub_in, sub_in, padding_mask)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)

            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(512, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, encoder_output, padding_mask=None):
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = self.embedding(sequence)
        embed_out += pes[:sequence.shape[1], :]

        bot_sub_in = embed_out

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            seq_len = bot_sub_in.shape[1]
            look_left_only_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            bot_sub_out = self.attention_bot[i](bot_sub_in, bot_sub_in, look_left_only_mask)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = self.attention_mid[i](mid_sub_in, encoder_output, padding_mask)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits


@tf.function
def train_step(source_seq, target_seq_in, target_seq_out):
    with tf.GradientTape() as tape:
        padding_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
        padding_mask = tf.expand_dims(padding_mask, axis=1)
        encoder_output = encoder(source_seq, padding_mask)

        decoder_output = decoder(target_seq_in, encoder_output, padding_mask)

        loss = loss_func(target_seq_out, decoder_output)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss

def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_output = encoder(tf.constant(test_source_seq))

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]], dtype=tf.int64)

    out_words = []

    while True:
        de_output = decoder(de_input, en_output)
        new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
        out_words.append(fr_tokenizer.index_word[new_word.numpy()[0][0]])

        de_input = tf.concat((de_input, new_word), axis=-1)

        if out_words[-1] == '<end>' or len(out_words) >= 14:
            break

    print(' '.join(out_words))


H = 2
NUM_LAYERS = 2

en_vocab_size = len(en_tokenizer.word_index) + 1
encoder = Encoder(en_vocab_size, MODEL_SIZE, NUM_LAYERS, H)
en_sequence_in = tf.constant([[1, 2, 3, 4, 6, 7, 8, 0, 0, 0],  [1, 2, 3, 4, 6, 7, 8, 0, 0, 0]])
encoder_output = encoder(en_sequence_in)

print('Input vocabulary size', en_vocab_size)
print('Encoder input shape', en_sequence_in.shape)
print('Encoder output shape', encoder_output.shape)

fr_vocab_size = len(fr_tokenizer.word_index) + 1
max_len_fr = data_fr_in.shape[1]
decoder = Decoder(fr_vocab_size, MODEL_SIZE, NUM_LAYERS, H)

fr_sequence_in = tf.constant([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0],
                           [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0]])
decoder_output = decoder(fr_sequence_in, encoder_output)

print('Target vocabulary size', fr_vocab_size)
print('Decoder input shape', fr_sequence_in.shape)
print('Decoder output shape', decoder_output.shape)

H = 2
NUM_LAYERS = 2

encoder = Encoder(en_vocab_size, MODEL_SIZE, NUM_LAYERS, H)
decoder = Decoder(fr_vocab_size, MODEL_SIZE, NUM_LAYERS, H)
crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)
def loss_func(targets, logits):
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


optimizer = tf.keras.optimizers.Adam()

NUM_EPOCHS = 100

start_time = time.time()
for e in range(NUM_EPOCHS):
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,
                          target_seq_out)

    print('Epoch {} Loss {:.4f}'.format(
          e + 1, loss.numpy()))

    if (e + 1) % 10 == 0:
        end_time = time.time()
        print('Average elapsed time: {:.2f}s'.format((end_time - start_time) / (e + 1)))
        try:
            predict()
        except Exception as e:
            print(e)
            continue