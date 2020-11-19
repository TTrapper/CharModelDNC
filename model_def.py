import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import process_config
import numpy as np
import tensorflow as tf
import data_pipe
import random
import matplotlib.pyplot as plt


class AttendedEmbeddings(tf.keras.layers.Layer):
    """
    Embeddings that are selected via soft attention. Inputs to the call are multiplicative queries
    rather than row indices. Attention is multi-headed and the embeddings shapes match the queries.
    """
    def __init__(self, n_embeds, **kwargs):
        super(AttendedEmbeddings, self).__init__(**kwargs)
        self.n_embeds = n_embeds

    def get_config(self):
        config = super(AttendedEmbeddings, self).get_config()
        config.update({'n_embeds':self.n_embeds})
        return config

    def build(self, input_shape):
        # Shape of the embeddings matches the multi-headed queries
        batchsize, num_queries, nheads, headsize = input_shape
        self.embeddings = self.add_weight(shape=(nheads, self.n_embeds, headsize),
                name='embeddings')
        self.keys = self.add_weight(shape=(nheads, self.n_embeds, headsize), name='keys')
        self.normlayer = tf.keras.layers.LayerNormalization()

    def call(self, queries):
        batchsize, num_queries, nheads, headsize = queries.shape
        embeds = tf.reshape(self.embeddings, [nheads, self.n_embeds, 1, headsize])
        # Map function performs attention over queries and embeddings for a single batch element
        def fn(queries):
            # [nheads, nembeds, headsize] * [nheads, headsize, nquery] =>  nheads, nembeds, nquery
            logits = tf.nn.softmax(tf.matmul(self.keys, queries), axis=1)
            logits = tf.reshape(logits, [nheads, self.n_embeds, num_queries, 1])
            return tf.reduce_sum(embeds * logits, axis=1)
        # Attend to the embeddings
        queries = tf.transpose(queries, [0, 2, 3, 1]) # batch, nheads, headsize, num_queries
        selected_embeds = tf.vectorized_map(fn, queries)
        selected_embeds = tf.transpose(selected_embeds, [0, 2, 1, 3])
        return self.normlayer(selected_embeds) # batch, num_queries, nheads, headsize

class LinearProject(tf.keras.layers.Layer):
    """
    Wraps a linear projection followed by layer normalization and dropout
    """
    def __init__(self, size, dropout, **kwargs):
        super(LinearProject, self).__init__(**kwargs)
        self.dropout = dropout
        self.size = size
        self.dense = tf.keras.layers.Dense(size, None)
        self.layernorm = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super(LinearProject, self).get_config()
        config.update({'size':self.size, 'dropout':self.dropout})
        return config

    def call(self, values):
        values = self.layernorm(self.dense(values))
        return tf.keras.layers.Dropout(rate=self.dropout)(values)

class Positioner(tf.keras.layers.Layer):
    """
    Takes a sequence of shape: [batchsize, numitems, itemsize] and adds position information
    """
    def __init__(self, dropout, **kwargs):
        super(Positioner, self).__init__(**kwargs)
        self.dropout = dropout

    def get_config(self):
        config = super(Positioner, self).get_config()
        config.update({'dropout':self.dropout})
        return config

    def build(self, input_shape):
        batchsize, nslots, slotsize = input_shape
        self.dense = tf.keras.layers.Dense(slotsize, tf.nn.relu)
        # Fixed position embeds are just the binary representations of the numerical values
        places = 16
        powers = tf.constant([[2 ** i for i in range(places)]])
        pos_embeds = tf.range(nslots, 0, -1) # Count backwards: last item should be most recent
        pos_embeds = tf.tile(tf.expand_dims(pos_embeds, axis=1), [1, places])
        pos_embeds = tf.bitwise.bitwise_and(pos_embeds, powers)
        self.pos_embeds = tf.cast(pos_embeds != 0, tf.float32)
        self.pos_project = tf.keras.layers.Dense(slotsize, None)

    def call(self, values):
        batchsize, nslots, slotsize = values.shape
        pos = self.pos_project(self.pos_embeds)
        pos = tf.tile(tf.expand_dims(pos, axis=0), [batchsize, 1, 1]) # batchsize, nslots, nslots
        values = self.dense(values + pos)
        return tf.keras.layers.Dropout(rate=self.dropout)(values)

class Reader(tf.keras.layers.Layer):
    """
    Uses multi-head attention to read from an array of shape [batchsize, nslots, nheads, headsize].
    Additive attention is applied over the slots.
    """
    def __init__(self, kernelsize, dropout, project=True, **kwargs):
        super(Reader, self).__init__(**kwargs)
        self.kernelsize = kernelsize
        self.dropout = dropout
        self.project = project

    def get_config(self):
        config = super(Reader, self).get_config()
        config.update({'kernelsize':self.kernelsize, 'project':self.project, 'dropout':self.dropout})
        return config

    def build(self, input_shape):
        batchsize, nslots, nheads, headsize = input_shape
        self.attend_layer = tf.keras.layers.Dense(nheads, None, name='attention_layer')
        self.kernel = tf.keras.layers.Dense(self.kernelsize, tf.nn.relu)
        self.projection_layer = LinearProject(nheads * headsize, self.dropout,
                trainable=self.trainable) # TODO trainable has to be set explicitly. why?

    @staticmethod
    def get_keys(values):
        """ Create attention keys by combining each slot with the weighted averegage of slots """
        batchsize, nslots, nheads, headsize = values.shape
        slots = tf.reshape(values, [batchsize, nslots, nheads * headsize])
        mean_slots = tf.tile(tf.reduce_mean(slots, axis=1, keepdims=True), [1, nslots, 1])
        return tf.concat([slots, mean_slots], axis=2)


    def call(self, values, weights=None):
        batchsize, nslots, nheads, headsize = values.shape
        # Compute attention weights
        if weights is None:
            keys = self.get_keys(values)
            weights = tf.nn.softmax(self.attend_layer(keys), axis=1)
        weights = tf.expand_dims(weights, axis=3) # batch, nslots, nheads, 1
        # Apply attention and process result
        attended = tf.reshape(weights * values, [batchsize, nslots, nheads * headsize])
        attended = tf.reduce_sum(attended, axis=1)
        attended = self.kernel(attended)
        attended = tf.keras.layers.Dropout(rate=self.dropout)(attended)
        attended = tf.reshape(attended, [batchsize, 1, self.kernelsize])
        # Project to original size and restore multi-head shape
        if self.project:
            attended = self.projection_layer(attended)
            attended = tf.reshape(attended, [batchsize, 1, nheads,  headsize])
        return attended

class MultiReader(tf.keras.layers.Layer):
    """
    Performs parallel reads on the input sequence and concatenates the results.
    """
    def __init__(self, kernelsize, dropout, numreads, weighted_keys=False, **kwargs):
        super(MultiReader, self).__init__(**kwargs)
        self.kernelsize = kernelsize
        self.dropout = dropout
        self.numreads = numreads
        self.weighted_keys = weighted_keys
        self.readers = [Reader(kernelsize, dropout, **kwargs) for _ in range(numreads)]

    def build(self, input_shape):
        batchsize, nslots, nheads, headsize = input_shape
        self.query_layer = tf.keras.layers.Dense(self.numreads * nheads, None, name='queries')
        if self.weighted_keys:
            self.key_weights = self.add_weight(shape=(1, nslots, nheads, 1),
                    initializer=tf.ones_initializer, name='key_weights')

    def get_config(self):
        config = super(MultiReader, self).get_config()
        config.update({'kernelsize':self.kernelsize,
                       'dropout':self.dropout,
                       'numreads': self.numreads,
                       'weighted_keys': self.weighted_keys})
        return config

    def call(self, values):
        # Compute attention weights for all Readers simultaneously
        if self.weighted_keys:
            values *= self.key_weights
        keys = Reader.get_keys(values)
        weights = tf.nn.softmax(self.query_layer(keys), axis=1) # batch, nslots, numreads * nheads
        weights = tf.split(weights, self.numreads, axis=2)
        reads = [reader(values, weights=weights) for reader, weights in zip(self.readers, weights)]
        return tf.concat(reads, axis=1), weights

class RecurrentReader(tf.keras.layers.Layer):
    """
    Performs recurrent reads on the input sequence.
    gate (boolean): whether to explicitly store all previous outputs or compress them.
        if gate == False:
            history[t + 1] = concat([out[t], ..., out[0]])
        if gate == True:
            history[t + 1] = Reader([out[t], out[t-1]])
    """
    def __init__(self, kernelsize, dropout, gate=False, **kwargs):
        super(RecurrentReader, self).__init__(**kwargs)
        self.kernelsize = kernelsize
        self.dropout = dropout
        self.reader = Reader(kernelsize, dropout, **kwargs)
        self.gate = Reader(kernelsize, dropout, **kwargs) if gate else None

    def get_config(self):
        config = super(RecurrentReader, self).get_config()
        config.update({'kernelsize':self.kernelsize,
                       'dropout':self.dropout})
        return config

    def call(self, values, context=None):
        batchsize, seqlen, nheads, headsize = values.shape
        predictions = []
        last_predict = tf.zeros([batchsize, 0, nheads, headsize], dtype=values.dtype)
        for i in range(seqlen):
            current = values[:, i:i+1, :, :]
            prediction = self.reader(tf.concat([context, last_predict, current], axis=1))
            predictions.append(prediction)
            last_predict = tf.concat([last_predict, prediction], axis=1)
            if self.gate:
                last_predict = self.gate(last_predict)

        return tf.concat(predictions, axis=1)


class ConvReader(tf.keras.layers.Layer):
    def __init__(self, kernelsize, window, stride, dropout, **kwargs):
        super(ConvReader, self).__init__(**kwargs)
        self.kernelsize = kernelsize
        self.window = window
        self.stride = stride
        self.dropout = dropout

    def get_config(self):
        config = super(ConvReader, self).get_config()
        config.update({'kernelsize':self.kernelsize,
                       'window':self.window,
                       'stride':self.stride,
                       'dropout':self.dropout})
        return config

    def build(self, input_shape):
        batchsize, nslots, nheads, headsize = input_shape
        self.reader = Reader(self.kernelsize, self.dropout, project=False, trainable=self.trainable)

    def call(self, values):
        batchsize, nslots, nheads, headsize = values.shape
        results = []
        for i in reversed(range(nslots - self.window, -1, -self.stride)):
#            print(i, i + self.window)
            current_window = values[:, i:i + self.window, :]
            results.append(self.reader(current_window))
        return tf.concat(results, axis=1)


def make_model(config):
    inputs_collector = []
    outputs_collector = []
    dropout = config['dropout']
    headsize = config['headsize']
    batchsize = config['batchsize']
    seqlen = config['seqlen']
    train_char_embeds = config['train_char_embeds']
    char_embed_layer = tf.keras.layers.Embedding(config['numclasses'], config['char_embed_size'],
            trainable=train_char_embeds)
    char_embeds = embed_characters(char_embed_layer, batchsize, seqlen, train_char_embeds,
            inputs_collector)
    previous_slotsize = char_embeds.shape[-1]
    # Model is organized as a series of blocks that transorm the input sequence
    for blocknum, block_config in enumerate(config['blocks']):
        trainable = block_config['trainable']
        slotsize = block_config['slotsize']
        nheads = block_config['numheads'] # TODO rename to nheads
        blocktype = block_config['type']
        if previous_slotsize != slotsize:
            char_embeds = tf.reshape(char_embeds, [batchsize, seqlen, previous_slotsize])
            char_embeds = LinearProject(slotsize, dropout, trainable=trainable)(char_embeds)
        previous_slotsize = slotsize
        if blocktype == 'position':
            char_embeds = Positioner(dropout=dropout, trainable=trainable)(char_embeds)
        elif blocktype == 'conv':
            window = block_config['window']
            stride = block_config['stride']
            kernelsize = block_config['kernelsize']
            char_embeds = tf.reshape(char_embeds, [batchsize, seqlen, nheads, headsize])
            char_embeds = ConvReader(kernelsize, window, stride, dropout, trainable=trainable)\
                    (char_embeds)
            previous_slotsize = kernelsize # Conv layer doesn't project back down after kernel
        elif blocktype == 'multi':
            kernelsize = block_config['kernelsize']
            numreads = block_config['numreads']
            char_embeds = tf.reshape(char_embeds, [batchsize, seqlen, nheads, headsize])
            char_embeds, attention = MultiReader(kernelsize, dropout, numreads, weighted_keys=True,
                    trainable=trainable)(char_embeds)
        else:
            raise ValueError('Unrecognized blocktype: {}'.format(blocktype))
        seqlen = char_embeds.shape[1]
#    memory = AttendedEmbeddings(64)(char_embeds[:, :1, :, :])
#    char_embeds = tf.concat([char_embeds, memory], axis=1)
    context = char_embeds
    batchsize, nslots, nheads, headsize = context.shape
    # Embed sequence of future characters to be predicted
    predict_ahead = config['predict_ahead']
    ahead_char_embeds = embed_characters(char_embed_layer, config['batchsize'], predict_ahead,
            train_char_embeds, inputs_collector, add_normalized=False, name='ahead_char_ids')
    ahead_char_embeds = LinearProject(nheads * headsize, dropout, trainable=train_char_embeds,
            name='ahead_projection')(ahead_char_embeds)
    ahead_char_embeds = tf.reshape(ahead_char_embeds, [batchsize, predict_ahead, nheads, headsize])
    # Logits
    predictions = RecurrentReader(kernelsize, dropout, gate=True, trainable=True)(ahead_char_embeds,
            context=context)
    predictions = tf.reshape(predictions, [batchsize, predict_ahead, nheads * headsize])
    future_chars = tf.keras.layers.Dense(config['numclasses'], None,
            name='{}f{}'.format(blocknum, 128), trainable=trainable)(predictions)
    outputs_collector.append(future_chars)
    context = tf.reshape(context, [batchsize, 1, nslots * slotsize])
    next_char = tf.keras.layers.Dense(config['numclasses'], None, name='next_char')(context)
    outputs_collector.append(next_char)
#    outputs_collector.insert(0, attention)
    model = tf.keras.Model(inputs=tuple(inputs_collector), outputs=tuple(outputs_collector))
    return model

def embed_characters(char_embed_layer, batchsize, seqlen, trainable, inputs_collector,
        add_normalized=True, name='char_ids'):
    char_ids = tf.keras.Input(shape=(seqlen), batch_size=batchsize, name=name)
    inputs_collector.append(char_ids)
    char_embeds = char_embed_layer(char_ids)
    if add_normalized:
        char_ids_normalized = tf.keras.Input(shape=(seqlen), batch_size=batchsize,
                name=name + '_normalized')
        inputs_collector.append(char_ids_normalized)
        char_embeds += char_embed_layer(char_ids_normalized)
    char_embeds = tf.keras.layers.LayerNormalization(trainable=trainable)(char_embeds)
    return char_embeds # batch, seqlen, embedsize


@tf.function
def sample_logits(logits, temperature):
    prediction = tf.random.categorical(logits/temperature, num_samples=1)
    return prediction

def _string_to_inputs(input_string, batchsize):
    contextlen = len(input_string)
    input_ids = data_pipe.string_to_ids(tf.constant(input_string))
    # Stateful layers need their batchsize to be fixed, so just passing the same input in for now.
    input_ids = tf.expand_dims(input_ids, axis=0)
    input_ids = tf.tile(input_ids, [batchsize, 1])
    return input_ids

def run_inference(model, config, input_string, numpredict, temperature=1e-16):
    print('\n******************************************')
    print('softmax temperature: {}'.format(temperature))
    print('******************************************\n')
    temperature = tf.constant(temperature)
    # Convert string to integers. Prepend the start-of-text byte.
    input_string = bytes( input_string, 'utf-8')
    batchsize = model.input_shape[0][0]
    seqlen = model.input_shape[0][1]
    input_ids = _string_to_inputs(input_string, batchsize)
    result = [input_ids]
    pad = tf.zeros([batchsize, seqlen], input_ids.dtype)
    input_ids = tf.concat([pad, input_ids], axis=1)
    ahead = config['predict_ahead']
    ahead_ids = tf.zeros([batchsize, ahead], input_ids.dtype)
    for i in range(numpredict):
        input_ids = input_ids[:, -seqlen:]
        outputs = model.predict_on_batch((input_ids, data_pipe.normalize(input_ids), ahead_ids))
        """
        if i > 256:
            plt.imshow(np.concatenate(outputs[0], axis=2)[0])
            plt.savefig('./attention/together_{}'.format(i), dpi=128)
            plt.clf()
        """
        first_logits = outputs[-1][:, 0, :]
        prediction = sample_logits(first_logits, temperature)
        prediction = tf.cast(prediction, input_ids.dtype)
        input_ids = tf.concat([input_ids, prediction], axis=1)
        result.append(prediction)
    # Convert to strings
    outstring = data_pipe.ids_to_python_string(tf.concat(result, axis=1))
    # Print the results for each sequence in the batch
    max_numlines = 8
    for line in outstring[:max_numlines]:
        print(line.replace('\\n', '\n'), '\n')
        print('--------------------------------------------')
    return outstring

def plot_position_embeds(model):
    for layer in model.layers:
        if isinstance(layer, Positioner):
            raw_pos = layer.pos_embeds.numpy()
            projected_pos = layer.dense(layer.pos_project(layer.pos_embeds)).numpy()
    savedir = os.path.join(os.path.curdir, 'pos_embeds')
    print('Saving position embedding plots to {}'.format(savedir))
    maybe_makedir(savedir)
    im = plt.imshow(raw_pos)
    plt.savefig('{}/raw'.format(savedir), dpi=512)
    plt.clf()
    im = plt.imshow(projected_pos)
    plt.colorbar(im)
    plt.savefig('{}/projected'.format(savedir), dpi=512)
    plt.clf()

def plot_key_weights(model):
    weights = [w for w in model.weights if 'key_weights' in w.name]
    savedir = os.path.join(os.path.curdir, 'key_weights')
    print('Saving key_weight plots to {}'.format(savedir))
    maybe_makedir(savedir)
    for w in weights:
        matrix = w.numpy()[0, :, :, 0]
        im = plt.imshow(matrix)
        plt.colorbar(im)
        plt.savefig('{}/{}'.format(savedir, w.name.replace('/', '_')), dpi=512)
        plt.clf()

def maybe_makedir(path):
    if not os.path.isdir(path):
        if os.path.exists(path):
            raise IOError('Expected {} to be a directory'.format(path))
        os.makedirs(path)

if __name__ == '__main__':
    # Run inference
    config = process_config.load_config()
    config['batchsize'] = 4
    model = make_model(config)
    model.load_weights('./model.h5', by_name=True, skip_mismatch=True)

    """
    new_model = make_model(config, weighted_keys=True)
    new_weights = [w for w in new_model.weights if 'key_weights' not in w.name]
    assert len(new_weights) == len(model.weights)
    for old_weight, new_weight in zip(model.weights, new_weights):
        new_weight = new_weight.assign(old_weight)
    new_model.save('./newmodel.h5', save_format='h5', overwrite=True, include_optimizer=False)
    exit()
    """

    model.summary()
    plot_key_weights(model)
    plot_position_embeds(model)

    numpredict = 512
    lines = ['This sentence is an example']
    context = '. '
    _ = run_inference(model, config, context, numpredict, 1e-16)
    lines = run_inference(model, config, context, numpredict, 0.5)
    _ = run_inference(model, config, context, numpredict, 0.75)
    _ = run_inference(model, config, context, numpredict, 1.0)
