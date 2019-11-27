import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import data_pipe

def make_model(batchsize, return_state=False):
    numclasses = 256 # assume utf-8 bytes
    layersize = 1024
    numlayers = 4
    memsize = 32
    inputs = tf.keras.Input((None,), batch_size=batchsize)
    # Embed Characters
    char_embeds_3 = tf.keras.layers.Embedding(numclasses, layersize)(inputs)
    # Sequence layers
    cell = DNCCell(layersize, memsize, numlayers)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state,
            stateful=True)
    outputs = rnn(char_embeds_3)
    if return_state:
        char_embeds_3 = outputs[0]
        memory_states = tuple(outputs[1:])
    else:
        char_embeds_3 = outputs
    # Output layer
    logits_3 = tf.keras.layers.Dense(numclasses,  None)(char_embeds_3)
    # Model
    if return_state:
        model = tf.keras.Model(inputs=inputs, outputs=(logits_3, (memory_states)))
        return model
    else:
        model = tf.keras.Model(inputs=inputs, outputs=logits_3)
        return model

class DNCCell(tf.keras.layers.Layer):
    """
    A dumbed down version of a differentiable neural computer. At each timestep it reads from
    memory, computes a new outputvalue, and writes to memory.

    Arguments:
        units: Positive integer, dimensionality of the output space.
        memsize: Positive integer, mumber of memory states
        depth: Positive integer, number of layers that will share this cell's memory.
    """

    def __init__(self, units, memsize, depth, **kwargs):
        self.units = units
        self.state_size = (units * memsize,) +  (depth*(memsize + 1, memsize))
        self.nested_state_size = (units * memsize, depth*((memsize + 1, memsize),))
        self.output_size = units
        self.memsize = memsize
        self.depth = depth
        super(DNCCell, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'memsize': self.memsize,
            'depth': self.depth
        })
        return config

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            raise ValueError('input size must match output size, got: {}, {}'.format(
                input_shape[-1], self.units))
        self.layers = []
        for layernum in range(self.depth):
            layer = {}
            layer['readlayer'] = tf.keras.layers.Dense(self.memsize + 1, tf.nn.softmax)
            layer['writelayer'] = tf.keras.layers.Dense(self.memsize, tf.nn.sigmoid)
            layer['kernel'] = tf.keras.layers.Dense(self.units, tf.nn.relu)
            self.layers.append(layer)
        self.built = True

    def call(self, inputs, states, training=False):
        memory_3 = tf.reshape(states[0], [-1, self.memsize, self.units])
        readwrites_by_layer = []
        for layer in self.layers:
            outputs, memory_3, readwrites = self._runlayer(inputs,
                    memory_3, layer)
            readwrites_by_layer.append(readwrites)
        memory_2 = tf.reshape(memory_3, [-1, self.memsize * self.units])
        return outputs, [memory_2, readwrites_by_layer]

    def _runlayer(self, inputs_2, memory_3, layer):
        """
        Process one of this cell's potentially many layers.
        inputs_2: 2D tensor input to call or output from a previous layer
        memory_3: The 3D memory tensor extracted from the the cell state
        kernels: A dict containing this layer's 'readlayer', 'writelayer', and 'kernel'
        """
        # Take the memory state from the previous step or layer and concat the current input to it
        inputs_3 = tf.expand_dims(inputs_2, axis=1)
        attended_mem_3 = tf.concat([memory_3, inputs_3], axis=1)
        # Compute attention over memory
        read_weights_2 = layer['readlayer'](tf.reduce_mean(attended_mem_3, axis=1))
        read_weights_3 = tf.expand_dims(read_weights_2, axis=2)
        attended_mem_2 = tf.reduce_sum(read_weights_3 * attended_mem_3, axis=1)
        # Compute a new value from the attended memory
        transformed_2 = layer['kernel'](attended_mem_2)
        # Write the new value to memory
        write_weights_2 = layer['writelayer'](transformed_2)
        write_weights_3 = tf.expand_dims(write_weights_2, axis=2)
        transformed_3 = tf.expand_dims(transformed_2, axis=1)
        memory_3 = ((1 - write_weights_3) * memory_3) + (write_weights_3 * transformed_3)
        # Residual connection
        outputs = transformed_2
        outputs = inputs_2 + transformed_2
        return outputs, memory_3, (read_weights_2, write_weights_2)


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

def run_inference(model, input_string, numpredict, temperature=1e-16):
    print('----------------------------\nsoftmax temperature: {}'.format(temperature))
    temperature = tf.constant(temperature)
    # Convert string to integers. Prepend the start-of-text byte.
    input_string = bytes(chr(2) + input_string, 'utf-8')
    input_ids = _string_to_inputs(input_string, model.input_shape[0])
    result = [input_ids]
    for _ in range(numpredict):
        outputs = model.predict_on_batch(input_ids)
        latest_logits = outputs[:, -1, :]
        prediction = sample_logits(latest_logits, temperature)
        prediction = tf.cast(prediction, input_ids.dtype)
        input_ids = prediction
        result.append(prediction)
    # Remove the GO byte and convert to strings
    outstring = data_pipe.ids_to_string(tf.concat(result, axis=1)[:, 1:])
    outstring = [str(line.numpy(), 'utf-8') for line in outstring]
    # Print the results for each sequence in the batch
    for line in outstring:
        print(line.replace('\\n', '\n'), '\n')
    return outstring

def inspect_memory(model, input_string):
    """
    Plots the read and write weights used by DNC layers as they process their memory
    model: a keras model compiled with the DNC states appended to its outputs.
    input_string: the string to process
    """
    assert model.input_shape[0] == 1 # TODO support other batch sizes
    # Find out how many DNC layers the model has
    numlayers = len(model.output_shape[1][1:])//2 # FIXME this is a hard coded solution
    reads = [[] for _ in range(numlayers)]
    writes = [[] for _ in range(numlayers)]
    # Convert string to integers.
    input_string = bytes(chr(2) + input_string, 'utf-8')
    input_ids = _string_to_inputs(input_string, model.input_shape[0])
    # Pass each character in and collect the read/write weights
    for idx in range(input_ids.shape[1]):
        current_input = input_ids[:, idx:idx+1]
        results = model.predict_on_batch(current_input)
        # Manually extract the states and pack them into layers
        readwrites = results[2:]
        readwrites_by_layer = [(readwrites[i: i+2]) for i in range(0, len(readwrites), 2)]
        for layernum, layer in enumerate(readwrites_by_layer):
            reads[layernum].append(layer[0])
            writes[layernum].append(layer[1])
    # Plot the weights
    layered_reads = [np.concatenate(layer, axis=0) for layer in reads]
    layered_writes = [np.concatenate(layer, axis=0) for layer in writes]
    maxseqlen = 256
    ticks = range(len(input_string[:maxseqlen]))
    ticklabels = [chr(c) for c in input_string[-maxseqlen:]]
    for reads, writes in zip(layered_reads, layered_writes):
        fig, axs = plt.subplots(nrows=2)
        yticks = range(reads.shape[1])
        axs[0].imshow(np.transpose(reads[-maxseqlen:]), cmap='gray')
        axs[0].set_yticks(yticks, minor=False)
        axs[0].set_xticks(ticks, minor=False)
        axs[0].set_xticklabels(ticklabels, minor=False)
        axs[0].set_title('Read Weights')
        axs[1].imshow(np.transpose(writes[-maxseqlen:]), cmap='gray')
        yticks = range(writes.shape[1])
        axs[1].set_yticks(yticks, minor=False)
        axs[1].set_xticks(ticks, minor=False)
        axs[1].set_xticklabels(ticklabels, minor=False)
        axs[1].set_title('Write Weights')
        plt.show()

if __name__ == '__main__':
    # Run inference
    model = tf.keras.models.load_model('./model.h5', custom_objects={'DNCCell':DNCCell},
        compile=True)
    numpredict = 512
    lines = ['This sentence is an example']
    _ = run_inference(model, 'she', numpredict, 1e-16)
    lines = run_inference(model, 'she', numpredict, 0.5)
    _ = run_inference(model, 'she', numpredict, 0.75)
    # Plot the attention weights
    model = make_model(1, return_state=True)
    model.load_weights('./model.h5')
    inspect_memory(model, lines[0])
