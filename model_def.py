import numpy as np
import tensorflow as tf

import data_pipe

def make_model(batchsize, seqlen):
    numclasses = 256 # assume utf-8 bytes
    layersize = 512
    numlayers = 4
    memsize = 8
    inputs = tf.keras.Input((seqlen))
    # Embed Characters
    char_embeds_3 = tf.keras.layers.Embedding(numclasses, layersize)(inputs)
    # Sequence layers
    cells = [DNCCell(layersize, memsize) for layernum in range(numlayers)]
    rnn = tf.keras.layers.RNN(cells, return_sequences=True, stateful=False)
    char_embeds_3 = rnn(char_embeds_3)
    # Output layer
    logits_3 = tf.keras.layers.Dense(numclasses,  None)(char_embeds_3)
    # Model
    model = tf.keras.Model(inputs=inputs, outputs=logits_3)
    return model

class DNCCell(tf.keras.layers.Layer):
    """
    A dumbed down version of a differentiable neural computer. At each timestep it reads from
    memory, computes a new outputvalue, and writes to memory.

    Arguments:
        units: Positive integer, dimensionality of the output space.
        memsize: Positive integer, mumber of memory states
    """

    def __init__(self, units, memsize, **kwargs):
        self.units = units
        self.state_size = units * memsize
        self.output_size = units
        self.memsize = memsize
        super(DNCCell, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            raise ValueError('input size must match output size, got: {}, {}'.format(
                input_shape[-1], self.units))
        self.readlayer = tf.keras.layers.Dense(self.memsize + 1, tf.nn.softmax)
        self.writelayer = tf.keras.layers.Dense(self.memsize, tf.nn.sigmoid)
        self.transformlayer = tf.keras.layers.Dense(self.units, tf.nn.relu)
        self.built = True

    def call(self, inputs, states):
        # Take the memory state from the previous step concat the current input to it
        memory_3 = tf.reshape(states[0], [-1, self.memsize, self.units])
        value_3 = tf.expand_dims(inputs, axis=1)
        attended_mem_3 = tf.concat([memory_3, value_3], axis=1)
        # Compute attention over memory
        read_weights_2 = self.readlayer(tf.reduce_mean(attended_mem_3, axis=1))
        read_weights_3 = tf.expand_dims(read_weights_2, axis=2)
        attended_mem_2 = tf.reduce_sum(read_weights_3 * attended_mem_3, axis=1)
        # Compute a new value from the attended memory
        transformed_2 = self.transformlayer(attended_mem_2)
        # Write the new value to memory
        write_weights_3 = tf.expand_dims(self.writelayer(transformed_2), axis=2)
        transformed_3 = tf.expand_dims(transformed_2, axis=1)
        memory_3 = ((1 - write_weights_3) * memory_3) + (write_weights_3 * transformed_3)
        memory_2 = tf.reshape(memory_3, [-1, self.memsize * self.units])
        return transformed_2, [memory_2]

def run_inference(model, context_string, seqlen):
    context_string = bytes(context_string, 'utf-8')
    contextlen = len(context_string)
    assert contextlen < seqlen
    while contextlen  < seqlen:
        context_padded = context_string.ljust(seqlen)
        input_ids = data_pipe.string_to_ids(tf.constant(context_padded))
        input_ids = tf.expand_dims(input_ids, axis=0)
        outputs = model.predict(input_ids, steps=1)
        logits = outputs[0, contextlen - 1, :]
        prediction = bytes(chr(np.argmax(logits)), 'utf-8')
        context_string += prediction
        contextlen = len(context_string)
    print(context_string)
