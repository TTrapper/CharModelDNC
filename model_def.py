import numpy as np
import tensorflow as tf

import data_pipe

def make_model():
    numclasses = 256 # assume utf-8 bytes
    layersize = 1024
    numlayers = 4
    memsize = 8
    inputs = tf.keras.Input((None,))
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'memsize': self.memsize,
        })
        return config

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            raise ValueError('input size must match output size, got: {}, {}'.format(
                input_shape[-1], self.units))
        self.readlayer = tf.keras.layers.Dense(self.memsize + 1, tf.nn.softmax)
        self.writelayer = tf.keras.layers.Dense(self.memsize, tf.nn.sigmoid)
        self.transformlayer = tf.keras.layers.Dense(self.units, tf.nn.relu)
        self.built = True

    def call(self, inputs, states, training=False):
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
        # Residual connection
        outputs = inputs + transformed_2
        return outputs, [memory_2]

@tf.function
def sample_logits(logits, temperature):
    prediction = tf.random.categorical(logits/temperature, num_samples=1)
    return prediction

def run_inference(model, context_string, numpredict, temperature=1e-16):
    context_string = bytes(context_string, 'utf-8')
    contextlen = len(context_string)
    temperature = tf.constant(temperature)
    input_ids = data_pipe.string_to_ids(tf.constant(context_string))
    input_ids = tf.expand_dims(input_ids, axis=0)
    for _ in range(numpredict):
        outputs = model.predict(input_ids, steps=1)
        latest_logits = outputs[:, -1, :]
        prediction = sample_logits(latest_logits, temperature)
        prediction = tf.cast(prediction, input_ids.dtype)
        input_ids = tf.concat([input_ids, prediction], axis=1)
    outstring = data_pipe.ids_to_string(input_ids)
    print(outstring[0].numpy())

if __name__ == '__main__':
#    model = make_model(seqlen)
    model = tf.keras.models.load_model('./model.hd5', custom_objects={'DNCCell':DNCCell},
        compile=True)
    run_inference(model, 'she', 63, 1e-16)
    run_inference(model, 'she', 63, 0.5)
    run_inference(model, 'she', 63, 0.775)


