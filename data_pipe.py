import model_def

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
import tensorflow as tf

def make_example_generator(filepath, seqlen):
    total_bytes = os.path.getsize(filepath)
    num_examples = total_bytes
    def example_generator():
        for i in range(num_examples):
            start = random.randint(0, total_bytes - seqlen)
            with open(filepath, 'rb') as f:
                f.seek(start)
                seq = f.read(seqlen)
                yield seq
    return example_generator

def file_to_dataset(filepath, batchsize, maxseqlen, maskinputs=True):
    seqlen = maxseqlen
    example_generator = make_example_generator(filepath, 1 + seqlen)
    lines = tf.data.Dataset.from_generator(example_generator, tf.string, tf.TensorShape([]))
    lines = lines.prefetch(32)
    lines = lines.map(lambda line: string_to_ids(line))
    lines = lines.map(lambda line: tf.reshape(line, [seqlen + 1])) # explicitly sets the shape
    lines = lines.batch(batchsize, drop_remainder=True)
    lines = lines.map(lambda line: (line[:, :-1], line[:, -1:]))
    lines = lines.map(lambda x,y: (x, (y, x)))
    if maskinputs:
        # Randomly mask some of the input values
        lines = lines.map(lambda x,y: (randomly_mask(x), y))
        lines = lines.map(lambda x,y: (randomly_sequence_mask(x), y))
    lines = lines.prefetch(4)
    return lines

def randomly_mask(tensor):
    maskprob = 0.20
    mask = tf.random.uniform(tensor.shape, minval=0, maxval=1, dtype=tf.dtypes.float32)
    mask = tf.where(tf.less(mask, maskprob), tf.zeros_like(mask), tf.ones_like(mask))
    return tensor * tf.cast(mask, tensor.dtype)

def randomly_sequence_mask(tensor):
    """
    Mask the first N characters of the first P examples in the batch, where N is uniformly sampled
    from range(0, len(example)) for each example, and P is a constant where 0 <= P <= batchsize.
    """
    percent_masked = 0.25 # proportion of batch elements that will have a mask applied
    batchsize, maxlen = tensor.shape
    num_masked = tf.cast(percent_masked * tf.cast(batchsize, tf.float32), tf.int32)
    lengths = tf.random.uniform(shape=[num_masked], minval=0, maxval=maxlen, dtype=tf.int32)
    lengths = tf.concat([lengths, tf.zeros([batchsize - num_masked], dtype=lengths.dtype)], axis=0)
    mask = tf.sequence_mask(lengths, maxlen=maxlen)
    return tf.where(mask, tf.ones_like(tensor), tensor)


def string_to_ids(tf_string):
    result = tf.strings.bytes_split(tf_string, 'UTF-8')
    # Decode raw bytes: data is preped to a fixed number of bytes per line so some valid utf-8
    # characters may get split into invalid utf-8 bytes if they lie on the boundary.
    result = tf.io.decode_raw(result, tf.uint8)
    result = tf.cast(result, tf.int32)
    result = tf.squeeze(result, axis=1)
    return result

def ids_to_string(tensor):
    result = tf.strings.unicode_encode(tensor, 'UTF-8', errors='ignore')
    return result

def ids_to_python_string(tensor):
    # Manually convert the ints to char bytes, then to a string. This avoids producing weird
    # characters when a unicode sequence has been broken up.
    result = tf.cast(tensor, tf.uint8).numpy()
    result = [str(bytes(line), 'utf-8', 'replace') for line in result]
    return result

if __name__ == '__main__':
    batchsize = 10
    maxseqlen = 32
    lines = file_to_dataset('./traindata.txt', batchsize, maxseqlen)
    print(lines)
    lines = iter(lines)
    for batch in range(5):
        print('\n\n\nbatch: {}'.format(batch))
        example = next(lines)
        x, (y, reconstruct) = example
        print(x)
        print(y)
        for x, y, reconstruct in zip(ids_to_python_string(x), ids_to_python_string(y),
                ids_to_python_string(reconstruct)):
            print(x.replace(chr(0), '_'))
            print(reconstruct)
            print(y)
            print('---------------------------------------')
        print('=======================================')

