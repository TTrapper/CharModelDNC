import tensorflow as tf

def file_to_dataset():
    batchsize = 8
    maxseqlen = 32
    lines = tf.data.TextLineDataset(tf.constant('./traindata.txt'))
    lines = lines.map(string_to_ids)
    lines = lines.batch(batchsize, drop_remainder=True)
    # Peek at the first example to get the shape and set it explicitly
    batchsize, seqlen = next(iter(lines)).shape
    lines = lines.map(lambda line: tf.reshape(line, [batchsize, seqlen]))
    # Add a GO byte to the beginning of the sequence.
    lines = lines.map(lambda line: tf.concat([2*tf.ones([batchsize, 1], line.dtype), line], axis=1))
    # Split x,y pair which are offset by one character
    lines = lines.map(lambda line: (line[:, :-1], line[:, 1:]))
    # Split long lines into consecutive batches which can be trained with stateful RNNs
    if seqlen > maxseqlen:
        if seqlen % maxseqlen != 0:
            raise ValueError('The maxseqlen must evenly divide the length of the lines in the input '
                'file, got maxseqlen:{} seqlen:{}'.format(maxseqlen, seqlen))
        nsplits = seqlen // maxseqlen
        lines = lines.map(lambda x,y: (tf.split(x, nsplits, axis=1), tf.split(y, nsplits, axis=1)))
        lines = lines.map(lambda x,y: (tf.reshape(x, [-1, maxseqlen]), tf.reshape(y, [-1, maxseqlen])))
        lines = lines.unbatch()
        lines = lines.batch(batchsize, drop_remainder=True)
    return lines

#@tf.function
def string_to_ids(tf_string):
    result = tf.strings.bytes_split(tf_string, 'UTF-8')
    result = tf.strings.unicode_decode(result, 'UTF-8', errors='replace', replacement_char=0)
    result = tf.squeeze(result.to_tensor())
    return result

#@tf.function
def ids_to_string(tensor):
    result = tf.strings.unicode_encode(tensor, 'UTF-8')
    return result

if __name__ == '__main__':
    lines = file_to_dataset()
    print(lines)
    lines = iter(lines)
    for batch in range(4):
        print('\nbatch: {}'.format(batch))
        example = next(lines)
        x, y = example
        print(x)
        for line in ids_to_string(x):
            print(line.numpy())
        print(y)
        for line in ids_to_string(y):
            print(line.numpy())


