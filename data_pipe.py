import tensorflow as tf

def file_to_dataset():
    batchsize = 8
    lines = tf.data.TextLineDataset(tf.constant('./traindata.txt'))
    lines = lines.map(string_to_ids)
    lines = lines.batch(batchsize, drop_remainder=True)
    # Peek at the first example to get the shape and set it explicitly
    batchsize, seqlen = next(iter(lines)).shape
    lines = lines.map(lambda line: tf.reshape(line, [batchsize, seqlen]))
    # Split x,y pair which are offset by one character
    lines = lines.map(lambda line: (line[:, :-1], line[:, 1:]))
    return lines

@tf.function
def string_to_ids(tf_string):
    result = tf.strings.bytes_split(tf_string, 'UTF-8')
    result = tf.strings.unicode_decode(result, 'UTF-8', errors='replace', replacement_char=0)
    result = tf.squeeze(result.to_tensor())
    return result

@tf.function
def ids_to_string(tensor):
    result = tf.strings.unicode_encode(tensor, 'UTF-8')
    return result

if __name__ == '__main__':
    lines = file_to_dataset()
    example = next(iter(lines))
    print(lines)
    print(example)
    x, y = example
    print(ids_to_string(x))
    print(ids_to_string(y))
