import numpy as np
import tensorflow as tf

import data_pipe
import model_def

def setup():
    learn_rate = 1e-4
    dataset = data_pipe.file_to_dataset()
    # Get the batch size and sequence length by peeking at the first Dataset element
    batchsize, seqlen = next(iter(dataset))[0].shape
    model = model_def.make_model(batchsize, seqlen)
    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.summary()
    return dataset, model

def train(model, dataset):
    batchsize, seqlen = next(iter(dataset))[0].shape
    inference_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: model_def.run_inference(model, 'she', seqlen))
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./saved', save_freq=int(1e5))
    callbacks = [inference_callback, checkpoint]
    model.fit(dataset, epochs=200, verbose=1, steps_per_epoch=None, use_multiprocessing=True,
        callbacks=callbacks)

if '__main__' == __name__:
    dataset, model = setup()
    train(model, dataset)
    _, seqlen = next(iter(dataset))[0].shape
    model_def.run_inference(model, 'hi there', seqlen)

