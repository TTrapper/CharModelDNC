import argparse

import numpy as np
import tensorflow as tf

import data_pipe
import model_def

parser = argparse.ArgumentParser()
parser.add_argument('--restore', type=str, choices=['yes', 'no'], default='no')

def parseargs(parser):
    args = parser.parse_args()
    args.restore = args.restore == 'yes'
    return args

def setup(restore):
    learn_rate = 1e-4
    dataset = data_pipe.file_to_dataset()
    if restore:
        model = tf.keras.models.load_model('./model.hd5', compile=True,
            custom_objects={'DNCCell':model_def.DNCCell})
    else:
        model = model_def.make_model()
        optimizer = tf.keras.optimizers.Adam(learn_rate)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.summary()
    return dataset, model

def train(model, dataset):
    batchsize, seqlen = next(iter(dataset))[0].shape
    def inference_fn(batch, logs):
        if batch % 2000 == 1999:
            model_def.run_inference(model, 'she', seqlen)
            model.save('./model.hd5', save_format='h5', overwrite=True, include_optimizer=True)
    inference_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=inference_fn)
    callbacks = [inference_callback]
    model.fit(dataset, epochs=200, verbose=1, steps_per_epoch=None, use_multiprocessing=True,
        callbacks=callbacks)

if '__main__' == __name__:
    args = parseargs(parser)
    dataset, model = setup(args.restore)
    train(model, dataset)
    _, seqlen = next(iter(dataset))[0].shape
    model_def.run_inference(model, 'hi there', seqlen)

