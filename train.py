import argparse

import numpy as np
import tensorflow as tf

import data_pipe
import model_def

parser = argparse.ArgumentParser()
parser.add_argument('--restore', action='store_true')

def parseargs(parser):
    args = parser.parse_args()
    return args

def setup(restore):
    learn_rate = 1e-4
    dataset = data_pipe.file_to_dataset()
    if restore:
        model = tf.keras.models.load_model('./model.h5', compile=True,
            custom_objects={'DNCCell':model_def.DNCCell})
    else:
        batchsize = dataset.element_spec[0].shape[0]
        model = model_def.make_model(batchsize)
        optimizer = tf.keras.optimizers.Adam(learn_rate)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.summary()
    return dataset, model

def train(model, dataset):
    def inference_fn(batch, logs):
        if batch % 20000 == 200:
            model.save('./model.h5', save_format='h5', overwrite=True, include_optimizer=True)
            for softmax_temp in [1e-16, 0.5, 0.75]:
                model_def.run_inference(model, 'she', 128, softmax_temp)
    inference_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=inference_fn)
    callbacks = [inference_callback]
    model.fit(dataset, epochs=200, verbose=1, steps_per_epoch=None, use_multiprocessing=True,
        callbacks=callbacks)

if '__main__' == __name__:
    args = parseargs(parser)
    dataset, model = setup(args.restore)
    train(model, dataset)
