import numpy as np
import tensorflow as tf

import functools

from src import global_args

def CD_decorator(func):
    functools.wraps(func)
    def ConcreteDrop_wrapper(*args, **kwargs):
        new_variable = func(*args, **kwargs)

        if kwargs['enable_dropout']:
            weight_regularizer_ratio = global_args.WEIGHT_REGULARIZER_RATIO[0] / 5e5
            dropout_regularizer_ratio = global_args.DROPOUT_REGULARIZER_RATIO[0] / 5e5
            print(
                "use Gumble in conv with weight regularizer ratio: {}, dropout regularizer ratio: {}".format
                (weight_regularizer_ratio, dropout_regularizer_ratio)
            )

            _shape = kwargs['shape'] if 'shape' in kwargs else args[1]
            _name = kwargs['name'] if 'name' in kwargs else args[0]

            in_channels = _shape[2] if len(_shape) == 4 else _shape[0]

            # Initializing dropout
            init_min = np.log(0.5) - np.log(1. - 0.5)
            init_max = np.log(0.5) - np.log(1. - 0.5)
            with tf.device("/cpu:0"):
                p_logit = tf.get_variable(_name+'_dropout_p_logit', (1,),
                                          initializer=tf.initializers.random_uniform(minval=init_min, maxval=init_max),
                                          trainable=True)
            p = tf.sigmoid(p_logit)
            tf.add_to_collection(name='dropout_p', value=p)

            weight_regularizer = weight_regularizer_ratio * tf.reduce_sum(tf.square(new_variable)) * (1. - p)
            dropout_regularizer = p * tf.log(p)
            dropout_regularizer += (1. - p) * tf.log(1. - p)
            dropout_regularizer *= dropout_regularizer_ratio * in_channels

            regularizer = tf.reduce_sum(weight_regularizer + dropout_regularizer)

            tf.add_to_collection(name='concrete_dropout_weight_regularizer', value=regularizer)
        else:
            p = None

        return new_variable, p
    return ConcreteDrop_wrapper


def lstm(x, prev_c, prev_h, w):
    ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)
    i, f, o, g = tf.split(ifog, 4, axis=1)
    i = tf.sigmoid(i)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    g = tf.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * tf.tanh(next_c)
    return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h

@CD_decorator
def create_weight(name, shape, initializer=None, trainable=True, seed=None, enable_dropout=True):
    if initializer is None:
        initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
    with tf.device("/cpu:0"):
        w = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return w


def create_bias(name, shape, initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0, dtype=tf.float32)
    with tf.device("/cpu:0"):
        b = tf.get_variable(name, shape, initializer=initializer)
    return b

