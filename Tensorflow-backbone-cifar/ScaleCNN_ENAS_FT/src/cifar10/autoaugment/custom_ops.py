# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains convenience wrappers for typical Neural Network TensorFlow layers.

   Ops that have different behavior during training or eval have an is_training
   parameter.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

import src.global_args as global_args

arg_scope = tf.contrib.framework.arg_scope


def variable(name, shape, dtype, initializer, trainable):
    """Returns a TF variable with the passed in specifications."""
    var = tf.get_variable(
        name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        trainable=trainable)
    return var


def global_avg_pool(x, scope=None):
    """Average pools away spatial height and width dimension of 4D tensor."""
    assert x.get_shape().ndims == 4
    with tf.name_scope(scope, 'global_avg_pool', [x]):
        kernel_size = (1, int(x.shape[1]), int(x.shape[2]), 1)
        squeeze_dims = (1, 2)
        result = tf.nn.avg_pool(
            x,
            ksize=kernel_size,
            strides=(1, 1, 1, 1),
            padding='VALID',
            data_format='NHWC')
        return tf.squeeze(result, squeeze_dims)


def zero_pad(inputs, in_filter, out_filter):
    """Zero pads `input` tensor to have `out_filter` number of filters."""
    outputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],
                              [(out_filter - in_filter) // 2,
                               (out_filter - in_filter) // 2]])
    return outputs


@tf.contrib.framework.add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               is_training=True,
               reuse=None,
               scope=None):
    """Small wrapper around tf.contrib.layers.batch_norm."""
    return tf.contrib.layers.batch_norm(
        inputs,
        decay=decay,
        center=center,
        scale=scale,
        epsilon=epsilon,
        activation_fn=None,
        param_initializers=None,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        is_training=is_training,
        reuse=reuse,
        trainable=True,
        fused=True,
        data_format='NHWC',
        zero_debias_moving_mean=False,
        scope=scope)


def stride_arr(stride_h, stride_w):
    return [1, stride_h, stride_w, 1]


@tf.contrib.framework.add_arg_scope
def conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           scope=None,
           reuse=None):
    """Adds a 2D convolution.
  
    conv2d creates a variable called 'weights', representing the convolutional
    kernel, that is convolved with the input.
  
    Args:
      inputs: a 4D tensor in NHWC format.
      num_filters_out: the number of output filters.
      kernel_size: an int specifying the kernel height and width size.
      stride: an int specifying the height and width stride.
      scope: Optional scope for variable_scope.
      reuse: whether or not the layer and its variables should be reused.
    Returns:
      a tensor that is the result of a convolution being applied to `inputs`.
    """
    with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
        num_filters_in = int(inputs.shape[3])
        weights_shape = [kernel_size, kernel_size, num_filters_in, num_filters_out]

        # Initialization
        n = int(weights_shape[0] * weights_shape[1] * weights_shape[3])
        weights_initializer = tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / n))

        weights = variable(
            name='weights',
            shape=weights_shape,
            dtype=tf.float32,
            initializer=weights_initializer,
            trainable=True)
        strides = stride_arr(stride, stride)
        outputs = tf.nn.conv2d(
            inputs, weights, strides, padding='SAME', data_format='NHWC')
        return outputs

@tf.contrib.framework.add_arg_scope
def conv2d_with_concrete_dropout(
        inputs,
        num_filters_out,
        kernel_size,
        stride=1,
        scope=None,
        reuse=None,
        init_min=0.1,
        init_max=0.1,
        weight_regularizer_ratio=global_args.WEIGHT_REGULARIZER_RATIO,
        dropout_regularizer_ratio=global_args.DROPOUT_REGULARIZER_RATIO,
        determinist=True):

    weight_regularizer_ratio = weight_regularizer_ratio[0] / 5e5
    dropout_regularizer_ratio = dropout_regularizer_ratio[0] / 5e5

    with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
        # Initializing weight
        input_shape = inputs.shape.as_list()
        num_filters_in = int(input_shape[3])
        weights_shape = [kernel_size, kernel_size, num_filters_in, num_filters_out]

        # Initialization
        n = int(weights_shape[0] * weights_shape[1] * weights_shape[3])
        weights_initializer = tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / n))

        # print(in_channels)
        weights = variable(
            name='weights',
            shape=weights_shape,
            dtype=tf.float32,
            initializer=weights_initializer,
            trainable=True)
        strides = stride_arr(stride, stride)
        # Initializing dropout
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        assert len(input_shape) == 4, 'only support conv2D layers'

        p_logit = tf.get_variable('dropout_p_logit', (1,),
                                  initializer=tf.initializers.random_uniform(minval=init_min, maxval=init_max),
                                  trainable=True)
        p = tf.sigmoid(p_logit)
        tf.add_to_collection(name='dropout_p', value=p)

        noise_shape = (tf.shape(inputs)[0], 1, 1, num_filters_in)

        weight_regularizer = weight_regularizer_ratio * tf.reduce_sum(tf.square(weights)) * (1. - p)
        dropout_regularizer = p * tf.log(p)
        dropout_regularizer += (1. - p) * tf.log(1. - p)
        dropout_regularizer *= dropout_regularizer_ratio * num_filters_in

        regularizer = tf.reduce_sum(weight_regularizer + dropout_regularizer)

        tf.add_to_collection(name='concrete_dropout_weight_regularizer', value=regularizer)
        ''' 
        # Conducting dropout
        eps = 1e-8
        temp = 1. / 5.

        if determinist:
            print('Using determinist drop.')
            unif_noise_var = tf.get_variable('unif_noise',
                                             initializer=tf.random_uniform(shape=[1] + list(noise_shape[1:])),
                                             trainable=False)
            drop_tensor = tf.floor(unif_noise_var + p)
            random_tensor = 1. - drop_tensor
        else:
            # Gumble
            print(
                "use Gumble in conv with weight regularizer ratio: {}, dropout regularizer ratio: {}".format
                (weight_regularizer_ratio, dropout_regularizer_ratio)
            )
            unif_noise_1 = tf.random_uniform(shape=noise_shape)
            unif_noise_2 = tf.random_uniform(shape=noise_shape)

            drop_prob = (
                    tf.log(p + eps)
                    - tf.log(1. - p + eps)
                    + tf.log(-tf.log(unif_noise_1 + eps) + eps)
                    - tf.log(-tf.log(unif_noise_2 + eps) + eps)
            )

            drop_prob = tf.sigmoid(drop_prob / temp)
            random_tensor = 1. - drop_prob

        inputs *= random_tensor
        '''
        inputs = tf.nn.dropout(x=inputs, keep_prob=0.9, noise_shape=noise_shape)

        return tf.nn.conv2d(
            inputs, weights, strides, padding='SAME', data_format='NHWC')


@tf.contrib.framework.add_arg_scope
def fc(inputs,
       num_units_out,
       scope=None,
       reuse=None):
    """Creates a fully connected layer applied to `inputs`.

    Args:
      inputs: a tensor that the fully connected layer will be applied to. It
        will be reshaped if it is not 2D.
      num_units_out: the number of output units in the layer.
      scope: Optional scope for variable_scope.
      reuse: whether or not the layer and its variables should be reused.

    Returns:
       a tensor that is the result of applying a linear matrix to `inputs`.
    """
    if len(inputs.shape) > 2:
        inputs = tf.reshape(inputs, [int(inputs.shape[0]), -1])

    with tf.variable_scope(scope, 'FC', [inputs], reuse=reuse):
        num_units_in = inputs.shape.as_list()[1]
        weights_shape = [num_units_in, num_units_out]
        unif_init_range = 1.0 / (num_units_out)**(0.5)
        weights_initializer = tf.random_uniform_initializer(
            -unif_init_range, unif_init_range)
        weights = variable(
            name='weights',
            shape=weights_shape,
            dtype=tf.float32,
            initializer=weights_initializer,
            trainable=True)
        bias_initializer = tf.constant_initializer(0.0)
        biases = variable(
            name='biases',
            shape=[num_units_out,],
            dtype=tf.float32,
            initializer=bias_initializer,
            trainable=True)
        outputs = tf.nn.xw_plus_b(inputs, weights, biases)
        return outputs


@tf.contrib.framework.add_arg_scope
def fc_with_concrete_dropout(inputs,
                             num_units_out,
                             scope=None,
                             reuse=None,
                             init_min=0.1,
                             init_max=0.1,
                             weight_regularizer_ratio=global_args.WEIGHT_REGULARIZER_RATIO,
                             dropout_regularizer_ratio=global_args.DROPOUT_REGULARIZER_RATIO,
                             determinist=True):
    """Creates a fully connected layer applied to `inputs`.

    Args:
      inputs: a tensor that the fully connected layer will be applied to. It
        will be reshaped if it is not 2D.
      num_units_out: the number of output units in the layer.
      scope: Optional scope for variable_scope.
      reuse: whether or not the layer and its variables should be reused.

    Returns:
       a tensor that is the result of applying a linear matrix to `inputs`.
    """
    weight_regularizer_ratio = weight_regularizer_ratio[0] / 5e5
    dropout_regularizer_ratio = dropout_regularizer_ratio[0] / 5e5

    if len(inputs.shape) > 2:
        inputs = tf.reshape(inputs, [int(inputs.shape[0]), -1])

    with tf.variable_scope(scope, 'FC', [inputs], reuse=reuse):
        num_units_in = inputs.shape.as_list()[1]
        weights_shape = [num_units_in, num_units_out]
        unif_init_range = 1.0 / (num_units_out) ** (0.5)
        weights_initializer = tf.random_uniform_initializer(
            -unif_init_range, unif_init_range)
        weights = variable(
            name='weights',
            shape=weights_shape,
            dtype=tf.float32,
            initializer=weights_initializer,
            trainable=True)
        bias_initializer = tf.constant_initializer(0.0)
        biases = variable(
            name='biases',
            shape=[num_units_out, ],
            dtype=tf.float32,
            initializer=bias_initializer,
            trainable=True)

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        assert len(inputs.shape) == 2, 'only support Fc layers'

        p_logit = tf.get_variable('dropout_p_logit', (1,),
                                  initializer=tf.initializers.random_uniform(minval=init_min, maxval=init_max),
                                  trainable=True)
        p = tf.sigmoid(p_logit)
        tf.add_to_collection(name='dropout_p', value=p)

        noise_shape = (tf.shape(inputs)[0], num_units_in)

        weight_regularizer = weight_regularizer_ratio * tf.reduce_sum(tf.square(weights)) * (1. - p)
        dropout_regularizer = p * tf.log(p)
        dropout_regularizer += (1. - p) * tf.log(1. - p)
        dropout_regularizer *= dropout_regularizer_ratio * num_units_in

        regularizer = tf.reduce_sum(weight_regularizer + dropout_regularizer)

        tf.add_to_collection(name='concrete_dropout_weight_regularizer', value=regularizer)
        '''
        # Conducting dropout
        eps = 1e-8
        temp = 1. / 5.
        if determinist:
            print('Using determinist drop.')
            unif_noise_var = tf.get_variable('unif_noise',
                                             initializer=tf.random_uniform(shape=[1] + list(noise_shape[1:])),
                                             trainable=False)
            drop_tensor = tf.floor(unif_noise_var + p)
            random_tensor = 1. - drop_tensor
        else:
            # Gumble
            print(
                "use Gumble in conv with weight regularizer ratio: {}, dropout regularizer ratio: {}".format
                (weight_regularizer_ratio, dropout_regularizer_ratio)
            )
            unif_noise_1 = tf.random_uniform(shape=noise_shape)
            unif_noise_2 = tf.random_uniform(shape=noise_shape)

            drop_prob = (
                    tf.log(p + eps)
                    - tf.log(1. - p + eps)
                    + tf.log(-tf.log(unif_noise_1 + eps) + eps)
                    - tf.log(-tf.log(unif_noise_2 + eps) + eps)
            )

            drop_prob = tf.sigmoid(drop_prob / temp)
            random_tensor = 1. - drop_prob

        inputs *= random_tensor
        '''
        inputs = tf.nn.dropout(x=inputs, keep_prob=0.9)

        outputs = tf.nn.xw_plus_b(inputs, weights, biases)
        return outputs


@tf.contrib.framework.add_arg_scope
def avg_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
    """Wrapper around tf.nn.avg_pool."""
    with tf.name_scope(scope, 'AvgPool', [inputs]):
        kernel = stride_arr(kernel_size, kernel_size)
        strides = stride_arr(stride, stride)
        return tf.nn.avg_pool(
            inputs,
            ksize=kernel,
            strides=strides,
            padding=padding,
            data_format='NHWC')

