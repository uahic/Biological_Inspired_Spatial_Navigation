# Copyright 2020 FZI Forschungszentrum Informatik 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy
import tensorflow as tf

from tensorflow import keras
import third_party.utils as utils

# from tensorflow.keras import Model

# Keras attempt


def displaced_linear_initializer(input_size, displace, dtype=tf.float32):
    """
    Used to prevent initialized weights from beeing more than 2 standard deviations away from the mean value
    """
    stddev = 1. / numpy.sqrt(input_size)
    return keras.initializers.TruncatedNormal(
        mean=displace*stddev, stddev=stddev, seed=None)


class GridCellInitLayer(keras.layers.Layer):
    def __init__(self, nh_lstm, pc_ensemble, hd_ensemble, **kwargs):
        super(GridCellInitLayer, self).__init__(**kwargs)

        self.pc_ensemble = pc_ensemble
        self.hd_ensemble = hd_ensemble
        self.state_weights = tf.keras.layers.Dense(
            nh_lstm, name="state_init")
        self.cell_weights = tf.keras.layers.Dense(
            nh_lstm, name="cell_init")

    def call(self, initial_activations):
        init_pos = initial_activations[0]
        init_hd = initial_activations[1]
        init_conds = utils.encode_initial_conditions(
            init_pos, init_hd, [self.pc_ensemble], [self.hd_ensemble])

        concat_init = tf.concat(init_conds, axis=0)
        init_state = self.state_weights(concat_init)
        init_cell = self.cell_weights(concat_init)
        initial_state = [init_state, init_cell]
        return initial_state


class GridCellNetwork(keras.layers.Layer):
    def __init__(self,
                 pc_ensemble,
                 hd_ensemble,
                 nh_lstm,
                 nh_bottleneck,
                 nh_embed=None,
                 dropoutrates_bottleneck=None,
                 bottleneck_weight_decay=0.0,
                 bottleneck_has_bias=False,
                 init_weight_disp=0.0,
                 use_kernel_regularizers=True):
        super(GridCellNetwork, self).__init__()

        # Params
        self.pc_ensemble = pc_ensemble
        self.hd_ensemble = hd_ensemble
        self._nh_embed = nh_embed
        self._nh_lstm = nh_lstm
        self._nh_bottleneck = nh_bottleneck
        self._dropoutrates_bottleneck = dropoutrates_bottleneck
        self._bottleneck_weight_decay = bottleneck_weight_decay
        self._bottleneck_has_bias = bottleneck_has_bias
        self._init_weight_disp = init_weight_disp

        # Topology
        self.dropout = keras.layers.Dropout(0.5)

        # return_sequences=True => returns hidden_state for each time_step (frame) of the input
        # return_state=True => return the cell_state: [latest hidden_state, hidden_state, cell_state]
        # if both are True => [hidden_states (for each frame), latest hidden_state, cell_state]
        self.lstm = keras.layers.LSTM(
            nh_lstm,
            # input_shape=(3,),
            stateful=False, return_state=True, return_sequences=True
            # kernel_initializer='glorot_uniform',
            # recurrent_initializer='orthogonal',
            # bias_initializer='zeros'
        )

        def get_kernel_regularizer():
            if use_kernel_regularizers:
                return keras.regularizers.l2(self._bottleneck_weight_decay)
            else:
                return None

        def get_kernel_initializer():
            return displaced_linear_initializer(self._nh_bottleneck, self._init_weight_disp)
        # ... := time_steps
        # Input shape: (batch_size, ..., input_dim)
        # Output shape: (batch_size, ..., units)
        self.bottleneck = keras.layers.Dense(self._nh_bottleneck,
                                             activation='linear',
                                             #  input_shape=(nh_lstm,),
                                             use_bias=self._bottleneck_has_bias,
                                             kernel_regularizer=get_kernel_regularizer()
                                             )
        self.place_cells = keras.layers.Dense(
            self.pc_ensemble.n_cells,
            # input_shape=(nh_bottleneck,),
            activation='linear',
            kernel_regularizer=get_kernel_regularizer(),
            kernel_initializer=get_kernel_initializer(),
            name='pc_out')
        self.head_direction_cells = keras.layers.Dense(
            self.hd_ensemble.n_cells,
            # input_shape=(nh_bottleneck,),
            activation='linear',
            kernel_regularizer=get_kernel_regularizer(),
            kernel_initializer=get_kernel_initializer(),
            name='hd_out')

        self.prev_state = None

    def call(self, x, initial_state=None, training=None):
        """
        inputs := [data tensor, [hidden_state, cell_state]]
        """
        # tf.print('initial state ', initial_state)
        lstm_out, hidden_state, cell_state = self.lstm(
            x, initial_state=initial_state)
        bottleneck_out = self.bottleneck(lstm_out)

        if training:
            bottleneck_out = self.dropout(bottleneck_out)

        pc_out = self.place_cells(bottleneck_out)
        hd_out = self.head_direction_cells(bottleneck_out)
        return pc_out, hd_out, bottleneck_out, lstm_out, hidden_state, cell_state


class VisualModule(keras.layers.Layer):
    def __init__(self,  img_width, img_height, pc_ensemble, hd_ensemble, name='visualModule', **kwargs):
        super(VisualModule, self).__init__(name=name, **kwargs)
        self.img_width = img_width
        self.img_height = img_height

    def build(self, input_shape):
        tf.print('Input shape ', input_shape)
        kernel_shape = (5, 5)
        strides = (2, 2)
        # same := adds zero padding around the input and the output, in turn, does not shrink
        padding = 'same'
        activation = 'relu'
        self.cl_1 = keras.layers.TimeDistributed(
            keras.layers.Conv2D(
                16, kernel_shape,  strides=strides, padding=padding, activation=activation,
            ),
            # input_shape=(None, img_width, img_height, 3)
            input_shape=input_shape
        )
        self.cl_2 = keras.layers.TimeDistributed(
            keras.layers.Conv2D(
                32, kernel_shape, strides=strides, padding=padding, activation=activation,
            ),
        )
        self.cl_3 = keras.layers.TimeDistributed(
            keras.layers.Conv2D(
                64, kernel_shape, strides=strides, padding=padding, activation=activation,
            ),
        )
        self.flatten = keras.layers.TimeDistributed(
            keras.layers.Flatten(),
        )
        self.dense = keras.layers.Dense(256)

    def call(self, rgb_input):
        cl_1_out = self.cl_1(rgb_input)
        cl_2_out = self.cl_2(cl_1_out)
        cl_3_out = self.cl_3(cl_2_out)
        conv_flatten = self.flatten(cl_3_out)
        dense_out = self.dense(conv_flatten)
        return dense_out


class ActorCriticNetwork(keras.layers.Layer):
    def __init__(self, img_width, img_height, pc_ensemble, hd_ensemble, **kwargs):
        super(ActorCriticNetwork, self).__init__(**kwargs)
        self.lstm = keras.layers.LSTM(256)
        self.critic = keras.layers.Dense(6)
        self.actor = keras.layers.Dense(6)
        self.visual_module = VisualModule(
            img_width, img_height, pc_ensemble, hd_ensemble)

    def call(self, rgb, grid_goal, grid_cur, reward, last_action):

        e = self.visual_module(rgb)
        concat_input = tf.concat(
            grid_goal, grid_cur, e, reward, last_action, axis=1)
        lstm_out = self.lstm(concat_input)
        critic_out = self.critic(lstm_out)
        actor_out = self.actor(lstm_out)
        return critic_out, actor_out
