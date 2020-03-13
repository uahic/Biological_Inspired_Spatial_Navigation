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

import tensorflow as tf
import networks

# Input for grid network


class VisualModel(tf.keras.Model):
    def __init__(self, img_width, img_height, pc_ensemble, hd_ensemble, **kwargs):
        super(VisualModel, self).__init__(**kwargs)
        self.visual_module = networks.VisualModule(
            img_width, img_height, pc_ensemble, hd_ensemble)
        self.place_cells = tf.keras.layers.Dense(
            pc_ensemble.n_cells, activation='linear')
        self.head_direction_cells = tf.keras.layers.Dense(
            hd_ensemble.n_cells, activation='linear')

        # self.visual_module.build((None, img_width, img_height, 3))
        # self.build((None, img_width, img_height, 3))

    def call(self, rgb):
        vis_out = self.visual_module(rgb)
        pc_out = self.place_cells(vis_out)
        hd_out = self.head_direction_cells(vis_out)
        return pc_out, hd_out


class GridNetworkModel(tf.keras.Model):
    def __init__(self, pc_ensemble, hd_ensemble, nh_lstm, nh_bottleneck, **kwargs):
        super(GridNetworkModel, self).__init__(**kwargs)
        self.grid_network = networks.GridCellNetwork(
            pc_ensemble, hd_ensemble, nh_lstm, nh_bottleneck)
        self.nh_lstm = nh_lstm
        self.bottleneck_layer = self.grid_network.bottleneck

    # def call(self, ego_vel, rgb):
    def call(self, inputs, training=None):
        ego_vel = inputs[0]
        ego_rot = inputs[1]
        vis_pc_out = inputs[2]
        vis_hd_out = inputs[3]
        # rgb = x[0]
        # ego_vel = x[1]
        # ego_rots = x[2]
        # vis_pc_out, vis_hd_out = self.visual_model(rgb)
        vis_e = tf.keras.layers.Concatenate(axis=-1)([vis_pc_out, vis_hd_out])

        rnd_num = tf.random.uniform((1,), minval=0., maxval=1.)
        vis_e_masked = tf.cond(rnd_num < 0.05,
                               lambda: vis_e,
                               lambda: tf.zeros_like(vis_e)
                               )
        vis_e_masked = tf.stop_gradient(vis_e_masked)

        # import ipdb; ipdb.set_trace()
        # TODO check if this concatenation is valid
        grid_network_input = tf.concat(
            [ego_vel, ego_rot, vis_e_masked], axis=-1)

        batch_size = grid_network_input.shape[0]
        # initial_state = [cell_init_state_c, cell_init_state_h]
        pc_out, hd_out, bottleneck_out, _, _, _ = self.grid_network(
            grid_network_input, initial_state=None)

        return pc_out, hd_out
