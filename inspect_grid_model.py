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

import numpy as np
import time
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

from train_scratch import load_dataset
import models
import third_party.utils as utils

grid_model_path = '/data/Projekte/spatial_navigation/sga2_spatial_nav/grid_model.h5'
visual_model_path = '/data/Projekte/spatial_navigation/sga2_spatial_nav/visual_model.h5'
dataset_path = '/data/Projekte/spatial_navigation/sga2_spatial_nav/train.tfrecords'

n_hd = 10
n_pc = 256
nh_lstm = 128
nh_bottleneck = 256

pc_ensemble = utils.get_place_cell_ensembles(
    env_size=2.2,
    neurons_seed=8341,
    targets_type='softmax',
    lstm_init_type='softmax',
    n_pc=[n_pc],
    pc_scale=[0.01]
)[0]

hd_ensemble = utils.get_head_direction_ensembles(
    neurons_seed=8341,
    targets_type='softmax',
    lstm_init_type='softmax',
    n_hdc=[n_hd],
    hdc_concentration=[20.]
)[0]

dataset = load_dataset(dataset_path)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
scatt = ax1.scatter([], [], cmap=plt.hsv)
scatt_true = ax3.scatter([], [], cmap=plt.hsv)
im = ax2.imshow(np.zeros((256, 256, 3)))
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax1.set_aspect('equal')
ax3.set_xlim(-1.1, 1.1)
ax3.set_ylim(-1.1, 1.1)
ax3.set_aspect('equal')

N = 1000000
xs = []
ys = []
arr = np.zeros((N, 2))
arr2 = np.zeros((N, 2))
colors = np.zeros((N, 4), dtype=np.float)
colors2 = np.zeros((N, 4), dtype=np.float)
cs = []
# i = 0

data_iter = iter(dataset)

# Params
img_width = 64
img_height = 64

# Load models
visual_model = models.VisualModel(
    img_width, img_height, pc_ensemble, hd_ensemble)
visual_model.load_weights(visual_model_path)


grid_model = models.GridNetworkModel(
    pc_ensemble, hd_ensemble, nh_lstm, nh_bottleneck)
grid_model.load_weights(grid_model_path)


def animate(frame):
    obs = next(data_iter)
    rgb = obs['rgb'].numpy().astype(int)
    target_pos = obs['target_pos'].numpy()
    target_rot = obs['target_rot'].numpy()
    ego_vel_trans = obs['ego_vel_trans'].numpy()
    ego_vel_rot = obs['ego_vel_rot'].numpy()
    arr[frame] = target_pos
    # xs.append(target_pos[0])
    # ys.append(target_pos[1])

    # Make predictions
    # rgb_i = np.swapaxes(rgb, 2, 4)
    # rgb_i = np.swapaxes(rgb_i, 2, 3)
    rgb = tf.transpose(rgb, [1, 2, 0])
    rgb = rgb[None, tf.newaxis, :]
    rgb = tf.cast(rgb, tf.float32)
    vis_pc_out, vis_hd_out = visual_model(rgb)

    pc_pred, hd_pred = grid_model.predict(
        [ego_vel_trans[None, tf.newaxis, :],
            ego_vel_rot[None, tf.newaxis, :], vis_pc_out, vis_hd_out]
    )

    # import ipdb; ipdb.set_trace()

    target_pos_aug = target_pos[tf.newaxis, tf.newaxis, :]
    target_rot_aug = target_rot[tf.newaxis, tf.newaxis, :]

    # print(rgb)
    # import ipdb; ipdb.set_trace()

    rgb = np.swapaxes(rgb, 0, 1)
    rgb = np.swapaxes(rgb, 1, 2)

    # Compute PC/HD activations
    pc_activations = pc_ensemble.get_targets(target_pos_aug)
    hd_activations = hd_ensemble.get_targets(target_rot_aug)

    # Plot Place cell activations along the driven path

    cmap = plt.get_cmap('hsv')
    # colors pred
    max_index = np.argmax(pc_pred)
    color = cmap(max_index)
    colors[frame] = color

    # colors true
    max_index2 = np.argmax(pc_activations)
    color2 = cmap(max_index2)
    colors2[frame] = color2

    scatt.set_offsets(arr[:frame])
    scatt.set_facecolors(colors)

    scatt_true.set_offsets(arr[:frame])
    scatt_true.set_facecolors(colors2)

    # Plot RGB
    im.set_data(rgb)
    return scatt, scatt_true, im


ani = FuncAnimation(fig, animate, frames=N, interval=30, blit=True)
plt.show()
