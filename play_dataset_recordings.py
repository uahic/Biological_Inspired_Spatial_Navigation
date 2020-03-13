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

import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import numpy as np
import time

from train_scratch import load_dataset
from matplotlib.animation import FuncAnimation

import third_party.utils as utils

n_hd = 10
n_pc = 256
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


dataset_path = '/data/Projekte/spatial_navigation/sga2_spatial_nav/train.tfrecords'

dataset = load_dataset(dataset_path)


fig, (ax1, ax2) = plt.subplots(1, 2)
scatt = ax1.scatter([], [], cmap=plt.hsv)
im = ax2.imshow(np.zeros((256, 256, 3)))
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax1.set_aspect('equal')

N = 1000000
xs = []
ys = []
arr = np.zeros((N, 2))
colors = np.zeros((N, 4), dtype=np.float)
cs = []
# i = 0

data_iter = iter(dataset)
# line, = ax1.plot([], [], lw=2)


def animate(frame):
    # for obs in dataset:
    obs = next(data_iter)
    rgb = obs['rgb'].numpy().astype(int)
    target_pos = obs['target_pos'].numpy()
    target_rot = obs['target_rot'].numpy()
    arr[frame] = target_pos
    # xs.append(target_pos[0])
    # ys.append(target_pos[1])

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
    max_index = np.argmax(pc_activations)
    cmap = plt.get_cmap('hsv')
    color = cmap(max_index)
    colors[frame] = color

    scatt.set_offsets(arr[:frame])
    scatt.set_facecolors(colors)

    # Plot RGB
    im.set_data(rgb)
    return scatt, im


# fig.subplots_adjust(0, 0, 1, 1)
# ax.axis("off")
# image = ax.imshow()
ani = FuncAnimation(fig, animate, frames=N, interval=30, blit=True)
# import ipdb; ipdb.set_trace()

plt.show()
