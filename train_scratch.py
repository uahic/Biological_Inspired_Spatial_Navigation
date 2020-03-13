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

import datetime
import logging
import os
import sys
import pathlib

from absl import app, flags
import numpy as np
import tensorflow as tf
import deepmind_lab

import agents
import networks
import models
import read_tfrecords
import buffers

import third_party.scores as scores
import third_party.ensembles as ensembles
import third_party.utils as utils


def softmax_cross_entropy_logits_loss(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred)
    return loss


def add_input_noise(ego_vel):
    ego_vel_noise = tf.random.normal(
        ego_vel.shape, 0.0, 1.0) * velocity_noise
    return ego_vel + ego_vel_noise


# TFRecords helper functions
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values.reshape(-1)))


logging.getLogger("tensorflow").setLevel(logging.ERROR)

# FLAGS = flags.FLAGS

# flags.DEFINE_float('task_env_size', 2.2,
#                    'Environment size (meters).')

# General
datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('/tmp', 'tensorboard', 'gridnetwork', datetime)
train_log_dir = os.path.join(log_dir, 'train')
test_log_dir = os.path.join(log_dir, 'test')
pathlib.Path(train_log_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(test_log_dir).mkdir(parents=True, exist_ok=True)


# DM-Lab Observations
img_width = 64
img_height = 64
max_velocity = 325.0
observations = ['RGB',
                'DEBUG.POS.ROT', 'DEBUG.POS.TRANS',
                'VEL.ROT', 'VEL.TRANS']

# DM-Lab Environment
level = 'tests/empty_room_test'
level_boundary_min = np.array([116.125, 116.125])
level_boundary_max = np.array([783.87, 783.87])
lab_config = {'width': str(img_width), 'height': str(
    img_height), 'fps': '30'}
env = deepmind_lab.Lab(level, observations,
                       config=lab_config, renderer='hardware')

# Replay memory
# memory_slots = int(1e7)

# Dataset
dataset_path = '/data/Projekte/spatial_navigation/sga2_spatial_nav/train.tfrecords'
dataset_size = int(1e7)

# Network parameters
n_hd = 10
n_pc = 256
nh_lstm = 128
nh_bottleneck = 256

# Train parameters
frac_train = 0.7
batch_size_eval = 4000
eval_frequency = 2
velocity_noise = [0.0, 0.0, 0.0]
sequence_length = 100

# Shared training params

# Grid training params
grid_params = {
    'batch_size': 10,
    'steps_per_epoch': dataset_size // 10,
    'epochs': 1
}

# Visual training params
visual_params = {
    'batch_size': 32,
    'steps_per_epoch': dataset_size // 32,
    'epochs': 1
}

# We keep the seed fixed to allow for reproducability
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

# General callbacks
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def load_model_weights_from_checkpoint(model, checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_ckpt:
        # tf.print('Found checkpoint for model {}. Loading weights...'.format(model))
        model.load_weights(latest_ckpt)
        initial_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
    else:
        tf.print(
            'Did not found any checkpoint for model {}. Continuing.'.format(model))
        initial_epoch = 0

    return initial_epoch


def train_visual_module(model, dataset, ckpt_path=None, initial_epoch=0):
    def batch_generator(batch_size=32):
        batched_dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(
            buffer_size=1000).repeat()

        for batch in batched_dataset:
            rgb = batch['rgb'][:, tf.newaxis, :]
            target_pos = batch['target_pos'][:, tf.newaxis, :]
            target_rot = batch['target_rot'][:, tf.newaxis, :]
            targets = utils.encode_targets(target_pos, target_rot, [
                pc_ensemble], [hd_ensemble])

            rgb = np.swapaxes(rgb, 2, 4)
            rgb = np.swapaxes(rgb, 2, 3)
            rgb = tf.convert_to_tensor(rgb)
            yield (rgb), (targets[0], targets[1])


    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        verbose=1,
        save_weights_only=True,
        period=1
    )

    optimizer = tf.optimizers.RMSprop(
        learning_rate=1e-5,
        momentum=0.9,
        clipvalue=1e-5
    )

    model.compile(
        optimizer=optimizer,
        loss={
            'output_1': softmax_cross_entropy_logits_loss,
            'output_2': softmax_cross_entropy_logits_loss
        }
    )

    # TODO add validation generator
    model.fit_generator(
        batch_generator(batch_size=visual_params['batch_size']),
        epochs=visual_params['epochs'],
        steps_per_epoch=visual_params['steps_per_epoch'],
        verbose=1,
        callbacks=[
            tensorboard_cb,
            ckpt_cb
        ],
        initial_epoch=initial_epoch
    )

    return model


def train_grid_module(model, visual_model, dataset, ckpt_path=None, initial_epoch=0):
    def batch_generator(batch_size=10):
        batched_dataset = dataset.batch(
            batch_size, drop_remainder=True).shuffle(buffer_size=1000).repeat()
        for batch in batched_dataset:
            rgb = batch['rgb'][:, tf.newaxis, :]
            target_pos = batch['target_pos'][:, tf.newaxis, :]
            target_rot = batch['target_rot'][:, tf.newaxis, :]
            ego_vel_trans = batch['ego_vel_trans'][:, tf.newaxis, :]
            ego_vel_rot = batch['ego_vel_rot'][:, tf.newaxis, :]
            targets = utils.encode_targets(target_pos, target_rot, [
                pc_ensemble], [hd_ensemble])

            rgb = np.swapaxes(rgb, 2, 4)
            rgb = np.swapaxes(rgb, 2, 3)
            rgb = tf.convert_to_tensor(rgb)
            vis_pc_out, vis_hd_out = visual_model(rgb)
            yield (ego_vel_trans, ego_vel_rot, vis_pc_out, vis_hd_out), (targets[0], targets[1])

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        verbose=1,
        save_weights_only=True,
        period=1
    )

    optimizer = tf.optimizers.RMSprop(
        learning_rate=1e-5,
        momentum=0.9,
        clipvalue=1e-5
    )

    model.compile(
        optimizer=optimizer,
        loss={'output_1': softmax_cross_entropy_logits_loss,
              'output_2': softmax_cross_entropy_logits_loss,
              }
    )

    model.fit_generator(
        batch_generator(batch_size=grid_params['batch_size']),
        epochs=grid_params['epochs'],
        steps_per_epoch=grid_params['steps_per_epoch'],
        verbose=1,
        callbacks=[
            tensorboard_cb,
            ckpt_cb
        ],
        initial_epoch=initial_epoch
    )
    return model

def load_dataset(path):
    raw_dataset = tf.data.TFRecordDataset(path)

    feature = {
        'target_pos_original': tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),
        'target_pos': tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),
        'target_rot': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
        'ego_vel_trans': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
        'ego_vel_rot': tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),
        'rgb': tf.io.FixedLenFeature(shape=[1], dtype=tf.string)
    }

    def _parse_example(example_proto):
        example = tf.io.parse_single_example(example_proto, feature)
        rgb_uint8 = tf.io.decode_raw(example['rgb'], tf.uint8)
        rgb_float32 = tf.dtypes.cast(rgb_uint8, tf.float32)
        example['rgb'] = tf.reshape(rgb_float32, (3, img_width, img_height))
        return example

    parsed_dataset = raw_dataset.map(_parse_example)
    return parsed_dataset


def generate_dataset(agent, path):
    pc_boundary_scale = (pc_ensemble.pos_max - pc_ensemble.pos_min) / \
        (level_boundary_max - level_boundary_min)

    level_boundary_mean = (
        level_boundary_max - level_boundary_min)/2. + level_boundary_min

    last_target_rot = 0.0
    env.reset()


    with tf.io.TFRecordWriter(path) as writer:
        for i in range(dataset_size):
            if i % 10000 == 0:
                tf.print('Iteration {}'.format(i))

            if not env.is_running():
                print("Environment stopped early")
                env.reset()
                agent.reset()
                last_ego_rot = 0.0

            obs = env.observations()
            if not obs:
                raise Exception("Observations empty")

            rgb = obs['RGB']
            target_pos = obs['DEBUG.POS.TRANS'][:2]
            target_rot = obs['DEBUG.POS.ROT'][1]
            target_pos -= level_boundary_mean
            target_pos *= pc_boundary_scale
            target_rot = target_rot * ((2. * np.pi)/360.)

            # tf.print(obs['DEBUG.POS.TRANS'], obs['DEBUG.POS.ROT'], obs['VEL.TRANS'], obs['VEL.ROT'])
            # Angular velocity
            ego_vel_rot = (obs['VEL.ROT'][1] * np.pi * 2.) / 360.
            # ego_vel_rot = (target_rot - last_target_rot)
            # last_target_rot = target_rot

            # tf.print('rot-velocity in angles', obs['VEL.ROT'][1])
            # tf.print('rot-velocity in radiants', ego_vel_rot)
            # tf.print('cos(rad)', np.cos(ego_vel_rot))
            # Translation vectors are relative to global coordinate system
            ego_vel_trans = obs['VEL.TRANS'][:2] / max_velocity

            ego_vel_trans = np.linalg.norm(ego_vel_trans)
            ego_vel_rot_cos = np.array([np.cos(ego_vel_rot)])
            ego_vel_rot_sin = np.array([np.sin(ego_vel_rot)])
            ego_vel_rot = np.concatenate((ego_vel_rot_cos, ego_vel_rot_sin))

            feature = {
                'target_pos_original': _float_feature(obs['DEBUG.POS.TRANS'][:2]),
                'target_pos': _float_feature(target_pos),
                'target_rot': _float_feature(target_rot),
                'ego_vel_trans': _float_feature(ego_vel_trans),
                'ego_vel_rot': _float_feature(ego_vel_rot),
                'rgb': _bytes_feature(tf.compat.as_bytes(rgb.tostring()))
            }

            action = agent.step(ego_vel_trans, ego_vel_rot)
            env.step(action, num_steps=1)

            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)
    tf.print('Created dataset successfully')


if __name__ == "__main__":
    # Dummy random agent for generating initial episodes
    agent = agents.GaussRandomAgent(
        env.action_spec(),
        forbidden_actions=['JUMP', 'FIRE', 'CROUCH',
                           'LOOK_DOWN_UP_PIXELS_PER_FRAME']
    )

    # replay_buffer = fill_replay_buffer(agent)

    # Checkpoints
    vis_ckpt_path = 'checkpoints/vis/model-{epoch:04d}.ckpt'
    grid_ckpt_path = 'checkpoints/grid/model-{epoch:04d}.ckpt'
    pathlib.Path('checkpoints/vis').mkdir(parents=True, exist_ok=True)
    pathlib.Path('checkpoints/grid').mkdir(parents=True, exist_ok=True)

    def confirmation_prompt(question):
        yes_choices = {'yes', 'y', 'ye'}
        no_choices = {'no', 'n'}
        while True:
            choice = input("{} [y]es/[n]o ".format(question)).lower()
            if choice in yes_choices:
                return True
            elif choice in no_choices:
                return False
            else:
                tf.print('This is a yes or no question!')

    do_generate_dataset = confirmation_prompt('Generate new dataset? ')
    do_train_visual_model = confirmation_prompt("Train visual model? ")
    do_train_grid_model = confirmation_prompt("Train grid model? ")

    if do_generate_dataset:
        tf.print('Generating dataset...')
        generate_dataset(agent, dataset_path)

    dataset = load_dataset([dataset_path])

    # # Visual module
    visual_model = models.VisualModel(
        img_width, img_height, pc_ensemble, hd_ensemble)
    initial_epoch = load_model_weights_from_checkpoint(
        visual_model, vis_ckpt_path)

    if do_train_visual_model:
        train_visual_module(visual_model, dataset,
                            ckpt_path=vis_ckpt_path, initial_epoch=initial_epoch)


    # # Grid module
    grid_model = models.GridNetworkModel(
        pc_ensemble, hd_ensemble, nh_lstm, nh_bottleneck)
    initial_epoch = load_model_weights_from_checkpoint(
        grid_model, grid_ckpt_path)

    if do_train_grid_model:
        train_grid_module(grid_model, visual_model, dataset,
                          ckpt_path=grid_ckpt_path, initial_epoch=initial_epoch)

    # # Save final weights
    visual_model.save_weights('visual_model.h5')
    grid_model.save_weights('grid_model.h5')

    tf.print("Training done! The trained models have been saved as visual_model.h5 and grid_model.h5")
