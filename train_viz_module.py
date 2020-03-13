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
import deepmind_lab
import random
import numpy as np
import agents
import buffers
import memory_profiler

import third_party.scores as scores
import third_party.ensembles as ensembles
import third_party.utils as utils


def softmax_cross_entropy_logits_loss(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred)
    return loss


def train_visual_module(img_width, img_height, pc_ensemble, hd_ensemble, lab_config, level, level_boundary_min, level_boundary_max):
    assert str(
        img_width) == lab_config['width'], "DM-Lab camera width does not match the width of the visual module"
    assert str(
        img_height) == lab_config['height'], "DM-Lab camera height does not match the height of the visual module"

    model = networks.VisualModule(
        img_width, img_height, pc_ensemble, hd_ensemble)

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(32, input_shape=(64,64,3))
    # ])

    # Prepare env and random agent
    observations = ['RGB', 'DEBUG.POS.ROT', 'DEBUG.POS.TRANS']
    env = deepmind_lab.Lab(level, observations,
                           config=lab_config, renderer='software')

    agent = agents.RandomAgent(
        env.action_spec(),
        forbidden_actions=['JUMP', 'FIRE', 'CROUCH',
                           'LOOK_DOWN_UP_PIXELS_PER_FRAME']
    )

    episode_length = 100
    total_frames = episode_length * 1e6
    batch_size = 32
    epochs = 1000
    training_steps_per_epoch = total_frames // batch_size
    # Record training data


    def generate_batch():
        replay_buffer = buffers.ReplayBuffer(
            batch_size * episode_length
        )
        # pc_boundary_= (pc_ensemble.pos_max - pc_ensemble.pos_min)
        pc_boundary_scale = (pc_ensemble.pos_max - pc_ensemble.pos_min) / \
            (level_boundary_max - level_boundary_min)

        level_boundary_mean = (
            level_boundary_max - level_boundary_min)/2. + level_boundary_min

        while True:
            env.reset()
            # Collect observations
            for _ in range(batch_size):
                for _ in range(episode_length):
                    if not env.is_running():
                        print('Environment stopped early')
                        env.reset()
                        agent.reset()
                    obs = env.observations()
                    if not obs:
                        raise Exception('Observations empty!')

                    # Normalize observations
                    rgb = obs['RGB']
                    target_pos = obs['DEBUG.POS.TRANS'][:2]
                    target_rot = obs['DEBUG.POS.ROT'][1]
                    target_pos -= level_boundary_mean
                    target_pos *= pc_boundary_scale
                    target_rot = target_rot * ((2. * np.pi)/360.)
                    replay_buffer.add([rgb, target_pos, target_rot])
                    action = agent.step()
                    env.step(action, num_steps=1)

            # Form batches
            # TODO make sure that the replay buffer is actually filled (no early environment stoppings, see above)
            target_pos_batch = np.zeros((batch_size, episode_length, 2))
            target_rot_batch = np.zeros((batch_size, episode_length, 1))
            rgb_batch = np.zeros(
                (batch_size, episode_length, 3, img_width, img_height))

            for i in range(batch_size):
                sampled_obs = replay_buffer.sample(episode_length)
                target_pos_batch[i, :, :] = np.array(
                    list(map(lambda x: x[1], sampled_obs)))
                target_rot_batch[i, :, 0] = np.array(
                    list(map(lambda x: x[2], sampled_obs)))
                rgb_batch[i, :, :, :, :] = np.array(
                    list(map(lambda x: x[0], sampled_obs)))

            replay_buffer.clear()

            targets = utils.encode_targets(target_pos_batch, target_rot_batch, [
                pc_ensemble], [hd_ensemble])

            # Compute training targets
            rgb_batch = np.swapaxes(rgb_batch, 2, 4)
            rgb_batch = np.swapaxes(rgb_batch, 2, 3)
            rgb_batch = tf.convert_to_tensor(rgb_batch)

            yield (rgb_batch), (targets[0], targets[1])

    # Prepare model training
    model.compile(
        optimizer=tf.optimizers.RMSprop(
            learning_rate=1e-5,
            momentum=0.9,
            clipvalue=1e-5
        ),
        loss={'output_1': softmax_cross_entropy_logits_loss,
              'output_2': softmax_cross_entropy_logits_loss}
    )

    # batch_generator = generate_batch(batch_size=10)
    # import ipdb; ipdb.set_trace()
    # for _ in range(epochs):
    #     for _ in range(training_steps_per_epoch):
    #         x, y = next(batch_generator)
    #         pc_pred, hd_pred = model.train_on_batch(x, y)

    # model.fit_generator(
    #     generate_batch(),
    #     epochs=epochs,
    #     steps_per_epoch=training_steps_per_epoch,
    #     verbose=1,
    # )
    model.fit(
        generate_batch(),
        epochs=epochs,
        steps_per_epoch=training_steps_per_epoch,
        verbose=1,
    )


if __name__ == '__main__':
    n_pc = 256
    n_hd = 10
    img_width = 32
    img_height = 32

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

    lab_config = {'width': str(img_width), 'height': str(
        img_height), 'fps': '30'}
    level = 'tests/empty_room_test'
    level_boundary_min = np.array([116.125, 116.125])
    level_boundary_max = np.array([783.87, 783.87])

    try:
        train_visual_module(img_width, img_height, pc_ensemble, hd_ensemble,
                            lab_config, level, level_boundary_min, level_boundary_max)
    except Exception as err:
        tf.print('Error raised: ', err)
