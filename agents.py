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
import numpy as np
import random


class RandomAgent(object):
    """Basic random agent for DeepMind Lab."""

    def __init__(self, action_spec, forbidden_actions=[]):
        self.action_spec = action_spec
        self.action_count = len(action_spec)
        self.forbidden_actions = forbidden_actions
        self.prev_action = None

    def step(self):
        """Choose a random amount of a randomly selected action."""
        action_choice = None
        while action_choice is None:
            action_choice = random.randint(0, self.action_count - 1)
            if self.action_spec[action_choice]['name'] in self.forbidden_actions:
                action_choice = None
        action_amount = random.randint(self.action_spec[action_choice]['min'],
                                       self.action_spec[action_choice]['max'])
        action = np.zeros([self.action_count], dtype=np.intc)
        action[action_choice] = action_amount
        return action

    def reset(self):
        self.prev_action = None


class RodentDynamicModel(object):
    def __init__(self, min_x, min_y, max_x, max_y):
        self.size_x = max_x - min_x
        self.size_y = max_y - min_y
        self.mu_x = min_x + self.size_x/2.
        self.mu_y = min_y + self.size_y/2.
        self.cov = np.diag([(self.size_x/3.)**2.0, (self.size_y/3.)**2.])

    def step(self):
        x, y = np.random.multivariate_normal(
            [self.mu_x, self.mu_y], self.cov, (1, 2))
        


class GaussRandomAgent(object):
    def __init__(self, action_spec, forbidden_actions=[]):
        self.action_spec = action_spec
        self.action_count = len(action_spec)
        self.forbidden_actions = forbidden_actions
        self.max_speed = 1.
        self.max_rotation = 512

        
        self.reset()

    def step(self, ego_vel_trans, ego_vel_rot):
        """Choose a random amount of a randomly selected action."""

        action = np.zeros([self.action_count], dtype=np.intc)
        speed = np.random.choice([-1, 0, 1], p=[0.05, 0.25, 0.7])
        action[3] = speed 

        left_right_pixel = max(min(self.max_rotation * np.random.normal(0.0, 0.2), self.max_rotation), -self.max_rotation)
        # left_right_pixel = self.max_rotation * np.random.vonmises(
        #     ego_vel_trans * self.max_rotation, 1.)
        action[0] = left_right_pixel + 0.2 * self.prev_action[0]

        self.prev_action_count += 1
        self.prev_action = action
        return action

    def reset(self):
        self.prev_action_count = 0
        self.prev_action = np.zeros([self.action_count], dtype=np.intc)
        self.prev_amount = None
