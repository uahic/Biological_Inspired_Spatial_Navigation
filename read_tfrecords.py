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
import collections
import os

import third_party.utils as utils


DatasetInfo = collections.namedtuple(
    'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])


_DATASETS = dict(
    square_room=DatasetInfo(
        basepath='square_room_100steps_2.2m_1000000',
        size=100,
        sequence_length=100,
        coord_range=((-1.1, 1.1), (-1.1, 1.1))),)

dataset_info = _DATASETS['square_room']


def get_coord_range():
    return dataset_info.coord_range


def get_dataset_files(dateset_info, root):
    """Generates lists of files for a given dataset version."""
    basepath = dateset_info.basepath
    base = os.path.join(root, basepath)
    num_files = dateset_info.size
    template = '{:0%d}-of-{:0%d}.tfrecord' % (4, 4)
    return [
        os.path.join(base, template.format(i, num_files - 1))
        for i in range(num_files)
    ]


def get_protobuf_example_parser():
    feature_map = {
        'init_pos':
            tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),
        'init_hd':
            tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
        'ego_vel':
            tf.io.FixedLenFeature(
                shape=[dataset_info.sequence_length, 3],
                dtype=tf.float32),
        'target_pos':
            tf.io.FixedLenFeature(
                shape=[dataset_info.sequence_length, 2],
                dtype=tf.float32),
        'target_hd':
            tf.io.FixedLenFeature(
                shape=[dataset_info.sequence_length, 1],
                dtype=tf.float32),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_map)
        return example

    return _parse_function


def get_square_room_dataset():
    file_names = get_dataset_files(dataset_info, './train_data')
    records = tf.data.TFRecordDataset(file_names, num_parallel_reads=4)
    parser = get_protobuf_example_parser()
    parsed_records = records.map(parser)
    return parsed_records
