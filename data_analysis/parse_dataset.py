#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parses the tfrecord dataset acquired from Kaggle as of Dec 2022.

Most code lines are from the Next Day Wildfire Spread github repo notebook.
"""

from typing import Dict, List, Text
import tensorflow as tf

INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']
OUTPUT_FEATURES = ['FireMask']


def _get_features_desc(
    sample_size: int,
    features: List[Text]
) -> Dict[Text, tf.io.FixedLenFeature]:
    """Creates a features dictionary for TensorFlow IO.
    Args:
        sample_size: Size of the input tiles (square).
        features: List of feature names.
    Returns:
        A features dictionary for TensorFlow IO.
    """
    sample_shape = [sample_size, sample_size]
    features = set(features)
    columns = [
        tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32)
        for _ in features
    ]
    return dict(zip(features, columns))


def _parse_function(
    example_proto: tf.train.Example,
    data_size: int
) -> Dict[Text, tf.Tensor]:
    features_desc =\
        _get_features_desc(data_size, INPUT_FEATURES + OUTPUT_FEATURES)
    features_dict = tf.io.parse_single_example(example_proto, features_desc)
    return features_dict  # {feat_name: feat_tensor(TensorShape([64, 64])), ..}


def get_dataset(
    file_pattern: Text,
    data_size: int
) -> tf.data.Dataset:
    """Gets the dataset from the file pattern.

    Args:
        file_pattern: Input file pattern.
        data_size: Size of tiles (square) as read from input files.

    Returns:
        A TensorFlow dataset loaded from the input file pattern as is.
    """
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=None),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x: _parse_function(x, data_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
