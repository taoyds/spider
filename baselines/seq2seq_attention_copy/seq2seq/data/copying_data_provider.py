# Based in part on code that was
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Data Provder that reads parallel (aligned) data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_provider
from tensorflow.contrib.slim.python.slim.data import parallel_reader

from seq2seq.data import copying_decoder, split_tokens_decoder

class BaseCopyingDataProvider(data_provider.DataProvider):
  """Base class for CopyingDataProvider. This data provider reads two datasets
  in parallel, keeping them aligned. It notes where each target copies
  from the parallel source or the schema.

  Args:
    dataset1: The first dataset. An instance of the Dataset class.
    dataset2: The second dataset. An instance of the Dataset class.
      Can be None. If None, only `dataset1` is read.
    schemas: The schema locations. An instance of the Dataset
      class. Can be None.
    num_readers: The number of parallel readers to use.
    shuffle: Whether to shuffle the data sources and common queue when
      reading.
    num_epochs: The number of times each data source is read. If left as None,
      the data will be cycled through indefinitely.
    common_queue_capacity: The capacity of the common queue.
    common_queue_min: The minimum number of elements in the common queue after
      a dequeue.
    seed: The seed to use if shuffling.
  """

  def __init__(self,
               dataset1,
               dataset2,
               schemas=None,
               shuffle=True,
               num_epochs=None,
               common_queue_capacity=4096,
               common_queue_min=1024,
               seed=None):

    if seed is None:
      seed = np.random.randint(10e8)

    _, data_source = parallel_reader.parallel_read(
        dataset1.data_sources,
        reader_class=dataset1.reader,
        num_epochs=num_epochs,
        num_readers=1,
        shuffle=False,
        capacity=common_queue_capacity,
        min_after_dequeue=common_queue_min,
        seed=seed)

    data_target = ""
    if dataset2 is not None:
      _, data_target = parallel_reader.parallel_read(
          dataset2.data_sources,
          reader_class=dataset2.reader,
          num_epochs=num_epochs,
          num_readers=1,
          shuffle=False,
          capacity=common_queue_capacity,
          min_after_dequeue=common_queue_min,
          seed=seed)

    data_schemas = ""
    # print ("schema is, " schemas)
    if schemas is not None:
      _, data_schemas = parallel_reader.parallel_read(
            schemas.data_sources,
            reader_class=schemas.reader,
            num_epochs=num_epochs,
            num_readers=1,
            shuffle=False,
            capacity=common_queue_capacity,
            min_after_dequeue=common_queue_min,
            seed=seed)

    # Optionally shuffle the data
    if shuffle:
      shuffle_queue = tf.RandomShuffleQueue(
          capacity=common_queue_capacity,
          min_after_dequeue=common_queue_min,
          dtypes=[tf.string, tf.string, tf.string],
          seed=seed)
      enqueue_ops = []
      enqueue_ops.append(shuffle_queue.enqueue([data_source, data_target, data_schemas]))
      tf.train.add_queue_runner(
          tf.train.QueueRunner(shuffle_queue, enqueue_ops))
      data_source, data_target, data_schemas = shuffle_queue.dequeue()

    # Decode source items
    items = dataset1.decoder.list_items()
    tensors = dataset1.decoder.decode(data_source, items)

    tensors_schema = [None]
    if schemas is not None:
      items_schema = schemas.decoder.list_items()
      tensors_schema = schemas.decoder.decode(data_schemas, items_schema)

    if dataset2 is not None:
      target_and_copy_sources = self._target_and_copy_sources(data_target, tensors[0], tensors_schema)

      # Decode target items
      items2 = dataset2.decoder.list_items()
      tensors2 = dataset2.decoder.decode(target_and_copy_sources, items2)

      # Merge items and results
      items = items + items2
      tensors = tensors + tensors2

    if schemas is not None:
      items = items + items_schema
      tensors = tensors + tensors_schema

    super(BaseCopyingDataProvider, self).__init__(
        items_to_tensors=dict(zip(items, tensors)),
        num_samples=dataset1.num_samples)

  def _target_and_copy_sources(self, data_target, source_tensors, schema_tensors):
    raise NotImplementedError

def _make_copying_data_provider_base(data_sources_source,
                                     data_sources_schema,
                                     reader=tf.TextLineReader,
                                     num_samples=None,
                                     source_delimiter=" ",
                                     **kwargs):
  """
  Prepare the Datasets that will be used to make the copying data provider.

  Args:
    data_sources_source: A list of data sources for the source text files.
    data_sources_schema: A list of data sources for the schema location text files.
    reader: A reader that can handle the source and schema files.
    num_samples: Optional, number of records in the dataset
    delimiter: Split tokens in the data on this delimiter. Defaults to space.
    kwargs: Additional arguments (shuffle, num_epochs, etc) that are passed
      to the data provider

  Returns:
    The Datasets for source and schema.
"""
  decoder_source = split_tokens_decoder.SplitTokensDecoder(
      tokens_feature_name="source_tokens",
      length_feature_name="source_len",
      append_token="SEQUENCE_END",
      delimiter=source_delimiter)

  dataset_source = tf.contrib.slim.dataset.Dataset(
      data_sources=data_sources_source,
      reader=reader,
      decoder=decoder_source,
      num_samples=num_samples,
      items_to_descriptions={})
  print ("source schema", data_sources_schema)
  dataset_schemas = None
  if data_sources_schema is not None:
    decoder_schemas = split_tokens_decoder.SplitMaskDecoder(
        decoder_mask_feature_name="decoder_mask",
        delimiter=" ")
    
#     decoder_schemas = split_tokens_decoder.SplitTokensDecoder(
#       tokens_feature_name="schema_loc",
#       delimiter=" ")
    dataset_schemas = tf.contrib.slim.dataset.Dataset(
      data_sources=data_sources_schema,
      reader=reader,
      decoder=decoder_schemas,
      num_samples=num_samples,
      items_to_descriptions={})
  return dataset_source, dataset_schemas

def make_schema_copying_data_provider(data_sources_source,
                                      data_sources_target,
                                      data_sources_schema,
                                      reader=tf.TextLineReader,
                                      num_samples=None,
                                      source_delimiter=" ",
                                      target_delimiter=" ", **kwargs):
  """
  Builds a copying data provider for schema-only copying.
  Args:
    data_sources_source: A list of data sources for the source text
      files.
    data_sources_target: A list of data sources for the targer text
      files.
    data_sources_schema: A list of data sources for the schema
      location text files.
    reader: A reader that can handle the source and schema files.
    num_samples: Optional, number of records in the dataset
    source_delimiter: Split tokens in the source data on this
      delimiter. Defaults to space.
    target_delimiter: Split tokens in the target data on this
      delimiter. Defaults to space.
    kwargs: Additional arguments (shuffle, num_epochs, etc) that are passed
      to the data provider

  Returns:
    A SchemaCopyingDataProvider.
  """

  dataset_source, dataset_schemas = _make_copying_data_provider_base(
    data_sources_source, data_sources_schema, reader=tf.TextLineReader,
    num_samples=num_samples, source_delimiter=" ", **kwargs)
  dataset_target = None
  if data_sources_target is not None:
    decoder_target = copying_decoder.SchemaCopyingDecoder(
        tokens_feature_name="target_tokens",
        length_feature_name="target_len",
        prepend_token="SEQUENCE_START",
        append_token="SEQUENCE_END",
        delimiter=target_delimiter)

    dataset_target = tf.contrib.slim.dataset.Dataset(
        data_sources=data_sources_target,
        reader=reader,
        decoder=decoder_target,
        num_samples=num_samples,
        items_to_descriptions={})

  return SchemaCopyingDataProvider(
    dataset1=dataset_source, dataset2=dataset_target,
    schemas=dataset_schemas, **kwargs)

def make_schema_and_word_copying_data_provider(data_sources_source,
                                               data_sources_target,
                                               data_sources_schema,
                                               reader=tf.TextLineReader,
                                               num_samples=None,
                                               source_delimiter=" ",
                                               target_delimiter=" ",
                                               **kwargs):
  """
  Builds a copying data provider for schema and word copying.
  Args:
    data_sources_source: A list of data sources for the source text
      files.
    data_sources_target: A list of data sources for the targer text
      files.
    data_sources_schema: A list of data sources for the schema
      location text files.
    reader: A reader that can handle the source and schema files.
    num_samples: Optional, number of records in the dataset
    source_delimiter: Split tokens in the source data on this
      delimiter. Defaults to space.
    target_delimiter: Split tokens in the target data on this
      delimiter. Defaults to space.
    kwargs: Additional arguments (shuffle, num_epochs, etc) that are passed
      to the data provider

  Returns:
    A SchemaAndWordCopyingDataProvider.
  """

  dataset_source, dataset_schemas = _make_copying_data_provider_base(
    data_sources_source, data_sources_schema, reader=tf.TextLineReader,
    num_samples=num_samples, source_delimiter=" ", **kwargs)
  dataset_target = None
  if data_sources_target is not None:
    decoder_target = copying_decoder.SchemaAndWordCopyingDecoder(
        tokens_feature_name="target_tokens",
        length_feature_name="target_len",
        prepend_token="SEQUENCE_START",
        append_token="SEQUENCE_END",
        delimiter=target_delimiter)

    dataset_target = tf.contrib.slim.dataset.Dataset(
        data_sources=data_sources_target,
        reader=reader,
        decoder=decoder_target,
        num_samples=num_samples,
        items_to_descriptions={})

  return SchemaAndWordCopyingDataProvider(
    dataset1=dataset_source, dataset2=dataset_target,
    schemas=dataset_schemas, **kwargs)

def make_word_copying_data_provider(data_sources_source,
                                    data_sources_target,
                                    data_sources_schema=None,
                                    reader=tf.TextLineReader,
                                    num_samples=None,
                                    source_delimiter=" ",
                                    target_delimiter=" ",
                                    **kwargs):
  """
  Builds a copying data provider for word-only copying.
  Args:
    data_sources_source: A list of data sources for the source text
      files.
    data_sources_target: A list of data sources for the targer text
      files.
    data_sources_schema: An optional list of data sources for the schema
      location text files.
    reader: A reader that can handle the source and schema files.
    num_samples: Optional, number of records in the dataset
    source_delimiter: Split tokens in the source data on this
      delimiter. Defaults to space.
    target_delimiter: Split tokens in the target data on this
      delimiter. Defaults to space.
    kwargs: Additional arguments (shuffle, num_epochs, etc) that are passed
      to the data provider

  Returns:
    A WordCopyingDataProvider.
  """

  dataset_source, dataset_schemas = _make_copying_data_provider_base(
    data_sources_source, data_sources_schema, reader=tf.TextLineReader,
    num_samples=num_samples, source_delimiter=" ", **kwargs)
  dataset_target = None
  if data_sources_target is not None:
    decoder_target = copying_decoder.WordCopyingDecoder(
        tokens_feature_name="target_tokens",
        length_feature_name="target_len",
        prepend_token="SEQUENCE_START",
        append_token="SEQUENCE_END",
        delimiter=target_delimiter)

    dataset_target = tf.contrib.slim.dataset.Dataset(
        data_sources=data_sources_target,
        reader=reader,
        decoder=decoder_target,
        num_samples=num_samples,
        items_to_descriptions={})

  return WordCopyingDataProvider(
    dataset1=dataset_source, dataset2=dataset_target,
    schemas=dataset_schemas, **kwargs)

class SchemaCopyingDataProvider(BaseCopyingDataProvider):
  def _target_and_copy_sources(self, data_target, source_tensors, schema):
    return [data_target, None, schema]

class WordCopyingDataProvider(BaseCopyingDataProvider):
  def _target_and_copy_sources(self, data_target, source_tensors, schema):
    return [data_target, source_tensors, None]

class SchemaAndWordCopyingDataProvider(BaseCopyingDataProvider):
  def _target_and_copy_sources(self, data_target, source_tensors, schema):
    return [data_target, source_tensors, schema]
