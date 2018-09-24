# Modified, CFD, May-August 2017
# from code that was
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
"""
Collection of input pipelines.

An input pipeline defines how to read and parse data. It produces a tuple
of (features, labels) that can be read by tf.learn estimators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import sys
import csv
import os

import six
import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder
from seq2seq import graph_utils

from seq2seq.configurable import Configurable
from seq2seq.data import split_tokens_decoder, parallel_data_provider, triple_data_provider
from seq2seq.data.sequence_example_decoder import TFSEquenceExampleDecoder
from seq2seq.data import copying_decoder, copying_data_provider


def make_input_pipeline_from_def(def_dict, mode, **kwargs):
  """Creates an InputPipeline object from a dictionary definition.

  Args:
    def_dict: A dictionary defining the input pipeline.
      It must have "class" and "params" that correspond to the class
      name and constructor parameters of an InputPipeline, respectively.
    mode: A value in tf.contrib.learn.ModeKeys

  Returns:
    A new InputPipeline object
  """
  if not "class" in def_dict:
    raise ValueError("Input Pipeline definition must have a class property.")

  class_ = def_dict["class"]
  if not hasattr(sys.modules[__name__], class_):
    raise ValueError("Invalid Input Pipeline class: {}".format(class_))

  pipeline_class = getattr(sys.modules[__name__], class_)

  # Constructor arguments
  params = {}
  if "params" in def_dict:
    params.update(def_dict["params"])
  params.update(kwargs)

  return pipeline_class(params=params, mode=mode)


@six.add_metaclass(abc.ABCMeta)
class InputPipeline(Configurable):
  """Abstract InputPipeline class. All input pipelines must inherit from this.
  An InputPipeline defines how data is read, parsed, and separated into
  features and labels.

  Params:
    shuffle: If true, shuffle the data.
    num_epochs: Number of times to iterate through the dataset. If None,
      iterate forever.
  """

  def __init__(self, params, mode):
    Configurable.__init__(self, params, mode)

  @staticmethod
  def default_params():
    return {
        "shuffle": True,
        "num_epochs": None,
    }

  def make_data_provider(self, **kwargs):
    """Creates DataProvider instance for this input pipeline. Additional
    keyword arguments are passed to the DataProvider.
    """
    raise NotImplementedError("Not implemented.")

  @property
  def feature_keys(self):
    """Defines the features that this input pipeline provides. Returns
      a set of strings.
    """
    return set()

  @property
  def label_keys(self):
    """Defines the labels that this input pipeline provides. Returns
      a set of strings.
    """
    return set()

  @staticmethod
  def read_from_data_provider(data_provider):
    """Utility function to read all available items from a DataProvider.
    """
    item_values = data_provider.get(list(data_provider.list_items()))
    items_dict = dict(zip(data_provider.list_items(), item_values))
    return items_dict

class ParallelTextInputPipeline(InputPipeline):
  """An input pipeline that reads two parallel (line-by-line aligned) text
  files.

  Params:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "source_files": [],
        "target_files": [],
        "source_delimiter": " ",
        "target_delimiter": " ",
        "build_schema_map_table": False,
        "build_schema_text_table": False
    })
    return params

  def make_data_provider(self, **kwargs):
    decoder_source = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="source_tokens",
        length_feature_name="source_len",
        append_token="SEQUENCE_END",
        delimiter=self.params["source_delimiter"])

    dataset_source = tf.contrib.slim.dataset.Dataset(
        data_sources=self.params["source_files"],
        reader=tf.TextLineReader,
        decoder=decoder_source,
        num_samples=None,
        items_to_descriptions={})

    dataset_target = None
    if len(self.params["target_files"]) > 0:
      decoder_target = split_tokens_decoder.SplitTokensDecoder(
          tokens_feature_name="target_tokens",
          length_feature_name="target_len",
          prepend_token="SEQUENCE_START",
          append_token="SEQUENCE_END",
          delimiter=self.params["target_delimiter"])

      dataset_target = tf.contrib.slim.dataset.Dataset(
          data_sources=self.params["target_files"],
          reader=tf.TextLineReader,
          decoder=decoder_target,
          num_samples=None,
          items_to_descriptions={})

    return parallel_data_provider.ParallelDataProvider(
        dataset1=dataset_source,
        dataset2=dataset_target,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_len"])

class ParallelTextAndMaskInputPipeline(ParallelTextInputPipeline):
    
    @staticmethod
    def default_params():
        params = ParallelTextInputPipeline.default_params()
        params.update({
        "decoder_mask_files": []
        })
        return params
    
    def make_data_provider(self, **kwargs):
        target_files = self.params["target_files"]
        if not target_files:
            target_files = None

        print("make mask data provider args:",
          self.params["source_files"],
          target_files,
          self.params["decoder_mask_files"],
          self.params["shuffle"],
          self.params["num_epochs"],
          self.params["source_delimiter"],
          self.params["target_delimiter"])
        # decoder
        if kwargs is not None:
          for key, value in kwargs.iteritems():
            print(key, value)
        print ("source_files", self.params["source_files"])
        return triple_data_provider.make_triple_data_provider(
          self.params["source_files"],
          target_files,
          self.params["decoder_mask_files"],
          shuffle=self.params["shuffle"],
          num_epochs=self.params["num_epochs"],
          source_delimiter=self.params["source_delimiter"],
          target_delimiter=self.params["target_delimiter"],
          **kwargs)
    @property
    def feature_keys(self):
        return set(["source_tokens", "source_len", "decoder_mask"])
    
    
    
class ParallelTextAndSchemaInputPipeline(ParallelTextInputPipeline):
  """
  An input pipeline that reads three parallel (line-by-line aligned) text files:
  a source, a target, and a schema location.

  Params:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
    schema_loc_files: An array of file names for the schema locations. Each
      file includes one schema location per line, aligned to the source_files.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = ParallelTextInputPipeline.default_params()
    params.update({
      "schema_loc_files": []#,
#      "build_schema_text_table": False
    })
    return params

  def _build_schema_lookup_tables(self):
    # May include: schema map, schema matrix, schema text.

    # Read in all the filenames from all schema_loc_files,
    # identifying unique filenames.
    schema_loc_files = self.params["schema_loc_files"]
    all_schema_locations = set()
    for loc_file in schema_loc_files:
      with open(loc_file, 'r') as f:
        locations = [l.strip() for l in  f.readlines()]
      all_schema_locations.update(locations)
    all_schema_locations = list(all_schema_locations) # fixed order

    # Build a lookup table of filename --> index
    # (Required for all models that use any schema representation)
    schema_file_lookup_table = tf.contrib.lookup.index_table_from_tensor(
      mapping=all_schema_locations, num_oov_buckets=0, default_value=-1)

    # For each filename, get its matrix from the npy file.
    # Note the length of the schema.
    schema_embeddings_matrices = []
    schema_lengths = []

    if self.params["build_schema_text_table"]:
      all_schema_strings = []
    if self.params["build_schema_map_table"]:
      schema_map_matrices = []
      schema_map_lengths = []
    # schema files in npy files
    def load_npy(matrix_list, length_list, file_location, fname):
      npy_file = os.path.join(file_location, fname)
      matrix_np = np.load(npy_file)
      matrix_list.append(matrix_np)
      length = matrix_np.shape[0]
      length_list.append(length)

    for schema_location in all_schema_locations:
      print ("current_schema_location", schema_location)
    # currently running:
      # Schema embeddings: required for all attn to schema models.
      load_npy(schema_embeddings_matrices, schema_lengths,
               schema_location, "schema_embeddings.npy")

      if self.params["build_schema_map_table"]:
        load_npy(schema_map_matrices, schema_map_lengths,
                 schema_location, "schema_map.npy")

      # Schema strings: required for schema-copying models.
      if self.params["build_schema_text_table"]:
        schema_csv_file = os.path.join(schema_location, "schema.csv")
        schema_string = self.get_schema_strings(schema_csv_file)
        all_schema_strings.append(schema_string)

    max_emb_len = max(schema_lengths)
    schema_lengths = tf.constant(schema_lengths)

    # Pad matrices with zeros as needed.
    def pad_to_size(matrix, length):
      if matrix.shape[0] == length:
        return matrix
      padding_size = length - matrix.shape[0]
      padded = np.pad(matrix,
                      pad_width=((0, padding_size), (0,0)),
                      mode='constant', constant_values=0)
      return padded

    schema_embeddings_matrices = [pad_to_size(
      m, max_emb_len) for m in schema_embeddings_matrices]
    # Assemble all the matrices into a big 3d tensor
    all_schema_embeddings = tf.convert_to_tensor(
      np.asarray(schema_embeddings_matrices), dtype=tf.float32)

    tables_dict = {
      "schema_file_lookup_table":schema_file_lookup_table,
      "all_schema_embeddings":all_schema_embeddings,
      "schema_lengths":schema_lengths,
    }

    # Assemble all the schema strings into a big lookup table.
    # (Required for schema-copying models)
    if self.params["build_schema_text_table"]:
      schema_strings_tbl = tf.contrib.lookup.index_to_string_table_from_tensor(
        all_schema_strings, name="schema_strings_lookup_table")
      tables_dict["all_schema_strings"] = schema_strings_tbl

    if self.params["build_schema_map_table"]:
      max_map_len = max(schema_map_lengths)
      schema_map_lengths = tf.constant(schema_map_lengths)
      schema_map_matrices = [pad_to_size(
        m, max_map_len) for m in schema_map_matrices]
      all_schema_maps = tf.convert_to_tensor(
        np.asarray(schema_map_matrices), dtype=tf.float32)
      tables_dict["all_schema_maps"] = all_schema_maps
      tables_dict["schema_map_lengths"] = schema_map_lengths

    graph_utils.add_dict_to_collection(tables_dict, "schema_tables")

  def make_data_provider(self, **kwargs):
    self._build_schema_lookup_tables()
    target_files = self.params["target_files"]
    if not target_files:
      target_files = None
    # TODO: read triple_data_provider

            
    print("make data provider args:",
      self.params["source_files"],
      target_files,
      self.params["schema_loc_files"],
      self.params["shuffle"],
      self.params["num_epochs"],
      self.params["source_delimiter"],
      self.params["target_delimiter"])
          
    if kwargs is not None:
      for key, value in kwargs.iteritems():
        print(key, value)
    print ("source_files", self.params["source_files"])
    return triple_data_provider.make_triple_data_provider(
      self.params["source_files"],
      target_files,
      self.params["schema_loc_files"],
      shuffle=self.params["shuffle"],
      num_epochs=self.params["num_epochs"],
      source_delimiter=self.params["source_delimiter"],
      target_delimiter=self.params["target_delimiter"],
      **kwargs)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len", "schema_loc"])

class ParallelTextAndSchemaMapInputPipeline(ParallelTextAndSchemaInputPipeline):
  """
  An input pipeline that reads three parallel (line-by-line aligned) text files:
  a source, a target, and a schema location. Expects both schema embeddings and
  schema map at the schema locations.

  Params:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
    schema_loc_files: An array of file names for the schema locations. Each
      file includes one schema location per line, aligned to the source_files.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = ParallelTextAndSchemaInputPipeline.default_params()
    params.update({
      "build_schema_map_table": True
    })
    return params

    @property
    def feature_keys(self):
        return set(["source_tokens", "source_len", "decoder_mask"])

class ParallelTextAndMaskCopyingPipeline(ParallelTextAndMaskInputPipeline):
    
  def make_data_provider(self, **kwargs):
    target_files = self.params["target_files"]
    if not target_files:
      target_files = None
    return self._get_copying_data_provider(
      target_files, **kwargs)

    # why need copying data provider?
  def _get_copying_data_provider(self, target_files, **kwargs):
    return copying_data_provider.make_word_copying_data_provider(
      self.params["source_files"],
      target_files,
      self.params["decoder_mask_files"],
      num_epochs=self.params["num_epochs"],
      shuffle=self.params["shuffle"],
      source_delimiter=self.params["source_delimiter"],
      target_delimiter=self.params["target_delimiter"],
      **kwargs)
  # what does this function do?
  def _get_copying_decoder(self, tokens_feature_name, length_feature_name,
                          prepend_token, append_token, delimiter):
    return copying_decoder.WordCopyingDecoder(
      tokens_feature_name=tokens_feature_name,
      length_feature_name=length_feature_name,
      prepend_token=prepend_token, 
    append_token=append_token,
      delimiter=delimiter)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len", "decoder_mask"])


  @property
  def label_keys(self):
    return set(["target_tokens", "target_len", "source_copy_indices"])

# BaseParallelCopyingPipeline: can copy from source or schema
class BaseParallelCopyingPipeline(ParallelTextAndSchemaInputPipeline):
  """A base class for copying input pipeline that reads three parallel
  (line-by-line aligned) text files and identifies tokens copied from
  the schema or source.

  Params:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
      names in column 0 and "" or field name in column 1.
    schema_loc_files: An array of file names for the schema locations. Each
      file includes one schema location per line, aligned to the source_files.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  def make_data_provider(self, **kwargs):
    self._build_schema_lookup_tables()
    target_files = self.params["target_files"]
    # no target file in inference?
    if not target_files:
      target_files = None
    return self._get_copying_data_provider(
      target_files, **kwargs)

  def _get_copying_data_provider(self, target_files, **kwargs):
    raise NotImplementedError

  def _get_copying_decoder(self, tokens_feature_name, length_feature_name,
                          prepend_token, append_token, delimiter):
    raise NotImplementedError

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len", "schema_loc"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_len"])

class ParallelSchemaCopyingPipeline(BaseParallelCopyingPipeline):
  """A copying input pipeline that reads two parallel (line-by-line aligned)
  text files and a schema. It identifies tokens copied from the schema.

  Params:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = BaseParallelCopyingPipeline.default_params()
    params.update({
      "build_schema_text_table": True
    })
    return params
    
    # why need copying data provider?
  def _get_copying_data_provider(self, target_files, **kwargs):
    return copying_data_provider.make_schema_copying_data_provider(
      self.params["source_files"],
      target_files,
      self.params["schema_loc_files"],
      num_epochs=self.params["num_epochs"],
      shuffle=self.params["shuffle"],
      source_delimiter=self.params["source_delimiter"],
      target_delimiter=self.params["target_delimiter"],
      **kwargs)

  # @property
  # def feature_keys(self):
  #   keys = super(ParallelSchemaCopyingPipeline, self).feature_keys
  #   keys.update({"schema_loc"})
  #   return keys

  @property
  def label_keys(self):
    keys = super(ParallelSchemaCopyingPipeline, self).label_keys
    keys.update({"schema_copy_indices"})
    return keys

  def get_schema_strings(self, schema_filename):
    # Open the file, and read the table and field names into schema.
    schema_strings = []

    with open(schema_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        is_header_row = True
        for row in reader:
          if is_header_row:
            is_header_row = False
            continue
          table_name = row[0].strip()
          field_name = row[1].strip()
          if len(field_name) == 0:
            schema_strings.append(table_name)
          else:
            schema_strings.append(table_name + "," + field_name)
    return " ".join(schema_strings)

class ParallelTextAndSchemaCopyingPipeline(ParallelSchemaCopyingPipeline):
  """A copying input pipeline that reads two parallel (line-by-line aligned)
  text files and a schema. It identifies tokens copied from both the schema
  and source.

  Params:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
    schema_loc_files: An array of file names for the schema locations directories. Each directory must include a schema_embeddings.npy file and a schema.csv file. The latter should have table name in the first column and field name in the second.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  def _get_copying_decoder(self, tokens_feature_name, length_feature_name,
                          prepend_token, append_token, delimiter):
    return copying_decoder.SchemaAndWordCopyingDecoder(
      tokens_feature_name=tokens_feature_name,
      length_feature_name=length_feature_name,
      prepend_token=prepend_token, append_token=append_token,
      delimiter=delimiter)
    # todo: read copying_data_provider
  def _get_copying_data_provider(self, target_files, **kwargs):
    return copying_data_provider.make_schema_and_word_copying_data_provider(
      self.params["source_files"],
      target_files,
      self.params["schema_loc_files"],
      num_epochs=self.params["num_epochs"],
      shuffle=self.params["shuffle"],
      source_delimiter=self.params["source_delimiter"],
      target_delimiter=self.params["target_delimiter"],
      **kwargs)

  @property
  def label_keys(self):
    keys = super(ParallelTextAndSchemaCopyingPipeline, self).label_keys
    keys.update({"source_copy_indices"})
    return keys


class ParallelTextCopyingPipeline(BaseParallelCopyingPipeline):
  """A copying input pipeline that reads two parallel (line-by-line aligned)
  text files. It identifies tokens copied from the input sequence to the
  output sequence.

  Params:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  def _get_copying_decoder(self, tokens_feature_name, length_feature_name,
                          prepend_token, append_token, delimiter):
    return copying_decoder.WordCopyingDecoder(
      tokens_feature_name=tokens_feature_name,
      length_feature_name=length_feature_name,
      prepend_token=prepend_token, append_token=append_token,
      delimiter=delimiter)

  def _get_copying_data_provider(self, target_files, **kwargs):
    return copying_data_provider.make_word_copying_data_provider(
      self.params["source_files"],
      target_files,
      self.params["schema_loc_files"],
      num_epochs=self.params["num_epochs"],
      shuffle=self.params["shuffle"],
      source_delimiter=self.params["source_delimiter"],
      target_delimiter=self.params["target_delimiter"],
      **kwargs)

  @property
  def label_keys(self):
    keys = super(ParallelTextCopyingPipeline, self).label_keys
    keys.update({"source_copy_indices"})
    return keys

class TFRecordInputPipeline(InputPipeline):
  """An input pipeline that reads a TFRecords containing both source
  and target sequences.

  Params:
    files: An array of file names to read from.
    source_field: The TFRecord feature field containing the source text.
    target_field: The TFRecord feature field containing the target text.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "files": [],
        "source_field": "source",
        "target_field": "target",
        "source_delimiter": " ",
        "target_delimiter": " ",
    })
    return params

  def make_data_provider(self, **kwargs):

    splitter_source = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="source_tokens",
        length_feature_name="source_len",
        append_token="SEQUENCE_END",
        delimiter=self.params["source_delimiter"])

    splitter_target = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="target_tokens",
        length_feature_name="target_len",
        prepend_token="SEQUENCE_START",
        append_token="SEQUENCE_END",
        delimiter=self.params["target_delimiter"])

    keys_to_features = {
        self.params["source_field"]: tf.FixedLenFeature((), tf.string),
        self.params["target_field"]: tf.FixedLenFeature(
            (), tf.string, default_value="")
    }

    items_to_handlers = {}
    items_to_handlers["source_tokens"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["source_field"]],
        func=lambda dict: splitter_source.decode(
            dict[self.params["source_field"]], ["source_tokens"])[0])
    items_to_handlers["source_len"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["source_field"]],
        func=lambda dict: splitter_source.decode(
            dict[self.params["source_field"]], ["source_len"])[0])
    items_to_handlers["target_tokens"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["target_field"]],
        func=lambda dict: splitter_target.decode(
            dict[self.params["target_field"]], ["target_tokens"])[0])
    items_to_handlers["target_len"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["target_field"]],
        func=lambda dict: splitter_target.decode(
            dict[self.params["target_field"]], ["target_len"])[0])

    decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                 items_to_handlers)

    dataset = tf.contrib.slim.dataset.Dataset(
        data_sources=self.params["files"],
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={})

    return tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset=dataset,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_len"])


class ImageCaptioningInputPipeline(InputPipeline):
  """An input pipeline that reads a TFRecords containing both source
  and target sequences.

  Params:
    files: An array of file names to read from.
    source_field: The TFRecord feature field containing the source text.
    target_field: The TFRecord feature field containing the target text.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "files": [],
        "image_field": "image/data",
        "image_format": "jpg",
        "caption_ids_field": "image/caption_ids",
        "caption_tokens_field": "image/caption",
    })
    return params

  def make_data_provider(self, **kwargs):

    context_keys_to_features = {
        self.params["image_field"]: tf.FixedLenFeature(
            [], dtype=tf.string),
        "image/format": tf.FixedLenFeature(
            [], dtype=tf.string, default_value=self.params["image_format"]),
    }

    sequence_keys_to_features = {
        self.params["caption_ids_field"]: tf.FixedLenSequenceFeature(
            [], dtype=tf.int64),
        self.params["caption_tokens_field"]: tf.FixedLenSequenceFeature(
            [], dtype=tf.string)
    }

    items_to_handlers = {
        "image": tfexample_decoder.Image(
            image_key=self.params["image_field"],
            format_key="image/format",
            channels=3),
        "target_ids":
        tfexample_decoder.Tensor(self.params["caption_ids_field"]),
        "target_tokens":
        tfexample_decoder.Tensor(self.params["caption_tokens_field"]),
        "target_len": tfexample_decoder.ItemHandlerCallback(
            keys=[self.params["caption_tokens_field"]],
            func=lambda x: tf.size(x[self.params["caption_tokens_field"]]))
    }

    decoder = TFSEquenceExampleDecoder(
        context_keys_to_features, sequence_keys_to_features, items_to_handlers)

    dataset = tf.contrib.slim.dataset.Dataset(
        data_sources=self.params["files"],
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={})

    return tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset=dataset,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  @property
  def feature_keys(self):
    return set(["image"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_ids", "target_len"])
