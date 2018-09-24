# Modified from attention_seq2seq.py, which is
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
Sequence to Sequence model with attention to input sequence and schema.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate

import tensorflow as tf

from seq2seq import decoders
from seq2seq.models.basic_seq2seq import BasicSeq2Seq
from seq2seq import graph_utils

import numpy as np

class SchemaAttentionSeq2Seq(BasicSeq2Seq):
  """Sequence2Sequence model with attention mechanism for both input sequence
  and database schema.

  Args:
    source_vocab_info: An instance of `VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `VocabInfo`
      for the target vocabulary
    params: A dictionary of hyperparameters
  """

  def __init__(self, params, mode, name="schema_att_seq2seq"):
    super(SchemaAttentionSeq2Seq, self).__init__(params, mode, name)

  @staticmethod
  def default_params():
    params = BasicSeq2Seq.default_params().copy()
    params.update({
        "attention.class": "AttentionLayerBahdanau",
        "attention.params": {"num_units": 150},
        "bridge.class": "seq2seq.models.bridges.ZeroBridge",
        "encoder.class": "seq2seq.encoders.BidirectionalRNNEncoder",
        "encoder.params": {"rnn_cell": {"cell_class": "LSTMCell",
                                        "cell_params":
                                        {"num_units": 150},
                                        "dropout_input_keep_prob": 0.5,
                                        "dropout_output_keep_prob": 0.5,
                                        "num_layers": 1}},
        "decoder.class": "seq2seq.decoders.SchemaAttentionDecoder",
        "decoder.params": {"max_decode_length": 250,
                           "rnn_cell": {"cell_class": "LSTMCell",
                                        "cell_params":
                                        {"num_units": 150},
                                        "dropout_input_keep_prob": 0.5,
                                        "dropout_output_keep_prob": 0.5,
                                        "num_layers": 1}},
        "optimizer.name": "Adam",
        "optimizer.params": {"epsilon": 0.0000008},
        "optimizer.learning_rate": 0.0005,
        "schema.attention.class":"AttentionLayerBahdanau",
        "schema.attention.params": {"num_units": 150},
        "source.max_seq_len": 50,
        "source.reverse": False,
        "target.max_seq_len": 250,
    })
    return params

  def _get_tables_and_ids(self, features):
    schema_tables = graph_utils.get_dict_from_collection("schema_tables")
    schema_locs = features['schema_loc']
    table = schema_tables["schema_file_lookup_table"]
    ids = table.lookup(schema_locs)
    return (schema_tables, ids)

  def _schema_lookups(self, features):
    schema_tables, ids = self._get_tables_and_ids(features)
    all_schema_embeddings = schema_tables["all_schema_embeddings"]
    schema_embeddings_3d = tf.squeeze(tf.gather(all_schema_embeddings, ids), [1])
    schema_lengths = schema_tables["schema_lengths"]
    schema_attn_values_length = tf.squeeze(tf.gather(schema_lengths, ids), [1])
    return (schema_embeddings_3d,
            schema_embeddings_3d, schema_attn_values_length)
  # provides the decode attention function
  def _get_decoder_args(self, encoder_output, features, _labels):
    attention_class = locate(self.params["attention.class"]) or \
      getattr(decoders.attention, self.params["attention.class"])
    attention_layer = attention_class(
        params=self.params["attention.params"], mode=self.mode)
    # dynamicly load a class!!!
    schema_attention_class = locate(self.params["schema.attention.class"]) or \
      getattr(decoders.attention, self.params["schema.attention.class"])
    schema_attention_layer = schema_attention_class(
        params=self.params["schema.attention.params"], mode=self.mode)

    # If the input sequence is reversed we also need to reverse
    # the attention scores.
    reverse_scores_lengths = None
    if self.params["source.reverse"]:
      reverse_scores_lengths = features["source_len"]
      if self.use_beam_search:
        reverse_scores_lengths = tf.tile(
            input=reverse_scores_lengths,
            multiples=[self.params["inference.beam_search.beam_width"]])

    mode = self.mode
    params = self.params["decoder.params"]
    vocab_size=self.target_vocab_info.total_size
    return (params, mode, vocab_size, encoder_output.attention_values,
            encoder_output.attention_values_length, encoder_output.outputs,
            attention_layer, reverse_scores_lengths, schema_attention_layer)

  def _create_decoder(self, encoder_output, features, _labels):
    (params, mode, vocab_size, attention_values, attention_values_length,
     attention_keys, attention_fn, reverse_scores_lengths,
     schema_attention_fn) = self._get_decoder_args(encoder_output,
                                                   features, _labels)
    (schema_attention_keys, schema_attention_values,
     schema_attention_values_length) = self._schema_lookups(features)

    return self.decoder_class(
        params=params,
        mode=mode,
        vocab_size=vocab_size,
        attention_values=attention_values,
        attention_values_length=attention_values_length,
        attention_keys=attention_keys,
        attention_fn=attention_fn,
        reverse_scores_lengths=reverse_scores_lengths,
        schema_attention_keys=schema_attention_keys,
        schema_attention_values=schema_attention_values,
        schema_attention_values_length=schema_attention_values_length,
        schema_attention_fn=schema_attention_fn
    )

class SchemaMapAttentionSeq2Seq(SchemaAttentionSeq2Seq):
  """Seq2Seq model with attention to input, schema, and schema map.
  Args:
    source_vocab_info: An instance of `VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `VocabInfo`
      for the target vocabulary
    params: A dictionary of hyperparameters
  """
  def __init__(self, params, mode, name="schema_map_att_seq2seq"):
    super(SchemaAttentionSeq2Seq, self).__init__(params, mode, name)

  @staticmethod
  def default_params():
    params = SchemaAttentionSeq2Seq.default_params().copy()
    params.update({
      "decoder.class": "seq2seq.decoders.SchemaMapAttentionDecoder",
      "schema_map.attention.class":"AttentionLayerBahdanau",
      "schema_map.attention.params": {"num_units": 150}
    })
    return params

  def _schema_lookups(self, features):
    schema_tables, ids = self._get_tables_and_ids(features)
    all_schema_embeddings = schema_tables["all_schema_embeddings"]
    schema_embeddings_3d = tf.squeeze(tf.gather(all_schema_embeddings, ids), [1])
    all_schema_maps = schema_tables["all_schema_maps"]
    schema_maps_3d = tf.squeeze(tf.gather(all_schema_maps, ids), [1])
    schema_lengths = schema_tables["schema_lengths"]
    schema_attn_values_length = tf.squeeze(tf.gather(schema_lengths, ids), [1])
    schema_map_lengths = schema_tables["schema_map_lengths"]
    schema_map_attn_values_length = tf.squeeze(
      tf.gather(schema_map_lengths, ids), [1])

    return (schema_embeddings_3d, schema_embeddings_3d,
            schema_attn_values_length, schema_maps_3d,
            schema_maps_3d, schema_map_attn_values_length)

  def _create_decoder(self, encoder_output, features, _labels):
    (params, mode, vocab_size, attention_values, attention_values_length,
     attention_keys, attention_fn, reverse_scores_lengths,
     schema_attention_fn) = self._get_decoder_args(encoder_output,
                                                   features, _labels)
    (schema_attention_keys, schema_attention_values,
     schema_attention_values_length,
     schema_map_attention_keys, schema_map_attention_values,
     schema_map_attention_values_length) = self._schema_lookups(features)

    schema_map_attention_class = locate(
      self.params["schema_map.attention.class"]) or \
      getattr(decoders.attention, self.params["schema.attention.class"])
    schema_map_attention_layer = schema_map_attention_class(
        params=self.params["schema_map.attention.params"], mode=self.mode)

    return self.decoder_class(
        params=params,
        mode=mode,
        vocab_size=vocab_size,
        attention_values=attention_values,
        attention_values_length=attention_values_length,
        attention_keys=attention_keys,
        attention_fn=attention_fn,
        reverse_scores_lengths=reverse_scores_lengths,
        schema_attention_keys=schema_attention_keys,
        schema_attention_values=schema_attention_values,
        schema_attention_values_length=schema_attention_values_length,
        schema_attention_fn=schema_attention_fn,
        schema_map_attention_keys=schema_map_attention_keys,
        schema_map_attention_values=schema_map_attention_values,
        schema_map_attention_values_length=schema_map_attention_values_length,
        schema_map_attention_fn=schema_map_attention_layer
    )
