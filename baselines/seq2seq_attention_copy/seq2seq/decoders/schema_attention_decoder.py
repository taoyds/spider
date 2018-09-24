# Modified from attention_decoder.py
"""
A sequence decoder with attention to schema that performs
a softmax based on the RNN state.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from seq2seq.decoders.rnn_decoder import RNNDecoder

from seq2seq.contrib.seq2seq.helper import CustomHelper


class SchemaAttentionDecoderOutput(
    # added namedtuple: schema_attention_scores", "schema_attention_context"
    namedtuple("DecoderOutput", [
        "logits", "predicted_ids", "cell_output", "attention_scores",
        "attention_context", "schema_attention_scores",
        "schema_attention_context"
    ])):
  """Augmented decoder output that also includes the attention scores.
  """
  pass

class SchemaCopyingAttentionDecoderOutput(
    # added namedtuple: schema_attention_copy_vals
    namedtuple("DecoderOutput", [
        "logits", "predicted_ids", "cell_output", "attention_scores",
        "attention_context", "schema_attention_scores",
        "schema_attention_context", "schema_attention_copy_vals"
    ])):
  """Augmented decoder output that also includes the attention scores
     and copy vals.
  """
  pass

class SchemaMapAttentionDecoderOutput(
    # added namedtuple: "schema_map_attention_scores", "schema_map_attention_context"
    namedtuple("DecoderOutput", [
        "logits", "predicted_ids", "cell_output", "attention_scores",
        "attention_context", "schema_attention_scores",
        "schema_attention_context","schema_map_attention_scores",
        "schema_map_attention_context"
    ])):
  """Augmented decoder output that also includes the attention scores.
  """
  pass


class SchemaAttentionDecoder(RNNDecoder):
  """An RNN Decoder that uses attention over an input sequence and a schema.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_keys: The sequence used to calculate attention scores.
      A tensor of shape `[B, T, ...]`.
    attention_values: The sequence to attend over.
      A tensor of shape `[B, T, input_dim]`.
    attention_values_length: Sequence length of the attention values.
      An int32 Tensor of shape `[B]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    reverse_scores: Optional, an array of sequence length. If set,
      reverse the attention scores in the output. This is used for when
      a reversed source sequence is fed as an input but you want to
      return the scores in non-reversed order.
  """
  # the definition of schema_attention_function is in models/schema_attention_seq2seq.py
  def __init__(self,
               params,
               mode,
               vocab_size,
               attention_keys,
               attention_values,
               attention_values_length,
               attention_fn,
               # 4 extra values
               reverse_scores_lengths=None,
               schema_attention_keys=None,
               schema_attention_values=None,
               schema_attention_values_length=None,
               schema_attention_fn=None,
               name="schema_attention_decoder"):
    super(SchemaAttentionDecoder, self).__init__(params, mode, name)
    self.vocab_size = vocab_size
    self.attention_keys = attention_keys
    self.attention_values = attention_values
    self.attention_values_length = attention_values_length
    self.attention_fn = attention_fn
    self.reverse_scores_lengths = reverse_scores_lengths
    self.schema_attention_keys = schema_attention_keys
    self.schema_attention_values = schema_attention_values
    self.schema_attention_values_length = schema_attention_values_length
    if schema_attention_fn:
      self.schema_attention_fn = schema_attention_fn
    else:
      self.schema_attention_fn = attention_fn

  @property
  def output_size(self):
    return SchemaAttentionDecoderOutput(
      logits=self.vocab_size,
      predicted_ids=tf.TensorShape([]),
      cell_output=self.cell.output_size,
      attention_scores=tf.shape(self.attention_values)[1:-1],
      attention_context=self.attention_values.get_shape()[-1],
      schema_attention_scores=tf.shape(self.schema_attention_values)[1:-1],
      schema_attention_context=self.schema_attention_values.get_shape()[-1])

  @property
  def output_dtype(self):
    return SchemaAttentionDecoderOutput(
      logits=tf.float32,
      predicted_ids=tf.int32,
      cell_output=tf.float32,
      attention_scores=tf.float32,
      attention_context=tf.float32,
      schema_attention_scores=tf.float32,
      schema_attention_context=tf.float32)

  def initialize(self, name=None):
    finished, first_inputs = self.helper.initialize()

    # Concat empty attention context
    attention_context = tf.zeros([
        tf.shape(first_inputs)[0],
        self.attention_values.get_shape().as_list()[-1]
    ])
    schema_attention_context = tf.zeros([
        tf.shape(first_inputs)[0],
        self.schema_attention_values.get_shape().as_list()[-1]
    ])
    first_inputs = tf.concat([first_inputs, attention_context, schema_attention_context], 1)

    return finished, first_inputs, self.initial_state

  def compute_output(self, cell_output, calculate_softmax=True):
    """Computes the decoder outputs."""

    # Compute attention
    att_scores, attention_context = self.attention_fn(
        query=cell_output,
        keys=self.attention_keys,
        values=self.attention_values,
        values_length=self.attention_values_length)
    # there is a key and a schema attention value
    # which is key? where to find the schema attention function?
    schema_att_scores, schema_attention_context = self.schema_attention_fn(
        query=cell_output,
        keys=self.schema_attention_keys,
        values=self.schema_attention_values,
        values_length=self.schema_attention_values_length)

    softmax_input = None
    logits = None
    if calculate_softmax:
      softmax_input, logits = self._calculate_softmax(
        [cell_output, attention_context, schema_attention_context])

    return softmax_input, logits, att_scores, attention_context, schema_att_scores, schema_attention_context

  def _calculate_softmax(self, list_of_contexts):
    softmax_input = tf.contrib.layers.fully_connected(
          inputs=tf.concat(list_of_contexts, 1),
          num_outputs=self.cell.output_size,
          activation_fn=tf.nn.tanh,
          scope="attention_mix")

    # Softmax computation
    logits = tf.contrib.layers.fully_connected(
          inputs=softmax_input,
          num_outputs=self.vocab_size,
          activation_fn=None,
          scope="logits")
    return softmax_input, logits

  def _setup(self, initial_state, helper):
    self.initial_state = initial_state

    def att_next_inputs(time, outputs, state, sample_ids, name=None):
      """Wraps the original decoder helper function to append the attention
      context.
      """
      finished, next_inputs, next_state = helper.next_inputs(
          time=time,
          outputs=outputs,
          state=state,
          sample_ids=sample_ids,
          name=name)
      next_inputs = tf.concat([next_inputs, outputs.attention_context, outputs.schema_attention_context], 1)
      return (finished, next_inputs, next_state)

    self.helper = CustomHelper(
        initialize_fn=helper.initialize,
        sample_fn=helper.sample,
        next_inputs_fn=att_next_inputs)

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self.cell(inputs, state)
    (cell_output_new, logits, attention_scores, attention_context,
     schema_attention_scores, schema_attention_context) = \
    self.compute_output(cell_output)

    if self.reverse_scores_lengths is not None:
      attention_scores = tf.reverse_sequence(
          input=attention_scores,
          seq_lengths=self.reverse_scores_lengths,
          seq_dim=1,
          batch_dim=0)

    sample_ids = self.helper.sample(
        time=time_, outputs=logits, state=cell_state)

    outputs = SchemaAttentionDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids,
        cell_output=cell_output_new,
        attention_scores=attention_scores,
        attention_context=attention_context,
        schema_attention_scores=schema_attention_scores,
        schema_attention_context=schema_attention_context)

    finished, next_inputs, next_state = self.helper.next_inputs(
        time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

    return (outputs, next_state, next_inputs, finished)

class SchemaAttentionCopyingDecoder(SchemaAttentionDecoder):
  """
  The version of SchemaAttentionCopyingDecoder that uses
  F(score_n, rowembedding_n, h, c, W) to generate a score for the
  n-th field in the schema.
  """
  def __init__(self,
               params,
               mode,
               vocab_size,
               attention_keys,
               attention_values,
               attention_values_length,
               attention_fn,
               reverse_scores_lengths=None,
               schema_attention_keys=None,
               schema_attention_values=None,
               schema_attention_values_length=None,
               schema_attention_fn=None,
               name="schema_attention_copying_decoder"):
    super(SchemaAttentionCopyingDecoder, self).__init__(
      params, mode, vocab_size, attention_keys, attention_values,
      attention_values_length, attention_fn, reverse_scores_lengths,
      schema_attention_keys, schema_attention_values,
      schema_attention_values_length, schema_attention_fn, name)
    self.schema_embs = schema_attention_values

  @property
  def output_size(self):
    return SchemaCopyingAttentionDecoderOutput(
      logits=self.vocab_size,
      predicted_ids=tf.TensorShape([]),
      cell_output=self.cell.output_size,
      attention_scores=tf.shape(self.attention_values)[1:-1],
      attention_context=self.attention_values.get_shape()[-1],
      schema_attention_scores=tf.shape(self.schema_attention_values)[1:-1],
      schema_attention_context=self.schema_attention_values.get_shape()[-1],
      schema_attention_copy_vals=tf.shape(self.schema_attention_values)[1:-1])

  @property
  def output_dtype(self):
    return SchemaCopyingAttentionDecoderOutput(
      logits=tf.float32,
      predicted_ids=tf.int32,
      cell_output=tf.float32,
      attention_scores=tf.float32,
      attention_context=tf.float32,
      schema_attention_scores=tf.float32,
      schema_attention_context=tf.float32,
      schema_attention_copy_vals=tf.float32)

  def compute_output(self, cell_output):
    (softmax_input, logits, att_scores,
     attention_context, schema_att_scores,
     schema_attention_context) = super(
       SchemaAttentionCopyingDecoder, self).compute_output(cell_output)
    schema_attention_copy_vals = schema_att_scores
    weighted_schema_embs_size = self.cell.output_size + \
                                self.attention_values.get_shape().as_list()[-1]
    weighted_schema_embs = tf.contrib.layers.fully_connected(
        inputs=self.schema_embs,
        num_outputs=weighted_schema_embs_size,
        activation_fn=None,
        scope="weighted_schema_embs")

    concatenated = tf.expand_dims(
      tf.concat([cell_output, attention_context], 1), axis=2)
    schema_attention_copy_vals = schema_att_scores * tf.squeeze(tf.matmul(
      weighted_schema_embs, concatenated), axis=2)

    return softmax_input, logits, att_scores, attention_context, schema_att_scores, schema_attention_context, schema_attention_copy_vals

  def _setup(self, initial_state, helper):
    #TODO: Take advantage of inheritance rather than copy-paste
    self.initial_state = initial_state

    def att_next_inputs(time, outputs, state, sample_ids, name=None):
      """Wraps the original decoder helper function to append the attention
      context.
      """
      finished, next_inputs, next_state = helper.next_inputs(
          time=time,
          outputs=outputs,
          state=state,
          sample_ids=sample_ids,
          name=name)
      next_inputs = tf.concat([next_inputs, outputs.attention_context, outputs.schema_attention_context], 1)
      return (finished, next_inputs, next_state)

    self.helper = CustomHelper(
        initialize_fn=helper.initialize,
        sample_fn=helper.sample,
        next_inputs_fn=att_next_inputs)

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self.cell(inputs, state)
    (cell_output_new, logits, attention_scores, attention_context,
     schema_attention_scores, schema_attention_context,
     schema_attention_copy_vals) = \
    self.compute_output(cell_output)

    if self.reverse_scores_lengths is not None:
      attention_scores = tf.reverse_sequence(
          input=attention_scores,
          seq_lengths=self.reverse_scores_lengths,
          seq_dim=1,
          batch_dim=0)

    sample_ids = self.helper.sample(
        time=time_, outputs=logits, state=cell_state)
    outputs = SchemaCopyingAttentionDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids,
        cell_output=cell_output_new,
        attention_scores=attention_scores,
        attention_context=attention_context,
        schema_attention_scores=schema_attention_scores,
        schema_attention_context=schema_attention_context,
        schema_attention_copy_vals=schema_attention_copy_vals)

    finished, next_inputs, next_state = self.helper.next_inputs(
      time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)
    return (outputs, next_state, next_inputs, finished)

class SchemaMapAttentionDecoder(SchemaAttentionDecoder):
  """An RNN Decoder that uses attention over an input sequence and a schema
  and a schema map.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_keys: The sequence used to calculate attention scores.
      A tensor of shape `[B, T, ...]`.
    attention_values: The sequence to attend over.
      A tensor of shape `[B, T, input_dim]`.
    attention_values_length: Sequence length of the attention values.
      An int32 Tensor of shape `[B]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    reverse_scores: Optional, an array of sequence length. If set,
      reverse the attention scores in the output. This is used for when
      a reversed source sequence is fed as an input but you want to
      return the scores in non-reversed order.
  """

  def __init__(self,
               params,
               mode,
               vocab_size,
               attention_keys,
               attention_values,
               attention_values_length,
               attention_fn,
               reverse_scores_lengths=None,
               schema_attention_keys=None,
               schema_attention_values=None,
               schema_attention_values_length=None,
               schema_attention_fn=None,
               schema_map_attention_keys=None,
               schema_map_attention_values=None,
               schema_map_attention_values_length=None,
               schema_map_attention_fn=None,
               name="schema_map_attention_decoder"):
    super(SchemaMapAttentionDecoder, self).__init__(
      params, mode, vocab_size, attention_keys, attention_values,
      attention_values_length, attention_fn, reverse_scores_lengths,
      schema_attention_keys, schema_attention_values,
      schema_attention_values_length, schema_attention_fn, name)
    self.schema_map_attention_keys = schema_attention_keys
    self.schema_map_attention_values = schema_attention_values
    self.schema_map_attention_values_length = schema_attention_values_length
    if schema_map_attention_fn:
      self.schema_map_attention_fn = schema_map_attention_fn
    else:
      self.schema_map_attention_fn = attention_fn

  @property
  def output_size(self):
    return SchemaMapAttentionDecoderOutput(
      logits=self.vocab_size,
      predicted_ids=tf.TensorShape([]),
      cell_output=self.cell.output_size,
      attention_scores=tf.shape(self.attention_values)[1:-1],
      attention_context=self.attention_values.get_shape()[-1],
      schema_attention_scores=tf.shape(self.schema_attention_values)[1:-1],
      schema_attention_context=self.schema_attention_values.get_shape()[-1],
      schema_map_attention_scores=tf.shape(self.schema_map_attention_values)[1:-1],
      schema_map_attention_context=self.schema_map_attention_values.get_shape()[-1])

  @property
  def output_dtype(self):
    return SchemaMapAttentionDecoderOutput(
        logits=tf.float32,
        predicted_ids=tf.int32,
        cell_output=tf.float32,
        attention_scores=tf.float32,
        attention_context=tf.float32,
        schema_attention_scores=tf.float32,
        schema_attention_context=tf.float32,
        schema_map_attention_scores=tf.float32,
        schema_map_attention_context=tf.float32)

  def initialize(self, name=None):
    (finished, first_inputs,
     initial_state) = super(
       SchemaMapAttentionDecoder, self).initialize(name=name)

    # Concat empty schema map attention context
    schema_map_attention_context = tf.zeros([
        tf.shape(first_inputs)[0],
        self.schema_attention_values.get_shape().as_list()[-1]
    ])
    first_inputs = tf.concat([first_inputs, schema_map_attention_context], 1)

    return finished, first_inputs, initial_state

  def compute_output(self, cell_output, calculate_softmax=True):
    (softmax_input, logits, att_scores, attention_context,
     schema_att_scores, schema_attention_context) = super(
       SchemaMapAttentionDecoder, self).compute_output(cell_output,
                                                       calculate_softmax=False)
    with tf.variable_scope("schema_map_att"):
      (schema_map_att_scores,
       schema_map_attention_context) = self.schema_map_attention_fn(
         query=cell_output,
         keys=self.schema_map_attention_keys,
         values=self.schema_map_attention_values,
         values_length=self.schema_map_attention_values_length)

    if calculate_softmax:
      softmax_input, logits = self._calculate_softmax(
        [cell_output, attention_context, schema_attention_context,
         schema_map_attention_context])

    return (softmax_input, logits, att_scores, attention_context,
            schema_att_scores, schema_attention_context,
            schema_map_att_scores, schema_map_attention_context)

  # TODO: Can we use inheritance to make this simpler?
  def _setup(self, initial_state, helper):
    self.initial_state = initial_state

    def att_next_inputs(time, outputs, state, sample_ids, name=None):
      """Wraps the original decoder helper function to append the attention
      context.
      """
      finished, next_inputs, next_state = helper.next_inputs(
          time=time,
          outputs=outputs,
          state=state,
          sample_ids=sample_ids,
          name=name)
      next_inputs = tf.concat([next_inputs, outputs.attention_context, outputs.schema_attention_context, outputs.schema_map_attention_context], 1)
      return (finished, next_inputs, next_state)

    self.helper = CustomHelper(
        initialize_fn=helper.initialize,
        sample_fn=helper.sample,
        next_inputs_fn=att_next_inputs)

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self.cell(inputs, state)
    (cell_output_new, logits, attention_scores, attention_context,
     schema_attention_scores, schema_attention_context,
     schema_map_attention_scores, schema_map_attention_context) = \
      self.compute_output(cell_output)

    if self.reverse_scores_lengths is not None:
      attention_scores = tf.reverse_sequence(
          input=attention_scores,
          seq_lengths=self.reverse_scores_lengths,
          seq_dim=1,
          batch_dim=0)

    sample_ids = self.helper.sample(
        time=time_, outputs=logits, state=cell_state)

    outputs = SchemaMapAttentionDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids,
        cell_output=cell_output_new,
        attention_scores=attention_scores,
        attention_context=attention_context,
        schema_attention_scores=schema_attention_scores,
        schema_attention_context=schema_attention_context,
        schema_map_attention_scores=schema_map_attention_scores,
        schema_map_attention_context=schema_map_attention_context)

    finished, next_inputs, next_state = self.helper.next_inputs(
        time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

    return (outputs, next_state, next_inputs, finished)
