"""
Sequence to Sequence models with attention-based copying of input
and/or schema.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate

import tensorflow as tf
from tensorflow.python.ops import math_ops

import numpy as np

from seq2seq import decoders
from seq2seq.models.attention_seq2seq import AttentionSeq2Seq
from seq2seq import losses as seq2seq_losses
from seq2seq import graph_utils

class BaseAttentionCopyingSeq2Seq(AttentionSeq2Seq):
    """Base class for Sequence2Sequence model with attention-based copying.
    Args:
      source_vocab_info: An instance of `VocabInfo`
        for the source vocabulary
      target_vocab_info: An instance of `VocabInfo`
        for the target vocabulary
      params: A dictionary of hyperparameters
    """

    def __init__(self, params, mode, name="att_copying_seq2seq"):
        super(BaseAttentionCopyingSeq2Seq, self).__init__(params, mode, name)
        self._global_step = tf.train.get_global_step()

    @staticmethod
    def default_params():
        params = AttentionSeq2Seq.default_params().copy()
        params.update({
            # "decoder.class": "seq2seq.decoders.SchemaAttentionCopyingDecoder",
            "decoder.class": "seq2seq.decoders.AttentionDecoder",
            "schema.attention.class":"AttentionLayerBahdanau",
            "schema.attention.params": {"num_units": 150}
        })
        return params

    def _create_decoder(self, encoder_output, features, _labels):
        # assert "decoder_mask" in features
        decoder_mask = features["decoder_mask"]
        print ("decoder mask type", decoder_mask.get_shape())
        attention_class = locate(self.params["attention.class"]) or \
                          getattr(decoders.attention,
                                  self.params["attention.class"])
        attention_layer = attention_class(
            params=self.params["attention.params"], mode=self.mode,
            name="attention_to_input_layer")

        # If the input sequence is reversed we also need to reverse
        # the attention scores.
        reverse_scores_lengths = None
        if self.params["source.reverse"]:
            reverse_scores_lengths = features["source_len"]
        if self.use_beam_search:
            reverse_scores_lengths = tf.tile(
                input=reverse_scores_lengths,
                multiples=[self.params["inference.beam_search.beam_width"]])

        if "schema" in self.params["decoder.class"].lower():
            schema_attention_class = locate(
                self.params["schema.attention.class"]) or \
                getattr(decoders.attention,
                        self.params["schema.attention.class"])
            schema_attention_layer = schema_attention_class(
                params=self.params["schema.attention.params"], mode=self.mode,
                name="attention_to_schema_layer")

            schema_tables = graph_utils.get_dict_from_collection("schema_tables")
            schema_locs = features['schema_loc']
            table = schema_tables["schema_file_lookup_table"]
            ids = table.lookup(schema_locs)
            all_schema_embeddings = schema_tables["all_schema_embeddings"]
            schema_embeddings_3d = tf.squeeze(tf.gather(
                all_schema_embeddings, ids), [1])
            schema_lengths = schema_tables["schema_lengths"]
            schema_attn_values_length = tf.squeeze(tf.gather(
                schema_lengths, ids), [1])
            

            return self.decoder_class(
                params=self.params["decoder.params"],
                mode=self.mode,
                vocab_size=self.target_vocab_info.total_size,
                attention_values=encoder_output.attention_values,
                attention_values_length=encoder_output.attention_values_length,
                attention_keys=encoder_output.outputs,
                attention_fn=attention_layer,
                reverse_scores_lengths=reverse_scores_lengths,
                schema_attention_keys=schema_embeddings_3d,
                schema_attention_values=schema_embeddings_3d,
                schema_attention_values_length=schema_attn_values_length,
                schema_attention_fn=schema_attention_layer,
                decoder_mask = decoder_mask
            )
        return self.decoder_class(
            params=self.params["decoder.params"],
            mode=self.mode,
            vocab_size=self.target_vocab_info.total_size,
            attention_values=encoder_output.attention_values,
            attention_values_length=encoder_output.attention_values_length,
            attention_keys=encoder_output.outputs,
            attention_fn=attention_layer,
            reverse_scores_lengths=reverse_scores_lengths,
            decoder_mask = decoder_mask
        )

    def _trim_copy_indices(self, labels, indices_name):
        # Slices source copy indices to max length
        if self.params["target.max_seq_len"] is not None:
            labels[indices_name] = labels[
                indices_name][:, :self.params["target.max_seq_len"]]
        return labels

    def _set_special_vocab_ids(self):
        # Find out the IDs for special vocab
        vocab_tables = graph_utils.get_dict_from_collection("vocab_tables")
        target_vocab_to_id = vocab_tables["target_vocab_to_id"]
        self.copy_word_id = target_vocab_to_id.lookup(
            tf.convert_to_tensor("COPY_WORD", dtype=tf.string))
        self.copy_schema_id = target_vocab_to_id.lookup(
            tf.convert_to_tensor("COPY_SCHEMA", dtype=tf.string))

    def _get_predicted_tokens(self, predictions):
        if "predicted_tokens" in predictions.keys():
            output_predicted_tokens = predictions["predicted_tokens"]
        else:
            vocab_tables = graph_utils.get_dict_from_collection("vocab_tables")
            target_id_to_vocab = vocab_tables["target_id_to_vocab"]
            output_predicted_tokens = target_id_to_vocab.lookup(
                tf.to_int64(predictions["predicted_ids"]))
        return output_predicted_tokens

    def _fill_in_copies(self, prediction_ids, predicted_tokens,
                        attn_scores, copy_source_tokens,
                        copy_id):
        predicted_copy_indices = math_ops.cast(
            math_ops.argmax(attn_scores, axis=-1), tf.int32)

        # Now index into the input sequences using those indices.
        # tf.gather_nd will need [row_num, index] format to index
        # into the input sequences, so need to build 3d tensor,
        # batch x output_seq_len x 2.
        row_nums_3d = tf.expand_dims(tf.tile(tf.expand_dims(
            tf.range(0, limit=tf.shape(predicted_copy_indices)[0]),
            axis=1), [1,tf.shape(predicted_copy_indices)[1]]), 2)
        indices_3d = tf.expand_dims(predicted_copy_indices, 2)
        indices_3d = tf.concat([row_nums_3d, indices_3d], axis=2)
        copy_predicted_tokens = tf.gather_nd(copy_source_tokens, indices_3d)

        # Where prediction_id == copy_id, fill in the token
        # the model predicts will be copied. In all other locations,
        # keep the predicted token we already had.
        copy_map = tf.equal(prediction_ids, tf.to_int32(copy_id))
        predicted_tokens = tf.where(copy_map, copy_predicted_tokens,
                                    predicted_tokens)
        return predicted_tokens

    def _copy_loss(self, targets, seq_len, attention_scores,
                   copy_indices, copy_id):
        copy_logits=attention_scores[:, :, :]
        copy_targets=tf.transpose(copy_indices[:, 1:], [1, 0])

        copy_loss = seq2seq_losses.cross_entropy_sequence_loss(
            logits=copy_logits,
            targets=copy_targets,
            sequence_length=seq_len)

        copy_mask = tf.equal(targets, copy_id,
                             "target_equals_copy_id")
        copy_mask = tf.to_float(copy_mask, "copy_mask_to_float")
        masked_loss = copy_loss * copy_mask
        return masked_loss

    def _targets_and_seq_len(self, labels):
        targets = tf.transpose(labels["target_ids"][:, 1:], [1, 0])
        seq_length = labels["target_len"] - 1
        return targets, seq_length

    def compute_loss(self, decoder_output, _features, labels):
        """Computes the sequence loss for this model.
        seq_loss is the cross entropy loss for the output sequence.
        Returns a tuple `(losses, loss)`, where `losses` are the per-batch
        losses and loss is a single scalar tensor to minimize.
        """
        targets, seq_len = self._targets_and_seq_len(labels)
        seq_loss = seq2seq_losses.cross_entropy_sequence_loss(
            logits=decoder_output.logits[:, :, :],
            targets=targets,
            sequence_length=seq_len)
        return seq_loss

class SchemaAttentionCopyingSeq2Seq(BaseAttentionCopyingSeq2Seq):
    """Sequence2Sequence model with attention-based copying from schema.
    Args:
      source_vocab_info: An instance of `VocabInfo`
        for the source vocabulary
      target_vocab_info: An instance of `VocabInfo`
        for the target vocabulary
      params: A dictionary of hyperparameters
    """

    def __init__(self, params, mode, name="schema_att_copying_seq2seq"):
        super(SchemaAttentionCopyingSeq2Seq, self).__init__(params, mode, name)
        self.copy_schema_id = None

    def _preprocess(self, features, labels):
        """Model-specific preprocessing for features and labels:
        - Creates vocabulary lookup tables for source and target vocab
        - Converts tokens into vocabulary ids
        - Trims copy indices to target.max_seq_len
        """
        features, labels = super(SchemaAttentionCopyingSeq2Seq,
                                 self)._preprocess(features, labels)

        if not labels or not "schema_copy_indices" in labels:
            return features, labels

        labels = self._trim_copy_indices(labels, "schema_copy_indices")
        self._set_special_vocab_ids()
        return features, labels

    def _create_predictions(self, decoder_output, features, labels,
                            losses=None):
        predictions = super(SchemaAttentionCopyingSeq2Seq,
                            self)._create_predictions(decoder_output,
                                                      features, labels, losses)
        if "predicted_ids" in predictions.keys():
            prediction_ids = predictions["predicted_ids"]
            output_predicted_tokens = self._get_predicted_tokens(predictions)
            if self.copy_schema_id is None:
                self._set_special_vocab_ids()

            schema_attention_copy_vals = predictions["schema_attention_copy_vals"]

            # Build the schema_tokens structure from the list of schema locations.
            schema_tables = graph_utils.get_dict_from_collection("schema_tables")
            schema_locs = features['schema_loc']
            table = schema_tables["schema_file_lookup_table"]
            ids = table.lookup(schema_locs)
            all_schema_strings = schema_tables["all_schema_strings"]
            schema_tokens = tf.squeeze(all_schema_strings.lookup(ids), axis=1)
            schema_tokens = tf.sparse_tensor_to_dense(
                tf.string_split(schema_tokens), default_value="UNK",
                name="desparsifying_schema_tokens")

            # Figure out copy_schema_predicted_tokens
            predicted_tokens = self._fill_in_copies(
                prediction_ids, output_predicted_tokens,
                schema_attention_copy_vals,
                schema_tokens, self.copy_schema_id)

            predictions["predicted_tokens"] = predicted_tokens
        return predictions

    def compute_loss(self, decoder_output, _features, labels):
        """Computes the loss for this model.
        Loss = seq_loss + schema_copy_loss.
        seq_loss is the cross entropy loss for the output sequence.
        schema_copy_loss is zero at any time step where output is not
        copy_schema, and the cross entropy loss for the schema attention
        score otherwise.
        Returns a tuple `(losses, loss)`, where `losses` are the per-batch
        losses and loss is a single scalar tensor to minimize.
        """
        seq_loss = super(SchemaAttentionCopyingSeq2Seq, self).compute_loss(
            decoder_output, _features, labels)

        targets, seq_length = self._targets_and_seq_len(labels)
        schema_copy_loss = self._copy_loss(
            targets, seq_length, decoder_output.schema_attention_copy_vals,
            labels["schema_copy_indices"], self.copy_schema_id)

        losses = seq_loss + schema_copy_loss

        # Calculate the average log perplexity
        loss = tf.reduce_sum(losses) / tf.to_float(
            tf.reduce_sum(labels["target_len"] - 1))

        return losses, loss

class InputAttentionCopyingSeq2Seq(BaseAttentionCopyingSeq2Seq):
    """Sequence2Sequence model with attention-based copying from input sequence.
    Args:
      source_vocab_info: An instance of `VocabInfo`
        for the source vocabulary
      target_vocab_info: An instance of `VocabInfo`
        for the target vocabulary
      params: A dictionary of hyperparameters
    """

    def __init__(self, params, mode, name="input_att_copying_seq2seq"):
        super(InputAttentionCopyingSeq2Seq, self).__init__(params, mode, name)
        self.copy_word_id = None

    def _preprocess(self, features, labels):
        """Model-specific preprocessing for features and labels:
        - Creates vocabulary lookup tables for source and target vocab
        - Converts tokens into vocabulary ids
        - Trims copy indices to target.max_seq_len
        """
        features, labels = super(InputAttentionCopyingSeq2Seq,
                                 self)._preprocess(features, labels)
        if not labels or not "source_copy_indices" in labels:
            return features, labels

        # Slices source copy indices to max length
        labels = self._trim_copy_indices(labels, "source_copy_indices")

        self._set_special_vocab_ids()
        return features, labels

    def _create_predictions(self, decoder_output, features, labels, losses=None):
        predictions = super(InputAttentionCopyingSeq2Seq,
                            self)._create_predictions(decoder_output,
                                                      features, labels, losses)
        
        print ("predictions", predictions["predicted_ids"])
        if "predicted_ids" in predictions.keys():
            prediction_ids = predictions["predicted_ids"]
            output_predicted_tokens = self._get_predicted_tokens(predictions)
            if self.copy_word_id is None:
                self._set_special_vocab_ids()

            # Figure out copy_word_predicted_tokens
            predicted_tokens = self._fill_in_copies(
                prediction_ids, output_predicted_tokens,
                predictions["attention_scores"],
                features["source_tokens"],
                self.copy_word_id)

            predictions["predicted_tokens"] = predicted_tokens
        return predictions

    def compute_loss(self, decoder_output, _features, labels):
        """Computes the loss for this model.
        Loss = seq_loss + word_copy_loss.
        seq_loss is the cross entropy loss for the output sequence.
        word_copy_loss is zero at any time step where output is not copy_word,
        and the cross entropy loss for the input attention score otherwise.
        Returns a tuple `(losses, loss)`, where `losses` are the per-batch
        losses and loss is a single scalar tensor to minimize.
        """
        seq_loss = super(InputAttentionCopyingSeq2Seq, self).compute_loss(
            decoder_output, _features, labels)
        targets, seq_length = self._targets_and_seq_len(labels)
        word_copy_loss = self._copy_loss(targets, seq_length,
                                         decoder_output.attention_scores,
                                         labels["source_copy_indices"],
                                         self.copy_word_id)

        losses = seq_loss + word_copy_loss

        # Calculate the average log perplexity
        loss = tf.reduce_sum(losses) / tf.to_float(
            tf.reduce_sum(labels["target_len"] - 1))

        return losses, loss


class SchemaAndInputAttentionCopyingSeq2Seq(SchemaAttentionCopyingSeq2Seq):
    """Sequence2Sequence model with attention-based copying from schema and input.
    Args:
      source_vocab_info: An instance of `VocabInfo`
        for the source vocabulary
      target_vocab_info: An instance of `VocabInfo`
        for the target vocabulary
      params: A dictionary of hyperparameters
    """

    def __init__(self, params, mode, name="schema_att_copying_seq2seq"):
        super(SchemaAndInputAttentionCopyingSeq2Seq, self).__init__(
            params, mode, name)
        self.copy_word_id = None

    def _preprocess(self, features, labels):
        """Model-specific preprocessing for features and labels:
        - Creates vocabulary lookup tables for source and target vocab
        - Converts tokens into vocabulary ids
        - Trims copy indices to target.max_seq_len
        """
        # features, labels already include schema-related
        # preprocessing, since this inherits from
        # SchemaAttentionCopyingSeq2Seq.
        features, labels = super(SchemaAndInputAttentionCopyingSeq2Seq,
                                 self)._preprocess(features, labels)

        if not labels or not "source_copy_indices" in labels:
            return features, labels

        # Slices source copy indices to max length
        labels = self._trim_copy_indices(labels, "source_copy_indices")
        self._set_special_vocab_ids()
        return features, labels

    def _create_predictions(self, decoder_output, features, labels, losses=None):
        predictions = super(SchemaAndInputAttentionCopyingSeq2Seq,
                            self)._create_predictions(decoder_output,
                                                      features, labels, losses)
        if "predicted_ids" in predictions.keys():
            prediction_ids = predictions["predicted_ids"]
            output_predicted_tokens = self._get_predicted_tokens(predictions)

            if self.copy_word_id is None:
                self._set_special_vocab_ids()

            # Figure out copy_word_predicted_tokens
            predicted_tokens = self._fill_in_copies(
                prediction_ids, output_predicted_tokens,
                predictions["attention_scores"],
                features["source_tokens"],
                self.copy_word_id)

            predictions["predicted_tokens"] = predicted_tokens
        return predictions

    def compute_loss(self, decoder_output, _features, labels):
        """Computes the loss for this model.
        Loss = seq_loss + schema_copy_loss + word_copy_loss.
        seq_loss is the cross entropy loss for the output sequence.
        word_copy_loss is zero at any time step where output is not copy_word,
        and the cross entropy loss for the input attention score otherwise.
        schema_copy_loss is like word_copy_loss, only for schema copying.
        Returns a tuple `(losses, loss)`, where `losses` are the per-batch
        losses and loss is a single scalar tensor to minimize.
        """
        seq_and_schema_loss, _ = super(SchemaAndInputAttentionCopyingSeq2Seq,
                                    self).compute_loss(decoder_output,
                                                       _features, labels)
        targets, seq_length = self._targets_and_seq_len(labels)
        word_copy_loss = self._copy_loss(targets, seq_length,
                                         decoder_output.attention_scores,
                                         labels["source_copy_indices"],
                                         self.copy_word_id)
        losses = seq_and_schema_loss + word_copy_loss

        # Calculate the average log perplexity
        loss = tf.reduce_sum(losses) / tf.to_float(
            tf.reduce_sum(labels["target_len"] - 1))

        return losses, loss
