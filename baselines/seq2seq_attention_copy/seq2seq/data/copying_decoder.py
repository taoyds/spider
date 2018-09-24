from __future__ import print_function

import numpy as np
import tensorflow as tf
from seq2seq.data import split_tokens_decoder
from seq2seq import graph_utils

class BaseCopyingDecoder(split_tokens_decoder.SplitTokensDecoder):
    """Base class for A DataDecoder that splits a string tensor into individual
    tokens and marks those copied from the input sequence or the schema.
    Optionally prepends or appends special tokens.

    Args:
      delimiter: Delimiter to split on. Must be a single character.
      tokens_feature_name: A descriptive feature name for the token values
      length_feature_name: A descriptive feature name for the length value
    """

    def decode(self, data, items):
        """
        Args:
          data: List of [target_string, source_tokens_list, schema_location].
          items: A list of strings, each of which indicate a particular data
            type.

        Returns: A dictionary with a tensor for each item in items.
        """
        if not isinstance(data, list):
            raise ValueError("'data' arg to decode should be a three-item list, but is a %s" % str(type(data)))
        if len(data) != 3:
            raise ValueError("'data' arg to decode should be a three-item list, but is a list of length %d" % len(data))

        decoded_items = {}

        # Split tokens
        tokens = tf.string_split([data[0]], delimiter=self.delimiter).values

        tokens, indices = self._mark_all_copies(tokens, data)

        # Optionally prepend a special token
        if self.prepend_token is not None:
            tokens, indices = self._prepend(tokens, indices)

        # Optionally append a special token
        if self.append_token is not None:
            tokens, indices = self._append(tokens, indices)

        decoded_items[self.length_feature_name] = tf.size(tokens)
        decoded_items[self.tokens_feature_name] = tokens
        decoded_items["indices"] = indices

        return decoded_items # dictionary

    def _prepend(self, tokens, indices):
        tokens = tf.concat([[self.prepend_token], tokens], 0,
                           name="prepend_to_tokens")
        return tokens, indices

    def _append(self, tokens, indices):
        tokens =  tf.concat([tokens, [self.append_token]], 0,
                            name="append_to_tokens")
        return tokens, indices

    def _mark_all_copies(self, tokens, data):
        raise NotImplementedError

    def _mark_copies(self, tokenized, copy_source, copy_token):
        """Replace any token in tokenized that can be copied from copy_source
        with the copy_token, and build a tensor with the indices of the copied
        item in the source.
        For instance, could be used with the query
          SELECT NUM_CREDITS FROM COURSE WHERE
          DEPARTMENT = " EECS " AND NUMBER = 280 ;
        and the question
          "Who teaches EECS 280 ?"
        to generate
          SELECT NUM_CREDITS FROM COURSE WHERE
          DEPARTMENT = " COPY_WORD " AND NUMBER = COPY_WORD ;
        and the indices
          [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0]
        Args:
          tokenized: a vector of tokens from the target string
          copy_source: a vector of tokens from the source string,
            or a vector of tokens from the schema.
          copy_token: the token with which to replace copies. E.g., "COPY_WORD"
            or "COPY_SCHEMA"
        Returns: tokenized with copied tokens replaced by the copy_token, and
          a vector of copying indices.
        """

        def copy_fn(tokens, copy_source):
            tokens_in_copy_source = np.in1d(tokens.ravel(), copy_source).reshape(tokens.shape)
            indices = [np.where(copy_source==t)[0] for t in tokens]
            indices = [t[0] if len(t) > 0 else 0 for t in indices]
            indices = np.asarray(indices)
            return tokens_in_copy_source, indices

        # Wrap the python function copy_fn and use it as a tf op.
        tf_copy_result = tf.py_func(copy_fn,
                                    [tokenized, copy_source], # inputs
                                    [tf.bool, tf.int64], # output types
                                    name="identify_copies_%s" % copy_token)
        copy_indices = tf_copy_result[1] # indices as returned from copy_fn

        # Make an array of the same shape as tokenized filled with the
        # copy token
        copy_tokens = tf.fill(tf.shape(tokenized), copy_token,
                              name="fill_with_%s"%copy_token)
        tokenized = tf.where(tf_copy_result[0], # bools from copy_fn
                             copy_tokens, # where true, copy copy_token
                             tokenized, # where false, copy from tokenized
                             name="mark_copied_%s_tokens" % copy_token)
        return tokenized, copy_indices

    def list_items(self):
        return [self.tokens_feature_name, self.length_feature_name]

class SchemaAndWordCopyingDecoder(BaseCopyingDecoder):
    """
    CopyingDecoder that marks where the output sequence copies from the input
    sequence and where it copies from the schema.

    Args:
      delimiter: Delimiter to split on. Must be a single character.
      tokens_feature_name: A descriptive feature name for the token values
      length_feature_name: A descriptive feature name for the length value
      source_copy_feature_name: A descriptive feature name for the indices
        representing where in the input sequence a word is copied from.
      schema_copy_feature_name: A descriptive feature name for the indices
        representing where in the schema a token is copied from.
      prepend_token: Optional token to prepend to output.
      append_token: Optional token to append to output.
    """
    def __init__(self,
                 delimiter=" ",
                 tokens_feature_name="tokens",
                 length_feature_name="length",
                 source_copy_feature_name="source_copy_indices",
                 schema_copy_feature_name="schema_copy_indices",
                 prepend_token=None,
                 append_token=None):
        super(SchemaAndWordCopyingDecoder, self).__init__(
            delimiter=delimiter, tokens_feature_name=tokens_feature_name,
            length_feature_name=length_feature_name,
            prepend_token=prepend_token, append_token=append_token)

        self.source_copy_feature_name = source_copy_feature_name
        self.schema_copy_feature_name = schema_copy_feature_name

        schema_tables = graph_utils.get_dict_from_collection("schema_tables")
        self.schema_lookup_table = schema_tables["schema_file_lookup_table"]
        self.schema_strings_table = schema_tables["all_schema_strings"]

    def decode(self, data, items):
        """
        Args:
          data: List of [target_string, source_tokens_list, schema_tokens_list].
          items: A list of strings, each of which indicate a particular data type.
        Returns: A tensor for each item in items.
        """
        decoded_items = super(SchemaAndWordCopyingDecoder, self).decode(
            data, items)
        indices = decoded_items.pop("indices")
        schema_copies_indices = indices[1]
        input_copies_indices = indices[0]
        decoded_items[self.schema_copy_feature_name] = schema_copies_indices
        decoded_items[self.source_copy_feature_name] = input_copies_indices
        return [decoded_items[_] for _ in items]

    def _prepend(self, tokens, indices):
        tokens, _ = super(SchemaAndWordCopyingDecoder, self)._prepend(
            tokens, indices)
        schema_copies_indices = indices[1]
        input_copies_indices = indices[0]
        input_copies_indices = tf.concat(
            [[0], input_copies_indices], 0,
            name="prepend_to_input_copies_indices")
        schema_copies_indices = tf.concat(
            [[0], schema_copies_indices], 0,
            name="prepend_to_schema_copies_indices")
        return tokens, [input_copies_indices, schema_copies_indices]

    def _append(self, tokens, indices):
        tokens, _ = super(SchemaAndWordCopyingDecoder, self)._append(tokens, indices)
        schema_copies_indices = indices[1]
        input_copies_indices = indices[0]
        input_copies_indices = tf.concat([input_copies_indices, [0]], 0,
                                         name="append_to_input_copies_indices")
        schema_copies_indices = tf.concat([schema_copies_indices, [0]], 0,
                                          name="append_to_schema_copies_indices")
        return tokens, [input_copies_indices, schema_copies_indices]

    def _mark_all_copies(self, tokens, data):
        words = data[1]
        # Note where we copy from input
        tokens, input_copies_indices = self._mark_copies(tokens, words,
                                                         "COPY_WORD")

        # Look up the schema string using the schema location
        schema_location = data[2][0]
        schema_id = self.schema_lookup_table.lookup(schema_location)
        schema_string = self.schema_strings_table.lookup(schema_id)
        schema = tf.string_split(schema_string, delimiter=" ").values

        # Note where we copy from schema
        tokens, schema_copies_indices = self._mark_copies(tokens, schema,
                                                         "COPY_SCHEMA")
        return tokens, [input_copies_indices, schema_copies_indices]

    def list_items(self):
        items = super(SchemaAndWordCopyingDecoder, self).list_items()
        items += [self.schema_copy_feature_name, self.source_copy_feature_name]
        return items

class SchemaCopyingDecoder(BaseCopyingDecoder):
    """
    CopyingDecoder that marks where the output sequence copies from the
    schema.

    Args:
      delimiter: Delimiter to split on. Must be a single character.
      tokens_feature_name: A descriptive feature name for the token values
      length_feature_name: A descriptive feature name for the length value
      schema_copy_feature_name: A descriptive feature name for the indices
        representing where in the schema a token is copied from.
      prepend_token: Optional token to prepend to output.
      append_token: Optional token to append to output.
    """
    def __init__(self,
                 delimiter=" ",
                 tokens_feature_name="tokens",
                 length_feature_name="length",
                 schema_copy_feature_name="schema_copy_indices",
                 prepend_token=None,
                 append_token=None):
        super(SchemaCopyingDecoder, self).__init__(
            delimiter=delimiter, tokens_feature_name=tokens_feature_name,
            length_feature_name=length_feature_name,
            prepend_token=prepend_token, append_token=append_token)
        self.schema_copy_feature_name = schema_copy_feature_name

        schema_tables = graph_utils.get_dict_from_collection("schema_tables")
        self.schema_lookup_table = schema_tables["schema_file_lookup_table"]
        self.schema_strings_table = schema_tables["all_schema_strings"]

    def decode(self, data, items):
        """
        Args:
          data: List of [target_string, None, schema_tokens_list].
          items: A list of strings, each of which indicate a particular data type.
        Returns: A tensor for each item in items.
        """
        decoded_items = super(SchemaCopyingDecoder, self).decode(data, items)
        indices = decoded_items.pop("indices")
        schema_copies_indices = indices[0]
        decoded_items[self.schema_copy_feature_name] = schema_copies_indices
        return [decoded_items[_] for _ in items]

    def _prepend(self, tokens, indices):
        tokens, _ = super(SchemaCopyingDecoder, self)._prepend(tokens, indices)
        schema_copies_indices = indices[0]
        schema_copies_indices = tf.concat(
            [[0], schema_copies_indices], 0,
            name="prepend_to_schema_copies_indices")
        return tokens, [schema_copies_indices]

    def _append(self, tokens, indices):
        tokens, _ = super(SchemaCopyingDecoder, self)._append(tokens, indices)
        schema_copies_indices = indices[0]
        schema_copies_indices = tf.concat(
            [schema_copies_indices, [0]], 0,
            name="append_to_schema_copies_indices")
        return tokens, [schema_copies_indices]

    def _mark_all_copies(self, tokens, data):
        # Look up the schema string using the schema location
        schema_location = data[2][0]
        schema_id = self.schema_lookup_table.lookup(schema_location)
        schema_string = self.schema_strings_table.lookup(schema_id)
        schema = tf.string_split(schema_string, delimiter=" ").values

        # Note where we copy from schema
        tokens, schema_copies_indices = self._mark_copies(tokens, schema,
                                                         "COPY_SCHEMA")
        return tokens, [schema_copies_indices]

    def list_items(self):
        items = super(SchemaCopyingDecoder, self).list_items()
        items += [self.schema_copy_feature_name]
        return items

class WordCopyingDecoder(BaseCopyingDecoder):
    """
    CopyingDecoder that marks where the output sequence copies from the input
    sequence.

    Args:
      delimiter: Delimiter to split on. Must be a single character.
      tokens_feature_name: A descriptive feature name for the token values
      length_feature_name: A descriptive feature name for the length value
      source_copy_feature_name: A descriptive feature name for the indices
        representing where in the input sequence a word is copied from.
      prepend_token: Optional token to prepend to output.
      append_token: Optional token to append to output.
    """
    def __init__(self,
                 delimiter=" ",
                 tokens_feature_name="tokens",
                 length_feature_name="length",
                 source_copy_feature_name="source_copy_indices",
                 prepend_token=None,
                 append_token=None):
        super(WordCopyingDecoder, self).__init__(
            delimiter=delimiter, tokens_feature_name=tokens_feature_name,
            length_feature_name=length_feature_name,
            prepend_token=prepend_token, append_token=append_token)
        self.source_copy_feature_name = source_copy_feature_name

    def decode(self, data, items):
        """
        Args:
          data: List of [target_string, source_tokens_list, None].
          items: A list of strings, each of which indicate a particular data type.
        Returns: A tensor for each item in items.
        """
        decoded_items = super(WordCopyingDecoder, self).decode(data, items)
        indices = decoded_items.pop("indices")
        input_copies_indices = indices[0]
        decoded_items[self.source_copy_feature_name] = input_copies_indices
        print ("decode", self.source_copy_feature_name)
        return [decoded_items[_] for _ in items]

    def _prepend(self, tokens, indices):
        tokens, _ = super(WordCopyingDecoder, self)._prepend(tokens, indices)
        input_copies_indices = indices[0]
        input_copies_indices = tf.concat(
            [[0], input_copies_indices], 0,
            name="prepend_to_input_copies_indices")
        return tokens, [input_copies_indices]

    def _append(self, tokens, indices):
        tokens, _ = super(WordCopyingDecoder, self)._append(tokens, indices)
        input_copies_indices = indices[0]
        input_copies_indices = tf.concat(
            [input_copies_indices, [0]], 0,
            name="append_to_input_copies_indices")
        return tokens, [input_copies_indices]

    def _mark_all_copies(self, tokens, data):
        words = data[1]
        # Note where we copy from input
        tokens, input_copies_indices = self._mark_copies(tokens, words,
                                                         "COPY_WORD")
        return tokens, [input_copies_indices]

    def list_items(self):
        items = super(WordCopyingDecoder, self).list_items()
        items += [self.source_copy_feature_name]
        return items
