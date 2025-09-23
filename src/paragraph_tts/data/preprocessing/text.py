# -*- coding: utf-8 -*-
"""Contains audio processing utilities."""
import dataclasses
import logging
import sys
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeAlias
import collections
import itertools
import re

import numpy as np
import gruut
import torch
from DeBERTa import deberta


def _logger():
    return logging.getLogger(__name__)


TokenMapping: TypeAlias = collections.OrderedDict[str, List[str]]


@dataclasses.dataclass
class TextFeatures:
    """Contains features extracted from text."""

    normalized_text: str
    words: List[str]
    word_phoneme_mapping: TokenMapping
    word_bert_mapping: TokenMapping

    def get_phoneme_sequence(self) -> List[str]:
        """Returns the full phoneme sequence for the text."""

        return list(itertools.chain(*self.word_phoneme_mapping.values()))

    def get_bert_token_sequence(self) -> List[str]:
        """Returns the full BERT token sequence for the text."""

        return list(itertools.chain(*self.word_bert_mapping.values()))

    def get_word_to_phoneme_spans(self) -> np.ndarray:
        """Returns spans mapping words to phonemes."""

        return np.ndarray([len(phonemes) for phonemes in self.word_phoneme_mapping.values()])

    def get_word_to_token_spans(self) -> np.ndarray:
        """Returns spans mapping words to BERT tokens."""

        return np.ndarray([len(tokens) for tokens in self.word_bert_mapping.values()])


def add_pauses(text_features: TextFeatures, pauses: List[Tuple[int, str]]):
    """Adds pauses to phonemes and updates word-phoneme spans.

    Args:
        text_features: Text features to modify.
        pauses: List of (word_index, pause_type) tuples.
    """

    for word_idx, pause_type in reversed(pauses):
        text_features.word_phoneme_mapping[text_features.words[word_idx]].append(pause_type)


@dataclasses.dataclass
class _WordStruct:
    """Contains word-level information during text processing.

    Each word represents a punctuation-less unit spanning a certain number of phonemes
    and BERT tokens.
    """

    # Word's textual content.
    text: str
    # List of phonemes in the word.
    phonemes: List[str]
    # Word's text with original punctuation (if any).
    text_with_punct: str


class TextProcessor:
    """Processes text data."""

    # Single punctuation marks to be attached to the leading word.
    single_puncts_replace = {
        ' . ': '. ',
        ' , ': ', ',
        ' ! ': '! ',
        ' ? ': '? ',
        ' " ': '" ',
        ' : ': ': ',
    }

    # Punctuation marks that should be moved outside from the quotation.
    puncts_before_quotes_replace = {
        '."': '".',
        '!"': '"!',
        '?"': '"?',
        ',"': '",',
        ':"': '":',
    }

    # Replacements for post-processing of word structs.
    puncts_after_quotes_replace = {
        '"!': '!"',
        '"?': '?"',
        '."': '."',
    }

    apostrophe_replacements = {
        ",'": ',"',
        ".'": '."',
        "!'": '!"',
        "?'": '?"',
        ":'": ':"',
        " '": ' "',
        "\"'": '""',
        "'\"": '""',
    }

    allowed_chars = (
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'abcdefghijklmnopqrstuvwxyz'
        '0123456789'
        " .,:!?'\"$€"
    )

    allowed_chars_for_word_repr = (
        'abcdefghijklmnopqrstuvwxyz'
        "'"
    )

    def __init__(self):
        """Inits the text processor."""

        vocab_path, vocab_type = deberta.load_vocab(pretrained_id='xxlarge-v2')
        self._tokenizer = deberta.tokenizers[vocab_type](vocab_path)

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans the text by removing unwanted characters and fixing quotes."""

        if text.startswith("'"):
            text = '"' + text[1:]

        if text.endswith("'"):
            text = text[:-1] + '"'

        for pattern, replacement in TextProcessor.apostrophe_replacements.items():
            text = text.replace(pattern, replacement)

        text = text.replace('-', ' ')

        text = ''.join(filter(lambda x: x in TextProcessor.allowed_chars, text))
        text = ' '.join(text.split())

        return text

    @staticmethod
    def load_text(text_path: str) -> str:
        """Loads text from a file."""

        with open(text_path, 'r', encoding='utf-8') as text_f:
            return text_f.read().strip()

    @staticmethod
    def get_pause_type(pause_length: float) -> str:
        """Returns the type of pause based on its length."""

        if pause_length < 0.2:
            return '<short_pause>'

        if pause_length < 0.7:
            return '<medium_pause>'

        return '<long_pause>'

    def tokenize_text(self, text: str) -> TextFeatures:
        """Processes and tokenizes text."""

        normalized_text = self._prepare_for_tokenization(text)

        word_structs = self._get_word_structs(normalized_text)

        words = []
        word_phoneme_mapping: TokenMapping = collections.OrderedDict()
        word_bert_mapping: TokenMapping = collections.OrderedDict()

        for word_struct in word_structs:
            words.append(word_struct.text)
            word_phoneme_mapping[word_struct.text] = word_struct.phonemes

            tokens = self._tokenizer.tokenize(word_struct.text_with_punct)
            word_bert_mapping[word_struct.text] = tokens

        return TextFeatures(
            normalized_text=normalized_text,
            words=words,
            phonemes=phonemes,
            bert_tokens=bert_tokens,
            word_to_phoneme_spans=word_to_phoneme_spans,
            word_to_token_spans=word_to_token_spans)

    def _prepare_for_tokenization(self, text: str) -> str:
        """Cleans the text and prepares it for tokenization."""

        text = self.clean_text(text)

        def scream_case_replacer(match: re.Match) -> str:
            return match.group(0).capitalize()

        text = re.sub(r'\b[A-Z]{2,}\b', scream_case_replacer, text)

        for pattern, replacement in self.single_puncts_replace.items():
            text = text.replace(pattern, replacement)

        return text

    def _get_word_structs(self, text: str) -> List[_WordStruct]:
        """Converts text to a list of word structs."""

        for pattern, replacement in self.puncts_before_quotes_replace.items():
            text = text.replace(pattern, replacement)

        text = text.replace('"', '`')

        word_structs: List[_WordStruct] = []

        for sentence in gruut.sentences(text,
                                        lang='en-us',
                                        punctuations=True,
                                        phonemes=True):
            for word in sentence.words:
                if not word.phonemes:
                    continue

                if word.is_break and not word_structs:
                    continue

                if word.is_break:
                    word_structs[-1].text_with_punct += word.text
                    continue

                word_structs.append(_WordStruct(
                    text=self._get_proper_word_representation(word.text.lower()),
                    phonemes=list(word.phonemes),
                    text_with_punct=word.text.replace('`', '"')))

        self._post_process_word_structs(word_structs)

        return word_structs

    def _get_proper_word_representation(self, word: str) -> str:
        """Converts a word to its proper representation for tokenization."""

        return ''.join(filter(lambda x: x in TextProcessor.allowed_chars_for_word_repr, word))

    def _post_process_word_structs(self, word_structs: List[_WordStruct]):
        """Post-processes word structs to fix punctuation placement."""

        for word_struct in word_structs:

            text_with_punct = word_struct.text_with_punct

            for pattern, replacement in self.puncts_after_quotes_replace.items():
                text_with_punct = text_with_punct.replace(pattern, replacement)

            word_struct.text_with_punct = text_with_punct
