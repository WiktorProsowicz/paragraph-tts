"""Contains audio processing utilities."""

from typing import List, Tuple, Dict, Any
import dataclasses

import gruut
from DeBERTa import deberta


@dataclasses.dataclass
class TextFeatures:
    """Contains features extracted from text."""

    normalized_text: str
    words: List[str]
    phonemes: List[str]
    bert_tokens: List[str]
    word_to_phoneme_spans: List[int]  # Lengths of words in phonemes.
    word_to_token_spans: List[int]  # Lengths of words in BERT tokens.


@dataclasses.dataclass
class _WordStruct:
    """Contains word-level information during text processing."""

    text: str
    phonemes: List[str]
    text_with_punct: str


class TextProcessor:
    """Processes text data."""

    def __init__(self):
        """Inits the text processor."""

        vocab_path, vocab_type = deberta.load_vocab(pretrained_id='xxlarge-v2')
        self._tokenizer = deberta.tokenizers[vocab_type](vocab_path)
        self._allowed_chars = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            " .,!?"
        )

    def normalize_text(self, text: str) -> str:
        """Performs text normalization."""

        text = filter(lambda x: x in self._allowed_chars, text)
        text = "".join(text)

        return " ".join(text.split())

    def tokenize_text(self, normalized_text: str) -> TextFeatures:
        """Processes and tokenizes text."""

        word_structs = self._get_word_structs(normalized_text)

        word_to_phoneme_spans = []
        word_to_token_spans = []
        bert_tokens = []
        phonemes = []
        words = []

        for word_struct in word_structs:
            words.append(word_struct.text)
            phonemes.extend(word_struct.phonemes)
            word_to_phoneme_spans.append(len(word_struct.phonemes))

            tokens = self._tokenizer.tokenize(word_struct.text_with_punct)
            bert_tokens.extend(tokens)
            word_to_token_spans.append(len(tokens))

        return TextFeatures(
            normalized_text=normalized_text,
            words=words,
            phonemes=phonemes,
            bert_tokens=bert_tokens,
            word_to_phoneme_spans=word_to_phoneme_spans,
            word_to_token_spans=word_to_token_spans)

    def _get_word_structs(self, text) -> List[_WordStruct]:
        """Converts text to a list of word structs."""

        word_structs = []

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
                    word_structs[-1].phonemes.append(word.text)
                    continue

                word_structs.append(_WordStruct(
                    text=word.text,
                    phonemes=word.phonemes,
                    text_with_punct=word.text))

        return word_structs
