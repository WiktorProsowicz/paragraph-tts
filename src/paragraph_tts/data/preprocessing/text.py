"""Contains audio processing utilities."""

from typing import List, Tuple, Dict, Any
import dataclasses

import gruut
from DeBERTa import deberta


def _logger():
    return logging.getLogger(__name__)


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
    }

    allowed_chars = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        " .,:!?'\"$€"
    )

    def __init__(self):
        """Inits the text processor."""

        vocab_path, vocab_type = deberta.load_vocab(pretrained_id='xxlarge-v2')
        self._tokenizer = deberta.tokenizers[vocab_type](vocab_path)

    def clean_text(self, text: str) -> str:
        """Cleans the text by removing unwanted characters."""

        text = filter(lambda x: x in self.allowed_chars, text)
        text = "".join(text)
        text = " ".join(text.split())

        return text
    
    @staticmethod
    def load_text(text_path: str) -> str:
        """Loads text from a file."""
        
        with open(text_path, 'r', encoding='utf-8') as text_f:
            return text_f.read().strip()

    def tokenize_text(self, text: str) -> TextFeatures:
        """Processes and tokenizes text."""

        normalized_text = self._prepare_for_tokenization(text)

        word_structs = self._get_word_structs(normalized_text.replace('"', '`'))
        self._post_process_word_structs(word_structs)

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

    

    def _prepare_for_tokenization(self, text: str) -> str:
        """Cleans the text and prepares it for tokenization."""

        text = self.clean_text(text)

        for pattern, replacement in self.single_puncts_replace.items():
            text = text.replace(pattern, replacement)

        for pattern, replacement in self.puncts_before_quotes_replace.items():
            text = text.replace(pattern, replacement)

        return text

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

                word_phonemes = list(word.phonemes)

                if word.text.startswith('`'):
                    word_phonemes = ['"'] + word_phonemes

                if word.text.endswith('`'):
                    word_phonemes = word_phonemes + ['"']

                word_structs.append(_WordStruct(
                    text=word.text.replace('`', ''),
                    phonemes=word_phonemes,
                    text_with_punct=word.text.replace('`', '"')))

        return word_structs

    def _post_process_word_structs(self, word_structs: List[_WordStruct]):
        """Post-processes word structs to fix punctuation placement."""

        for word_struct in word_structs:

            word_phonemes = ' '.join(word_struct.phonemes)
            word_phonemes = word_phonemes.replace('" !', '! "').replace('" ?', '? "')
            word_phonemes = word_phonemes.split(' ')

            text_with_punct = word_struct.text_with_punct

            for pattern, replacement in self.puncts_after_quotes_replace.items():
                text_with_punct = text_with_punct.replace(pattern, replacement)

            word_struct.phonemes = word_phonemes
            word_struct.text_with_punct = text_with_punct
