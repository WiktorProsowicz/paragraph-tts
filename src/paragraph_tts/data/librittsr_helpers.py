"""Contains utilities for accessing and transforming LibriTTS-R dataset."""

from typing import Dict, List, Tuple
import pathlib
import os
import csv
import dataclasses
import logging
import sys
import itertools

_THIS_MODULE_DIR = pathlib.Path(__file__).parent
_BOOKS_PATH = os.path.join(_THIS_MODULE_DIR, 'res', 'books.csv')
_SPEAKERS_PATH = os.path.join(_THIS_MODULE_DIR, 'res', 'speakers.tsv')
_READER_BOOK_MAPPING_PATH = os.path.join(_THIS_MODULE_DIR, 'res', 'reader_book.tsv')


def _logger():
    return logging.getLogger(__name__)


@dataclasses.dataclass
class BookInfo:
    """Contains information about a book in the LibriTTS-R dataset."""
    id: str
    title: str


@dataclasses.dataclass
class SpeakerInfo:
    """Contains information about a speaker in the LibriTTS-R dataset."""
    id: str
    gender: str
    name: str


class LibriTTSRMetadata:
    """Contains metadata of the LibriTTS-R dataset."""

    def __init__(self):
        """Initializes LibriTTS-R metadata."""

        books: Dict[str, BookInfo] = {}

        with open(_BOOKS_PATH, 'r', encoding='utf-8') as books_f:
            for line in csv.reader(books_f, delimiter='|'):
                line = [it.strip() for it in line]
                books[line[0]] = BookInfo(id=line[0], title=line[1])

        self._speakers: Dict[str, SpeakerInfo] = {}

        with open(_SPEAKERS_PATH, 'r', encoding='utf-8') as speakers_f:
            for line in itertools.dropwhile(lambda x: not x[0].isnumeric(),
                                            csv.reader(speakers_f, delimiter='\t')):
                line = [it.strip() for it in line]
                self._speakers[line[0]] = SpeakerInfo(id=line[0],
                                                      gender=line[1],
                                                      name=line[3])

        # Maps books to speaker, chapter pairs, as in the original mapping file.
        self._reader_book_mapping: Dict[Tuple[str, str], BookInfo] = {}

        with open(_READER_BOOK_MAPPING_PATH, 'r', encoding='utf-8') as mapping_f:
            for line in itertools.dropwhile(lambda x: not x[0].isnumeric(),
                                            csv.reader(mapping_f, delimiter='\t')):
                line = [it.strip() for it in line]
                spk_id, chapter_id, book_id = line
                self._reader_book_mapping[(spk_id, chapter_id)] = books[book_id]

    def get_speaker_and_book(self, spk_id: int, chapter_id: int) -> Tuple[SpeakerInfo, BookInfo]:
        """Returns speaker and book information for a given speaker and chapter ID"""

        if (str(spk_id), str(chapter_id)) not in self._reader_book_mapping:
            _logger().critical('Cannot find book for speaker ID %s and chapter ID %s',
                               spk_id, chapter_id)
            sys.exit(1)

        return (
            self._speakers[str(spk_id)],
            self._reader_book_mapping[(str(spk_id), str(chapter_id))]
        )


def _count_left_quotes(text: str) -> int:

    n_quotes = 0

    if text.startswith('"'):
        n_quotes += 1

    n_quotes += text.count(' "')

    return n_quotes


def _count_right_quotes(text: str) -> int:

    n_quotes = 0

    if text.endswith('"'):
        n_quotes += 1

    for pattern in ('".', '"!', '"?', '",', '";', '":', '" '):
        n_quotes += text.count(pattern)

    return n_quotes


def is_sentence_whole(text: str) -> bool:
    """Returns True if the text is a whole sentence.

    A whole sentence does not contain fragmented sub-sentences or dialogue parts.
    """

    ends_as_a_whole = any(text.strip().endswith(p) for p in ('.', '!', '?', '"', ':'))
    starts_as_a_whole = text[0].isupper() or text[0] == '"'
    has_even_quotes = _count_left_quotes(text) == _count_right_quotes(text)

    return ends_as_a_whole and starts_as_a_whole and has_even_quotes
