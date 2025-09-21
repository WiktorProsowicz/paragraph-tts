# -*- coding: utf-8 -*-
"""Contains utilities for context enrichment."""
import dataclasses
import json
import logging
import random
from typing import Dict
from typing import List
from typing import Optional

import ollama

from paragraph_tts.data import librittsr_helpers


def _logger():
    return logging.getLogger(__name__)


@dataclasses.dataclass
class UtteranceForEnrichment:
    """Contains information about an utterance to be enriched."""

    text: str
    original_preceding_sentences: List[str]
    original_following_sentences: List[str]
    book: librittsr_helpers.BookInfo
    speaker: librittsr_helpers.SpeakerInfo


class ContextEnricher:
    """Generates additional context for text samples."""

    SYSTEM_MESSAGE = """You are an expert on literature and a creative writer. You can create
    engaging and contextually relevant paragraphs of text containing utterances with considerable
    diversity of length, style and content."""

    LLM_TEMPLATE = """Your task is to generate context sentences for a given sentence. The sentence
	comes from a book titled "{book_title}" read by a {speaker_gender} speaker. Pay attention
	to the given guidelines.

	### Given sentence
	{input_sentence}

	### Number of preceding sentences to generate
	{n_of_preceding_sentences}

	### Number of following sentences to generate
	{n_of_following_sentences}

	### Guidelines:
	1. The generated sentences should maintain coherence with the given sentence. That is, one
	should be able to read the generated sentences as if they were extracted from the original book.
	2. {sentences_length_guideline}
	3. The output should adhere to the following json format:
		It should be a single JSON object with two fields:
		- "preceding_sentences": a list of strings with the generated preceding sentences.
		- "following_sentences": a list of strings with the generated following sentences.
	4. DO NOT include any explanations or additional text outside of the JSON format.
	5. Ensure that the JSON format is strictly followed, with proper use of brackets, commas, etc.
    6. Return exactly as much sentences as requested, no more no less."""

    SHORT_SENTENCES_GUIDELINE = 'Create short sentences (5-10 words each).'

    LONG_SENTENCES_GUIDELINE = 'Create long sentences (15-20 words each).'

    def __init__(self,
                 model_name: str,
                 ollama_host: str,
                 max_paragraph_len: int,
                 min_paragraph_len: int,
                 should_retry: bool = True):
        """Initializes the ContextEnricher.

        Args:
            model_name: The name of the LLM model to use for context generation.
            ollama_host: The host address of the Ollama server.
            max_paragraph_len: The maximum number of context sentences to generate.
            min_paragraph_len: The minimum number of context sentences to generate.
            should_retry: Whether to retry generation if the LLM response is invalid.
        """

        self._max_paragraph_len = max_paragraph_len
        self._min_paragraph_len = min_paragraph_len

        self._model_name = model_name
        self._ollama_host = ollama_host
        self._should_retry = should_retry

        self._ollama_client = ollama.Client(host=ollama_host)

    def generate_context_for_utt(self,
                                 utt: UtteranceForEnrichment
                                 ) -> Optional[Dict[str, List[str]]]:
        """Generates additional context for a given utterance.

        Returns:
            A dictionary with two keys: 'preceding_sentences' and 'following_sentences', each
            containing a list of generated sentences. Returns None if generation fails.
        """

        context_len = random.randint(
            self._min_paragraph_len - 1, self._max_paragraph_len - 1)

        if utt.original_following_sentences or utt.original_preceding_sentences:

            if not utt.original_following_sentences:
                n_of_preceding_sentences = context_len
                n_of_following_sentences = 0

            else:
                n_of_preceding_sentences = 0
                n_of_following_sentences = context_len

        else:

            n_of_preceding_sentences = random.randint(1, context_len - 1)
            n_of_following_sentences = context_len - n_of_preceding_sentences

        gender = 'male' if utt.speaker.gender == 'M' else 'female'

        length_guideline = random.choice(
            [self.SHORT_SENTENCES_GUIDELINE, self.LONG_SENTENCES_GUIDELINE])

        user_prompt = self.LLM_TEMPLATE.format(
            book_title=utt.book.title,
            speaker_gender=gender,
            input_sentence=utt.text,
            n_of_preceding_sentences=n_of_preceding_sentences,
            n_of_following_sentences=n_of_following_sentences,
            sentences_length_guideline=length_guideline
        )

        _logger().debug('Generating context (%d preceding, %d following) for: %s',
                        n_of_preceding_sentences, n_of_following_sentences, utt)

        contexts = self._generate_contexts(
            prompt=user_prompt,
            n_of_preceding_sentences=n_of_preceding_sentences,
            n_of_following_sentences=n_of_following_sentences
        )

        if contexts is None and self._should_retry:
            _logger().debug('Retrying context generation for utterance: %s', utt)

            return self._generate_contexts(
                prompt=user_prompt,
                n_of_preceding_sentences=n_of_preceding_sentences,
                n_of_following_sentences=n_of_following_sentences
            )

        return contexts

    def is_model_available(self, model_name: str) -> bool:
        """Checks if a given model is available on the Ollama server."""

        available_models = self._ollama_client.list().models  # pylint: disable=no-member
        return any(model.model == model_name for model in available_models)

    def _generate_contexts(self,
                           prompt: str,
                           n_of_preceding_sentences: int,
                           n_of_following_sentences: int
                           ) -> Optional[Dict[str, List[str]]]:
        """Generates contexts using the LLM and retrieves them from the response."""

        response = self._ollama_client.chat(self._model_name,
                                            messages=[
                                                {
                                                    'role': 'system',
                                                    'content': self.SYSTEM_MESSAGE
                                                },
                                                {
                                                    'role': 'user',
                                                    'content': prompt
                                                }
                                            ])

        _logger().debug('Received response: %s', response)

        return self._retrieve_contexts_from_response(response,
                                                     n_of_preceding_sentences,
                                                     n_of_following_sentences)

    def _retrieve_contexts_from_response(self,
                                         response: ollama.ChatResponse,
                                         n_preceding_sentences: int,
                                         n_following_sentences: int
                                         ) -> Optional[Dict[str, List[str]]]:
        """Validates the LLM's response and retrieves the contexts."""

        message_content = response.message['content']

        try:
            response_json = json.loads(message_content)

        except json.JSONDecodeError as e:
            _logger().debug('Failed to parse LLM response as JSON: %s', e)
            return None

        for required_key in ('preceding_sentences', 'following_sentences'):
            if required_key not in response_json:
                _logger().debug('LLM response is missing required key: %s', required_key)
                return None

            if not isinstance(response_json[required_key], list) or not all(
                    isinstance(s, str) for s in response_json[required_key]):
                _logger().debug('LLM response has invalid format for key: %s', required_key)
                return None

        return {
            'preceding_sentences': response_json['preceding_sentences'][:n_preceding_sentences],
            'following_sentences': response_json['following_sentences'][:n_following_sentences]
        }
