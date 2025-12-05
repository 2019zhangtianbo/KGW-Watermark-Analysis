# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone implementation of the KGW watermarking algorithm.

This module provides a complete, self-contained implementation of the KGW
watermarking algorithm, including watermark generation and detection, and can be
directly integrated into any project that requires text watermarking.
"""

import os, tqdm, argparse
import torch
import nltk
import json
from math import sqrt
from typing import Union, List, Tuple, Dict, Optional
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, AutoModelForSeq2SeqLM, GenerationConfig, AutoConfig)
import time
import itertools
from transformers.utils import logging as hf_logging
from pathlib import Path
import pyarrow.parquet as pq
import ast

hf_logging.set_verbosity_error()
import sys
from datetime import datetime


# nltk.download('punkt_tab', quiet=True)  # Download NLTK sentence tokenizer model if needed


gen_config = {
    "max_new_tokens": 200,
    "min_new_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    # "num_beams": 1,
    # "length_penalty": 2,
    # "no_repeat_ngram_size": 3,
    # **generate_kwargs
}


kgw_config = {
    "gamma": 0.5,
    "delta": 2.0,
    "hash_key": 15485863,
    "z_threshold": 4.0,
    "prefix_length": 1,
    "f_scheme": "time",
    "window_scheme": "left",
}


llm_batchsize = {
        "t5-small": 600, "t5-base":100, "flan-t5-base":400, "flan-t5-small":800,
        "led-base-16384":40, "led-large-16384":40, "bart-large-cnn":80, "bart-large-xsum":55, 
        "pegasus-xsum":35
    }


class KGWConfig:
    """
    Configuration class for the KGW algorithm.

    Attributes:
        gamma: Proportion of the vocabulary placed in the green list.
        delta: Logit bias applied to green-list tokens.
        hash_key: Seed/key for the hash function.
        z_threshold: z-score threshold for watermark detection.
        prefix_length: Length of prefix used in the hash function.
        f_scheme: Hash function scheme.
        window_scheme: Windowing scheme used to build the green list.
        vocab_size: Vocabulary size.
        device: Compute device.
    """

    def __init__(self,
                 gamma: float = 0.5,
                 delta: float = 2.0,
                 hash_key: int = 15485863,
                 z_threshold: float = 4.0,
                 prefix_length: int = 1,
                 f_scheme: str = "time",
                 window_scheme: str = "left",
                 vocab_size: int = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 **kwargs):
        """
        Initialize KGW configuration.

        Args:
            gamma: Proportion of the vocabulary in the green list.
            delta: Logit bias applied to green-list tokens.
            hash_key: Seed/key for the hash function.
            z_threshold: z-score threshold for watermark detection.
            prefix_length: Length of prefix used in the hash function ("context length").
            f_scheme: Hash function scheme ("time", "additive", "skip", "min").
            window_scheme: Windowing scheme ("left", "self").
            vocab_size: Vocabulary size.
            device: Compute device.
            **kwargs: Additional parameters (ignored unless matched to attributes).
        """
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.z_threshold = z_threshold
        self.prefix_length = prefix_length
        self.f_scheme = f_scheme
        self.window_scheme = window_scheme
        self.vocab_size = vocab_size
        self.device = device

        # Validate parameters
        if self.f_scheme not in ["time", "additive", "skip", "min"]:
            raise ValueError("f_scheme must be one of ['time', 'additive', 'skip', 'min']")
        if self.window_scheme not in ["left", "self"]:
            raise ValueError("window_scheme must be one of ['left', 'self']")
        if self.gamma <= 0 or self.gamma >= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

    @classmethod
    def from_json_file(cls, config_path: str) -> 'KGWConfig':
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the JSON config file.

        Returns:
            KGWConfig: An initialized configuration object.
        """
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON config file: {config_path}")


class KGWUtils_old:
    """
    Utility class for the KGW algorithm.

    Provides helper functions needed by the algorithm, including
    green-list generation and z-score computation.
    """

    def __init__(self, config: KGWConfig) -> None:
        """
        Initialize KGW utility class.

        Args:
            config: KGW configuration instance.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)

        # Ensure vocab_size is set
        if self.config.vocab_size is None:
            raise ValueError("vocab_size must be set before initializing KGWUtils")

        self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng)
        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min
        }
        self.window_scheme_map = {
            "left": self._get_greenlist_ids_left,
            "self": self._get_greenlist_ids_self
        }

    def _f(self, input_ids: torch.LongTensor) -> int:
        """
        Compute a hash value based on the input token IDs.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            int: Hash value.
        """
        if len(input_ids) < self.config.prefix_length:
            raise ValueError(f"input_ids length must be >= prefix_length ({self.config.prefix_length})")
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))

    def _f_time(self, input_ids: torch.LongTensor) -> int:
        """Multiplicative hash scheme.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            int: Hash value.
        """
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        return self.prf[time_result % self.config.vocab_size]

    def _f_additive(self, input_ids: torch.LongTensor) -> int:
        """Additive hash scheme.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            int: Hash value.
        """
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        return self.prf[additive_result % self.config.vocab_size]

    def _f_skip(self, input_ids: torch.LongTensor) -> int:
        """Skip-based hash scheme.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            int: Hash value.
        """
        return self.prf[input_ids[-self.config.prefix_length].item()]

    def _f_min(self, input_ids: torch.LongTensor) -> int:
        """Minimum-value-based hash scheme.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            int: Hash value.
        """
        return min(self.prf[input_ids[-1 - i].item()] for i in range(0, self.config.prefix_length))

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> List[int]:
        """
        Get green-list token IDs.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            List[int]: List of green-list token IDs.
        """
        return self.window_scheme_map[self.config.window_scheme](input_ids)

    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> List[int]:
        """Get green-list token IDs using the left-window scheme.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            List[int]: List of green-list token IDs.
        """
        self.rng.manual_seed((self.config.hash_key * self._f(input_ids)) % self.config.vocab_size)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids.tolist()

    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> List[int]:
        """Get green-list token IDs using the self-hashing scheme.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            List[int]: List of green-list token IDs.
        """
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        greenlist_ids = []
        f_x = self._f(input_ids)

        # Self-hash scheme can be slow; this is an optimized version
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf[k])
            self.rng.manual_seed(h_k % self.config.vocab_size)
            vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)
        return greenlist_ids

    def _compute_z_score(self, observed_count: int, total_tokens: int) -> float:
        """
        Compute z-score for the number of green tokens.

        Args:
            observed_count: Observed count of green tokens.
            total_tokens: Number of scored tokens.

        Returns:
            float: z-score.
        """
        expected_count = self.config.gamma * total_tokens
        numerator = observed_count - expected_count
        denominator = sqrt(total_tokens * self.config.gamma * (1 - self.config.gamma))

        # Avoid division by zero
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def score_sequence(self, input_ids: torch.Tensor) -> Tuple[float, List[int]]:
        """
        Score an input sequence with the KGW watermark detector.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            Tuple[float, List[int]]: (z-score, green-token flags).
                green-token flags: -1 for prefix tokens, 1 for green, 0 for non-green.
        """
        if len(input_ids) < self.config.prefix_length:
            raise ValueError(
                f"Sequence length must be >= prefix_length ({self.config.prefix_length})")

        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                "Need at least 1 token after the prefix to compute a score")

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]  # -1 marks prefix tokens

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token.item() in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)  # 1 = green token
            else:
                green_token_flags.append(0)  # 0 = non-green token

        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class KGWUtils:
    """
    Utility class for the KGW algorithm (batched version).

    Provides helper functions needed by the algorithm, including
    green-list generation and z-score computation, with batched support.
    """

    def __init__(self, config: KGWConfig) -> None:
        """
        Initialize KGW utility class.

        Args:
            config: KGW configuration instance.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)

        # Ensure vocab_size is set
        if self.config.vocab_size is None:
            raise ValueError("vocab_size must be set before initializing KGWUtils")

        self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng)
        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min
        }
        self.window_scheme_map = {
            "left": self._get_greenlist_ids_left,
            "self": self._get_greenlist_ids_self
        }

    def _f(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Compute hash values based on batched input token IDs.

        Args:
            input_ids: Tensor of token IDs with shape [batch, seq_len].

        Returns:
            Tensor: Hash values for each batch element.
        """
        if input_ids.shape[1] < self.config.prefix_length:
            raise ValueError(f"input_ids length must be >= prefix_length ({self.config.prefix_length})")
        return self.f_scheme_map[self.config.f_scheme](input_ids)

    def _f_time(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Multiplicative hash scheme (batched).

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            Tensor: Hash values.
        """
        time_result = input_ids[..., -self.config.prefix_length:].prod(dim=-1)

        return self.prf[time_result % self.config.vocab_size]

    def _f_additive(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Additive hash scheme (batched).

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            Tensor: Hash values.
        """
        additive_result = input_ids[..., -self.config.prefix_length:].sum(dim=-1)
        return self.prf[additive_result % self.config.vocab_size]

    def _f_skip(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Skip-based hash scheme (batched).

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            Tensor: Hash values.
        """
        return self.prf[input_ids[..., -self.config.prefix_length]]

    def _f_min(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Minimum-value-based hash scheme (batched).

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            Tensor: Hash values.
        """
        return torch.min(self.prf[input_ids[..., -self.config.prefix_length:]], dim=-1).values

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Get green-list masks (batched).

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            Tensor: Green-list mask tensor of shape [batch, vocab_size],
                    where 1 marks green tokens and 0 marks others.
        """
        return self.window_scheme_map[self.config.window_scheme](input_ids)

    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> List[int]:
        """Get green-list masks using the left-window scheme (batched).

        Args:
            input_ids: Tensor of token IDs with shape [batch, seq_len].

        Returns:
            Tensor: Green-list mask tensor [batch, vocab_size].
        """
        # greenlist_all[b, v] = 1 if vocab index v is in the green list for batch element b
        greenlist_all = torch.zeros([input_ids.shape[0], self.config.vocab_size]).to(self.config.device)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        for i, seed in enumerate(self._f(input_ids)):
            self.rng.manual_seed((self.config.hash_key * seed.item()) % self.config.vocab_size)
            vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
            greenlist_ids = vocab_permutation[:greenlist_size]
            greenlist_all[i, greenlist_ids] = 1

        return greenlist_all

    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> List[int]:
        """Get green-list token IDs using the self-hashing scheme.

        NOTE: This variant is not batched and can be slow for large vocabularies.

        Args:
            input_ids: Tensor of token IDs.

        Returns:
            List[int]: List of green-list token IDs.
        """
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        greenlist_ids = []
        f_x = self._f(input_ids)

        # Self-hash scheme can be slow; this is an optimized version
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf[k])
            self.rng.manual_seed(h_k % self.config.vocab_size)
            vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)
        return greenlist_ids

    def _compute_z_score(self, observed_count: int, total_tokens: int) -> float:
        """
        Compute z-score for the number of green tokens.

        Args:
            observed_count: Observed count of green tokens.
            total_tokens: Number of scored tokens.

        Returns:
            float: z-score.
        """
        expected_count = self.config.gamma * total_tokens
        numerator = observed_count - expected_count
        denominator = sqrt(total_tokens * self.config.gamma * (1 - self.config.gamma))

        # Avoid division by zero
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def score_sequence(self, input_ids: torch.Tensor) -> Tuple[float, List[int]]:
        """
        Score an input sequence with the KGW watermark detector (non-batched API).

        Args:
            input_ids: Tensor of token IDs with shape [seq_len].

        Returns:
            Tuple[float, List[int]]: (z-score, green-token flags).
                green-token flags: -1 for prefix tokens, 1 for green, 0 for non-green.
        """
        if len(input_ids) < self.config.prefix_length:
            raise ValueError(
                f"Sequence length must be >= prefix_length ({self.config.prefix_length})")

        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                "Need at least 1 token after the prefix to compute a score")

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]  # -1 marks prefix tokens

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx].view(1, -1))
            greenlist = torch.where(greenlist_ids[0] == 1)[0]
            if curr_token.item() in greenlist:
                green_token_count += 1
                green_token_flags.append(1)  # 1 = green token
            else:
                green_token_flags.append(0)  # 0 = non-green token

        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class KGWLogitsProcessor_old(LogitsProcessor):
    """
    LogitsProcessor for the KGW algorithm (original implementation).

    Used to modify logits during generation to embed a watermark.
    """

    def __init__(self, config: KGWConfig, utils: KGWUtils) -> None:
        """
        Initialize the logits processor.

        Args:
            config: KGW configuration instance.
            utils: KGW utility instance.
        """
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: List[List[int]]) -> torch.BoolTensor:
        """
        Compute a green-list mask tensor.

        Args:
            scores: Model logits.
            greenlist_token_ids: List of green-list token IDs per batch element.

        Returns:
            torch.BoolTensor: Boolean mask indicating green tokens.
        """
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            for token_id in greenlist_token_ids[b_idx]:
                if token_id < self.config.vocab_size - 1:
                    green_tokens_mask[b_idx][token_id] = 1
        return green_tokens_mask.bool()

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float) -> torch.Tensor:
        """
        Add logit bias for green-list tokens.

        Args:
            scores: Model logits.
            greenlist_mask: Green-list boolean mask.
            greenlist_bias: Bias value added to green tokens.

        Returns:
            torch.Tensor: Modified logits.
        """
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        scores = scores[..., : self.config.vocab_size]
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Modify logits to embed the watermark.

        Args:
            input_ids: Input token IDs.
            scores: Model logits.

        Returns:
            torch.FloatTensor: Modified logits.
        """
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = []
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids.append(greenlist_ids)

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
        scores = self._bias_greenlist_logits(
            scores=scores,
            greenlist_mask=green_tokens_mask,
            greenlist_bias=self.config.delta
        )
        return scores


class KGWLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor for the KGW algorithm (batched, mask-based).

    Used to modify logits during generation to embed a watermark.
    """

    def __init__(self, config: KGWConfig, utils: KGWUtils) -> None:
        """
        Initialize the logits processor.

        Args:
            config: KGW configuration instance.
            utils: KGW utility instance.
        """
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: List[List[int]]) -> torch.BoolTensor:
        """
        (Legacy helper; kept for compatibility with the old implementation.)

        Args:
            scores: Model logits.
            greenlist_token_ids: List of green-list token IDs per batch element.

        Returns:
            torch.BoolTensor: Boolean mask indicating green tokens.
        """
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            for token_id in greenlist_token_ids[b_idx]:
                if token_id < self.config.vocab_size - 1:
                    green_tokens_mask[b_idx][token_id] = 1
        return green_tokens_mask.bool()

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float) -> torch.Tensor:
        """
        (Legacy helper; kept for compatibility with the old implementation.)

        Args:
            scores: Model logits.
            greenlist_mask: Green-list boolean mask.
            greenlist_bias: Bias value added to green tokens.

        Returns:
            torch.Tensor: Modified logits.
        """
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        scores = scores[..., : self.config.vocab_size]
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Modify logits to embed the watermark (mask-based interface).

        Args:
            input_ids: Input token IDs.
            scores: Model logits with shape [batch, vocab_size].

        Returns:
            torch.FloatTensor: Modified logits.
        """
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        green_tokens_mask = self.utils.get_greenlist_ids(input_ids)
        scores += green_tokens_mask * self.config.delta

        return scores


class ClampVocabLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, scores):
        # scores shape: [batch, vocab]
        scores[..., self.vocab_size:] = -float('inf')
        return scores


class KGWWatermark:
    """
    High-level wrapper for the KGW watermarking algorithm.

    Provides high-level APIs for watermark generation and detection.
    """

    def __init__(self,
                 tokenizer,
                 model,
                 config: Optional[KGWConfig] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 **kwargs):
        """
        Initialize the KGW watermark wrapper.

        Args:
            tokenizer: Tokenizer instance.
            model: Language model instance.
            config: KGW configuration. If None, a default config is created.
            device: Compute device.
            **kwargs: Additional configuration overrides.
        """
        # Attach tokenizer and model
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        # Initialize config
        if config is None:
            config = KGWConfig(vocab_size=self.model.config.vocab_size, device=device, **kgw_config)
        else:
            # Make sure vocab_size and device in the config match the model
            config.vocab_size = self.model.config.vocab_size
            config.device = device
            # Update other configuration attributes if provided in kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config
        self.utils = KGWUtils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str | list[str], **kwargs) -> str:
        return self._generate(prompt, use_watermark=True, **kwargs)

    def generate_unwatermarked_text(self, prompt: str | list[str], **kwargs) -> str:
        return self._generate(prompt, use_watermark=False, **kwargs)

    def _generate(self, prompt: str | list[str], use_watermark: bool, **kwargs) -> str:
        """
        Generate text with (or without) KGW watermarking.

        Args:
            prompt: Prompt string or list of prompt strings.
            use_watermark: If True, apply watermark; if False, generate normally.

        Returns:
            list[str]: List of generated texts.
        """
        try:
            # Prepare generation parameters
            clamp = ClampVocabLogitsProcessor(self.model.config.vocab_size - 1)

            gen_params = gen_config.copy()
            # gen_params.update(kwargs)
            processors = [self.logits_processor, clamp] if use_watermark else [clamp]
            gen_params["logits_processor"] = LogitsProcessorList(processors)

            # Encode prompts
            if self.tokenizer.model_max_length > 1e7:
                encoded_prompt = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=1024,
                    padding=True
                ).to(self.device)
            else:
                encoded_prompt = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    padding=True
                ).to(self.device)

            # Generate texts
            with torch.no_grad():
                encoded_generated_text = self.model.generate(**encoded_prompt, **gen_params)

            # Decode generated texts
            generated_text = self.tokenizer.batch_decode(
                encoded_generated_text,
                skip_special_tokens=True
            )

            return generated_text

        except Exception as e:
            raise RuntimeError(f"Error during generation: {str(e)}")

    def detect_watermark(self,
                         text: str,
                         return_dict: bool = True) -> Union[Tuple[bool, float], Dict[str, Union[bool, float]]]:
        """
        Detect a KGW watermark in text.

        Args:
            text: Text to test.
            return_dict: If True, return a dictionary; otherwise a tuple.

        Returns:
            Union[Tuple[bool, float], Dict[str, Union[bool, float]]]:
                If return_dict is True:
                    {"is_watermarked": bool, "score": float}
                Otherwise:
                    (is_watermarked, score)
        """
        try:
            # Encode text
            encoded_text = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0].to(self.device)

            # Compute z-score
            z_score, _ = self.utils.score_sequence(input_ids=encoded_text)

            # Decide whether the text is watermarked
            is_watermarked = z_score > self.config.z_threshold

            # Return according to the requested format
            if return_dict:
                return {
                    "is_watermarked": is_watermarked,
                    "score": z_score
                }
            else:
                return (is_watermarked, z_score)

        except Exception as e:
            raise RuntimeError(f"Error during watermark detection: {str(e)}")

    def visualize_watermark(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Visualize the KGW watermark pattern in text.

        Args:
            text: Input text.

        Returns:
            Tuple[List[str], List[int]]: (tokens, green-flags)
                green-flags:
                    -1 = prefix token,
                    0 = non-green token,
                    1 = green token.
        """
        try:
            # Encode text
            encoded_text = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0].to(self.device)

            # Compute z-score and per-token green flags
            _, highlight_values = self.utils.score_sequence(encoded_text)

            # Decode tokens one by one
            decoded_tokens = []
            for token_id in encoded_text:
                token = self.tokenizer.decode(token_id.item())
                decoded_tokens.append(token)

            return decoded_tokens, highlight_values

        except Exception as e:
            raise RuntimeError(f"Error during watermark visualization: {str(e)}")


class KGWHierarchicalSummarizer:
    """
    Hierarchical long-text summarizer based on facebook/bart-large-cnn.

    Strategy: Lead-k extraction → chunk remaining sentences → local summaries → global summary.
    """

    def __init__(
        self,
        model_name_or_path: str = 'test/models/bart-large-cnn',
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        overlap_sents: int = 2,
        lead_k_ratio: float = 0.4,
    ):
        """
        Initialization.

        Args:
            model_name_or_path: Path or name of the base summarization model.
            device: Compute device.
            overlap_sents: Number of overlapping sentences between chunks.
            lead_k_ratio: Ratio of leading sentences used as the "lead-k" extract.
        """
        self.device = device

        # Load model and tokenizer
        try:
            cfg = AutoConfig.from_pretrained(model_name_or_path)
            if cfg.architectures[0].endswith("ForConditionalGeneration"):
                model_cls = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            else:
                model_cls = AutoModelForCausalLM.from_pretrained(model_name_or_path)

            self.model = model_cls.from_pretrained(
                model_name_or_path,
                # device_map="auto",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.kgw = KGWWatermark(tokenizer=self.tokenizer, model=self.model, device=device)

        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")

        self.max_input_tok = min(self.tokenizer.model_max_length, 1e5) - 10
        self.chunk_tok = self.max_input_tok - 2  # Reserve space for <s> and </s>
        self.overlap_sents = overlap_sents
        self.lead_k_ratio = lead_k_ratio

    # ------------------ private helpers ------------------
    def _sent_split(self, text: str) -> List[str]:
        return nltk.sent_tokenize(text)

    def _tok_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _chunk_by_sentences(self, sents: List[str]) -> List[str]:
        chunks, cur, cur_len = [], [], 0
        for sent in sents:
            l = self._tok_len(sent)
            if cur_len + l > self.chunk_tok and cur:
                chunks.append(" ".join(cur))
                cur = cur[-self.overlap_sents :] if self.overlap_sents else []
                cur_len = sum(self._tok_len(s) for s in cur)
            cur.append(sent)
            cur_len += l
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    def _single_summarize(self, text: list[str], watermark: bool) -> str:
        text = ["Please summarize the following article: " + t for t in text]
        if watermark:
            summary_ids = self.kgw.generate_watermarked_text(text)
        else:
            summary_ids = self.kgw.generate_unwatermarked_text(text)
        return summary_ids

    # ------------------ public API ------------------
    def summarize(self, long_text: str, watermark: bool) -> str:
        """
        Summarize an arbitrarily long input text.

        Args:
            long_text: Full input document.
            watermark: If True, apply KGW watermark to the final summary.

        Returns:
            str: Final summary string.
        """
        # 1. Lead-k extraction
        sents = self._sent_split(long_text)
        if self.lead_k_ratio > 0:
            k = max(1, int(len(sents) * self.lead_k_ratio))
        else:
            k = 0
        extracted = " ".join(sents[:k])
        remain = " ".join(sents[k:])

        # 2. Chunk the remainder and summarize each chunk
        chunks = self._chunk_by_sentences(self._sent_split(remain))
        
        if len(chunks) > 1:
            chunk_summaries = self._single_summarize(chunks, watermark=False)
        else:
            chunk_summaries = chunks

        # 3. Concatenate lead-k and local chunk summaries
        concat_summary = extracted + " " + " ".join(chunk_summaries)

        # 4. If still too long, recurse
        if self._tok_len(concat_summary) > self.max_input_tok:
            concat_summary = self.summarize(concat_summary, watermark=False)  # recursive call

        # 5. Final global summary
        return self._single_summarize([concat_summary], watermark)[0]

    def detect_watermark(self,
                         text: str,
                         return_dict: bool = True) -> Union[Tuple[bool, float], Dict[str, Union[bool, float]]]:
        return self.kgw.detect_watermark(text, return_dict)
    

if __name__ == "__main__":
    # Read CLI arguments
    # Supported models: "led-base-16384", "led-large-16384", "pegasus-xsum"
    # Example (PowerShell):
    #   $env:CUDA_VISIBLE_DEVICES="2"; python kgw_hierarchical_summary.py -run_style 'batch' -max_lines 11334 -device 'cuda:0' -input_path 'dataset/xsum_test.parquet' -output_path 'outputs/xsum'
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=str, default='cuda:1')
    parser.add_argument("-llm_names", type=str, default="'t5-small', 't5-base','flan-t5-base', 'flan-t5-small','bart-large-cnn', 'bart-large-xsum'")
    parser.add_argument("-model_path", type=str, default='models/')
    parser.add_argument("-input_path", type=str, default='dataset/cnn_test.parquet')
    parser.add_argument("-output_path", type=str, default='outputs/cnn')
    parser.add_argument('-run_style', type=str, default='single')
    parser.add_argument('-max_lines', type=int, default=11490)  # Maximum number of lines to process
    
    paras = parser.parse_args()
    device = paras.device
    llm_names = ast.literal_eval(paras.llm_names)
    input_path = str((Path(__file__).parent / paras.input_path).resolve())
    output_path = str((Path(__file__).parent / paras.output_path).resolve())
    model_path = str((Path(__file__).parent / paras.model_path).resolve())
    run_style = paras.run_style
    max_lines = paras.max_lines
    
    print(f" device: {device}\n llm: {llm_names}\n model_path: {model_path}\n input_path: {input_path}\n out_path: {output_path}\n run_style: {run_style}\n max_lines: {max_lines}")
    
    os.makedirs(output_path, exist_ok=True)

    dataset = pq.ParquetDataset(input_path)
    df = dataset.read().to_pandas()
    
    # Process line by line
    for llm in llm_names:
        print(f"===== {llm} =====")

        dt_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        out_path = os.path.join(output_path, f"{llm}-{run_style}.jsonl")
        writer = open(out_path, 'w', encoding='utf-8')

        llm_path = (Path(__file__).parent / (paras.model_path + llm)).resolve()
        kgw_wm = KGWHierarchicalSummarizer(llm_path, overlap_sents=2, lead_k_ratio=0)

        if run_style == 'batch':
            batch_size = int(llm_batchsize[llm] * 0.8)
            for i in tqdm.tqdm(range(int(max_lines/batch_size) + 1), desc='processing'):
                start = i * batch_size
                end = min((i + 1) * batch_size, max_lines)
                articles = list(df.article[start: end])
                ids = list(df.id[start: end])
                
                res_wa = kgw_wm.kgw.generate_watermarked_text(articles)
                res_un = kgw_wm.kgw.generate_unwatermarked_text(articles)
                for j in range(len(articles)):
                    record = {"id": ids[j], "watermarked_text": res_wa[j], "unwatermarked_text": res_un[j]}
                    writer.write(json.dumps(record, ensure_ascii=False) + '\n')
                writer.flush()  # Flush to disk periodically for safety
        elif run_style == 'single':
            for i in tqdm.tqdm(range(max_lines)):
                res_wa = kgw_wm.summarize(df.article[i], watermark=True)
                res_un = kgw_wm.summarize(df.article[i], watermark=False)
                record = {"id": df.id[i], "watermarked_text": res_wa, "unwatermarked_text": res_un}
                writer.write(json.dumps(record, ensure_ascii=False) + '\n')
                if i % 2000 == 1:
                    writer.flush()  # Flush to disk periodically for safety

        writer.close()

   
    # # "pegasus-cnn", 
    # llm_names = [
    #     "t5-small", "t5-base", "flan-t5-base", "flan-t5-small", "led-base-16384", "led-large-16384", 
    #     "bart-large-cnn", "bart-large-xsum", "pegasus-xsum"
    # ]

    # {'t5-small': 94.5526182499998, 't5-base': 149.7127515000102, 'bart-large-cnn': 488.3891116667049, 
    #  'bart-large-xsum': 546.9955636363671, 'flan-t5-base': 152.6385723749945, 'flan-t5-small': 137.80852049999794, 
    #  'pegasus-xsum': 1705.3784962498867, 'led-base-16384': 244.5667620000313, 'led-large-16384': 377.62305000001106}
    # 3897.6654461780035


    # 5000 samples, with chunking considered
    # {'t5-small': 250.33671666642476, 't5-base': 295.33646666671603, 'bart-large-cnn': 258.52925833351037, 
    #  'bart-large-xsum': 290.3138916666042, 'flan-t5-base': 624.5172666664681, 'flan-t5-small': 233.02735833325036, 
    #  'pegasus-xsum': 909.0127500000259, 'led-base-16384': 321.463391666839, 'led-large-16384': 515.4308749997654}
    # 3697.967974999604
