"""Multimodal processor for Voxtral (speech-to-text) models."""

import math
import re
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.voxtral import VoxtralForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_audio

# Special token IDs for Voxtral audio (from tekken.json vocabulary)
AUDIO_TOKEN_ID = 24  # [AUDIO]
BEGIN_AUDIO_TOKEN_ID = 25  # [BEGIN_AUDIO]
INST_TOKEN_ID = 3  # [INST]


class VoxtralMultimodalProcessor(BaseMultimodalProcessor):
    models = [VoxtralForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        audio_config = getattr(hf_config, "audio_config", None)
        self.audio_token_id = getattr(hf_config, "audio_token_id", AUDIO_TOKEN_ID)
        self.sampling_rate = getattr(audio_config, "sampling_rate", 16000)
        self.hop_length = getattr(audio_config, "hop_length", 160)
        self.max_source_positions = getattr(
            audio_config, "max_source_positions", 1500
        )
        self.conv_downsample = 2  # conv1 stride=1 * conv2 stride=2
        self.downsample_factor = getattr(
            audio_config,
            "downsample_factor",
            getattr(audio_config, "intermediate_size", 5120)
            // getattr(audio_config, "hidden_size", 1280),
        )

    def _compute_audio_token_count(self, n_samples: int) -> int:
        """Compute the number of [AUDIO] tokens for a given audio length."""
        mel_frames = n_samples / self.hop_length
        chunk_size = self.max_source_positions * self.conv_downsample
        n_chunks = math.ceil(mel_frames / chunk_size) if mel_frames > 0 else 1
        tokens_per_chunk = self.max_source_positions // self.downsample_factor
        return n_chunks * tokens_per_chunk

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not audio_data:
            return None

        # Load audio waveforms
        waveforms: List[torch.Tensor] = []
        for audio in audio_data:
            wav = load_audio(audio, self.sampling_rate)
            if not isinstance(wav, torch.Tensor):
                wav = torch.tensor(wav, dtype=torch.float32)
            if wav.dim() > 1:
                wav = wav.mean(dim=0)
            waveforms.append(wav)

        # Compute audio token counts and build input_ids with audio tokens
        audio_token_counts = [
            self._compute_audio_token_count(wav.shape[-1]) for wav in waveforms
        ]
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        input_ids = self._build_input_ids_with_audio(
            tokenizer, input_text, audio_token_counts
        )

        # Find offsets of [AUDIO] token runs and build mm_items
        audio_offsets = self._find_audio_offsets(input_ids, self.audio_token_id)
        mm_items = []
        for i, wav in enumerate(waveforms):
            item = MultimodalDataItem(feature=wav, modality=Modality.AUDIO)
            if i < len(audio_offsets):
                item.offsets = [audio_offsets[i]]
            mm_items.append(item)

        return {
            "input_ids": input_ids,
            "mm_items": mm_items,
            "audio_token_id": self.audio_token_id,
        }

    @staticmethod
    def _find_audio_offsets(input_ids: List[int], audio_token_id: int) -> List[tuple]:
        """Find consecutive runs of audio_token_id in input_ids."""
        offsets = []
        start = None
        for i, tok_id in enumerate(input_ids):
            if tok_id == audio_token_id:
                if start is None:
                    start = i
            elif start is not None:
                offsets.append((start, i - 1))
                start = None
        if start is not None:
            offsets.append((start, len(input_ids) - 1))
        return offsets

    def _build_input_ids_with_audio(
        self,
        tokenizer,
        input_text: str,
        audio_token_counts: List[int],
    ) -> List[int]:
        """Build input_ids by tokenizing text and inserting audio tokens.

        The input_text is a decoded Mistral prompt (from text-only
        apply_chat_template).  We re-tokenize to get proper special tokens
        (BOS, [INST], [/INST]), then insert [BEGIN_AUDIO] + [AUDIO]*N after
        the last [INST].
        """
        messages = self._parse_mistral_prompt(input_text)
        try:
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        except (ValueError, KeyError):
            # Fallback if prompt parsing produces malformed messages
            input_ids = tokenizer.encode(input_text)

        # Insert audio tokens after the last [INST]
        inst_positions = [
            i for i, t in enumerate(input_ids) if t == INST_TOKEN_ID
        ]
        insert_pos = (inst_positions[-1] + 1) if inst_positions else 1

        audio_tokens = []
        for count in audio_token_counts:
            audio_tokens.append(BEGIN_AUDIO_TOKEN_ID)
            audio_tokens.extend([AUDIO_TOKEN_ID] * count)

        return input_ids[:insert_pos] + audio_tokens + input_ids[insert_pos:]

    @staticmethod
    def _parse_mistral_prompt(prompt: str) -> List[Dict[str, str]]:
        """Parse a Mistral-formatted prompt into a list of messages."""
        messages = []
        text = prompt.strip()

        for marker in ["<s>", "</s>"]:
            text = text.replace(marker, "")
        text = text.strip()

        # Extract system prompt
        system_match = re.search(
            r"\[SYSTEM_PROMPT\]\s*(.*?)\s*\[/SYSTEM_PROMPT\]", text, re.DOTALL
        )
        if system_match:
            messages.append(
                {"role": "system", "content": system_match.group(1).strip()}
            )
            text = text[: system_match.start()] + text[system_match.end() :]
            text = text.strip()

        # Split by [INST] / [/INST]
        parts = re.split(r"\[/?INST\]", text)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 1:
                messages.append({"role": "user", "content": part})
            elif i > 0:
                messages.append({"role": "assistant", "content": part})

        if not messages:
            messages.append({"role": "user", "content": text})

        return messages
