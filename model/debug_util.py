import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


class SpecDebugRecorder:
    def __init__(self, debug_dir: Optional[str], tokenizer=None, root_dir: Optional[Path] = None):
        self.tokenizer = tokenizer
        self.debug_path = self._resolve_debug_dir(debug_dir, root_dir)
        self._steps = []
        self._run_index = 0
        self._active_run_file: Optional[Path] = None

    @staticmethod
    def _resolve_debug_dir(debug_dir: Optional[str], root_dir: Optional[Path] = None) -> Optional[Path]:
        debug_dir = debug_dir or os.environ.get("DFLASH_DEBUG_DIR")
        if not debug_dir:
            return None

        debug_path = Path(debug_dir)
        if not debug_path.is_absolute():
            if root_dir is None:
                root_dir = Path(__file__).resolve().parent.parent
            debug_path = root_dir / debug_path
        debug_path.mkdir(parents=True, exist_ok=True)
        return debug_path

    @property
    def enabled(self) -> bool:
        return self.debug_path is not None

    def _next_run_file(self) -> Optional[Path]:
        if not self.enabled:
            return None
        self._run_index += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return self.debug_path / f"spec_generate_{timestamp}_{self._run_index:04d}.jsonl"

    def _append_event(self, event: dict) -> None:
        if not self.enabled or self._active_run_file is None:
            return
        with self._active_run_file.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def token_ids_from_tensor(token_tensor: Optional[torch.Tensor]) -> list[int]:
        if token_tensor is None:
            return []
        return [int(token_id) for token_id in token_tensor.detach().reshape(-1).tolist()]

    def decode_token_ids(self, token_ids: list[int]) -> Optional[str]:
        if self.tokenizer is None:
            return None
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception:
            return None

    def decode_token_pieces(self, token_ids: list[int]) -> Optional[list[Optional[str]]]:
        if self.tokenizer is None:
            return None

        pieces = []
        for token_id in token_ids:
            piece = None
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                try:
                    piece = self.tokenizer.convert_ids_to_tokens(token_id)
                except Exception:
                    piece = None
            if piece is None:
                try:
                    piece = self.tokenizer.decode([token_id], skip_special_tokens=False)
                except Exception:
                    piece = None
            pieces.append(piece)
        return pieces

    def start_run(
        self,
        *,
        temperature: float,
        block_size: int,
        num_input_tokens: int,
        max_new_tokens: int,
        prompt_token_ids: list[int],
    ) -> Optional[str]:
        if not self.enabled:
            return None

        self._steps = []
        self._active_run_file = self._next_run_file()

        self._append_event(
            {
                "event": "meta",
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "temperature": float(temperature),
                "block_size": int(block_size),
                "num_input_tokens": int(num_input_tokens),
                "max_new_tokens": int(max_new_tokens),
            }
        )
        self._append_event(
            {
                "event": "prompt",
                "token_ids": prompt_token_ids,
                "text": self.decode_token_ids(prompt_token_ids),
            }
        )
        return str(self._active_run_file)

    def record_prefill(self, *, first_sampled_token_ids: list[int]) -> None:
        if not self.enabled:
            return
        self._append_event(
            {
                "event": "prefill",
                "first_sampled_token_ids": first_sampled_token_ids,
                "first_sampled_tokens": self.decode_token_pieces(first_sampled_token_ids),
                "first_sampled_text": self.decode_token_ids(first_sampled_token_ids),
            }
        )

    def add_step(
        self,
        *,
        step: int,
        start: int,
        block_size: int,
        context_token_ids: list[int],
        block_seed_token_ids: list[int],
        block_position_ids: list[int],
        draft_sampled_token_ids: list[int],
        posterior_token_ids: list[int],
        acceptance_length: int,
        accepted_token_ids: list[int],
        replacement_token_id: int,
    ) -> None:
        if not self.enabled:
            return

        step_payload = {
            "step": int(step),
            "start": int(start),
            "block_size": int(block_size),
            "context_length": len(context_token_ids),
            "context_token_ids": context_token_ids,
            "context_text": self.decode_token_ids(context_token_ids),
            "block_seed_token_ids": block_seed_token_ids,
            "block_seed_tokens": self.decode_token_pieces(block_seed_token_ids),
            "draft_position_ids": block_position_ids,
            "draft_predicted_token_ids": draft_sampled_token_ids,
            "draft_predicted_tokens": self.decode_token_pieces(draft_sampled_token_ids),
            "draft_predicted_text": self.decode_token_ids(draft_sampled_token_ids),
            "target_posterior_token_ids": posterior_token_ids,
            "target_posterior_tokens": self.decode_token_pieces(posterior_token_ids),
            "target_posterior_text": self.decode_token_ids(posterior_token_ids),
            "acceptance_length": int(acceptance_length),
            "accepted_token_ids": accepted_token_ids,
            "accepted_tokens": self.decode_token_pieces(accepted_token_ids),
            "accepted_text": self.decode_token_ids(accepted_token_ids),
            "replacement_token_id": int(replacement_token_id),
            "replacement_token": self.decode_token_pieces([replacement_token_id]),
            "replacement_text": self.decode_token_ids([replacement_token_id]),
        }

        self._steps.append(step_payload)
        self._append_event({"event": "step", **step_payload})

    def dump(
        self,
        *,
        temperature: float,
        block_size: int,
        num_input_tokens: int,
        max_new_tokens: int,
        prompt_token_ids: list[int],
        first_sampled_token_ids: list[int],
        final_output_token_ids: list[int],
    ) -> Optional[str]:
        if not self.enabled:
            return None

        if self._active_run_file is None:
            self.start_run(
                temperature=temperature,
                block_size=block_size,
                num_input_tokens=num_input_tokens,
                max_new_tokens=max_new_tokens,
                prompt_token_ids=prompt_token_ids,
            )
            self.record_prefill(first_sampled_token_ids=first_sampled_token_ids)

        self._append_event(
            {
                "event": "final_output",
                "token_ids": final_output_token_ids,
                "text": self.decode_token_ids(final_output_token_ids),
            }
        )

        debug_file = str(self._active_run_file) if self._active_run_file is not None else None
        self._active_run_file = None
        return debug_file
