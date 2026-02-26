import copy
import gc
import logging
import os
import sys
from importlib.metadata import version
from importlib.util import find_spec
from multiprocessing import Process, Queue
from queue import Empty
from time import sleep
from typing import Any, TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import jinja2
import torch
from more_itertools import distribute
from packaging.version import parse as parse_version
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    configure_pad_token,
    handle_stop_sequences,
    postprocess_generated_text,
    undistribute,
)
from lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
)


try:
    import ray
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config.model import ModelConfig
    from vllm.config.utils import config
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer, init_tokenizer_from_configs
    try:
        from vllm.utils.network_utils import get_open_port  # type: ignore
    except (ModuleNotFoundError, ImportError):
        from vllm.utils import get_open_port
    from vllm.v1.sample.logits_processor.interface import (
        BatchUpdate,
        LogitsProcessor,
        MoveDirectionality,
    )

    if parse_version(version("vllm")) >= parse_version("0.8.3"):
        from vllm.entrypoints.chat_utils import resolve_hf_chat_template
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass

eval_logger = logging.getLogger(__name__)
# Candidate Continuation Behavior:
#   1. Sample Next Most Likely Token
#   2. Sample: "Wait" Token
class ThinkingTokenBudgetLogitsProcessor(LogitsProcessor):
    """Limits the number of tokens allowed inside a 'thinking' section."""

    def __init__(
        self, 
        vllm_config: "VllmConfig", 
        device="cuda:0", 
        is_pin_memory=True,
    ):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        tokenizer = get_tokenizer(
            vllm_config.model_config.tokenizer if vllm_config.model_config.tokenizer else vllm_config.model_config.model,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            revision=vllm_config.model_config.tokenizer_revision,
            # add_bos_token=vllm_config.add_bos_token,
        )

        self.think_start_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<think>"))
        self.think_continuation_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Let's verify this solution"))
        self.think_end_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("</think>")) 
        self.think_termination_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
            # self.tokenizer.tokenize("</think>\n\n The answer is:")
            # TODO: This could be intercepted with prefill rather than decoding, but we'll do drop in replacement for now for performance scaling.
            #   Output predictions should be the same asssuming determinism. This termination sequence is from the Qwen3 documentation.
            "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
        ))

        self.is_enabled = True
        self.pin_memory = is_pin_memory
        self.device = device
        # Per-request state tracking for thinking token management
        # Key: request_index, Value: state dict containing:
        # "in_think": bool - currently in thinking mode
        # "in_end": bool - currently forcing end tokens output
        # "check_count_down": int - steps remaining until next think
        #                            start/end token parsing
        # "think_count": int - number of thinking tokens generated
        # "end_count": int - number of end tokens forced so far
        # "thinking_token_budget": int - max allowed thinking tokens
        # "output_tok_ids": list[int] - generated output tokens
        # "prev_output_length": int - previous output length for
        #                               incremental processing
        self._state: dict[int, dict[str, Any]] = {}

        # Preallocate reusable tensors
        self.mask = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.force_token_ids = torch.full(
            (max_num_reqs,), -1, dtype=torch.long, device=device
        )

    @staticmethod
    def _find_last_sequence_index(target_list: list[int], token_ids: list[int]) -> int:
        """
        Returns the index of the last occurrence of token_ids in target_list.
        Args:
          target_list (list[int]): The list of token IDs.
          token_ids (list[int]): The sequence of token IDs to find.
        """
        if not token_ids:
            return -1
        for i in range(len(target_list) - len(token_ids), -1, -1):
            if target_list[i : i + len(token_ids)] == token_ids:
                return i
        return -1

    def _init_state_entry(
        self, prompt_tok_ids: Optional[list[int]], thinking_token_budget_max: int, thinking_token_budget_min: int, answer_prefix_ids:[list[int]], continuation_mode: str
    ) -> dict[str, Any]:
        """Initializes the tracking state for a given sequence index."""
        if answer_prefix_ids is not None:
            self.think_termination_token_ids.extend(answer_prefix_ids)

        if prompt_tok_ids is None:
            last_start = -1
            last_end = -1
            in_think = False
            think_count = 0
        else:
            last_start = self._find_last_sequence_index(
                prompt_tok_ids, self.think_start_token_ids
            )
            last_end = self._find_last_sequence_index(
                prompt_tok_ids, self.think_end_token_ids
            )
            in_think = last_start > last_end
            if in_think:
                think_count = len(prompt_tok_ids) - (
                    last_start + len(self.think_start_token_ids)
                )
            else:
                think_count = 0
        return {
            "in_think": in_think,  # Currently in thinking mode
            "in_end": in_think and thinking_token_budget_max == 0,
            "in_continuation": False,
            "check_count_down": 1, # thinking_token_budget_max,
            "think_count": think_count,  # Number of tokens in thinking section
            "end_count": 0,  # Number of end tokens forced so far
            "continuation_count": 0,
            "continuation_mode": continuation_mode,
            "prompt_tok_ids": prompt_tok_ids,
            "output_tok_ids": [],
            "thinking_token_budget_max": thinking_token_budget_max,
            "thinking_token_budget_min": float(thinking_token_budget_min),
            "prev_output_length": 0,
            # Track previous output length for incremental updates
        }

    def _update_think_state(self, state: dict[str, Any]):
        """Updates the state based on newly generated output tokens."""
        if not state.get("in_end", False) and state.get("check_count_down", 0) > 0:
            state["check_count_down"] -= 1
            return

        output = state.get("output_tok_ids", [])
        if not output:
            return

        # Track previous output length for incremental processing
        prev_length = state.get("prev_output_length", 0)
        current_length = len(output)

        if current_length <= prev_length:
            return

        # Process only newly added tokens
        new_tokens = output[prev_length:]
        state["prev_output_length"] = current_length

        # Check if new tokens contain think start or end sequences
        start_len = len(self.think_start_token_ids)
        end_len = len(self.think_end_token_ids)

        # Look for think sequences in recent tokens (including boundary)
        # Check overlapping regions where sequences might span boundaries
        check_start_idx = max(0, prev_length - max(start_len, end_len) + 1)
        recent_tokens = output[check_start_idx:]

        # Find any think start/end sequences in recent tokens
        recent_start_pos = self._find_last_sequence_index(
            recent_tokens, self.think_start_token_ids
        )
        recent_end_pos = self._find_last_sequence_index(
            recent_tokens, self.think_end_token_ids
        )
        # TODO(Jared): Suppress ANY EoS Token until; including eot besides </think>
        # recent_end_seq_pos = self._find_last_sequence_index(
        #     recent_tokens
        # )

        # Update state based on recent sequences
        if not state["in_end"]:
            if recent_start_pos >= 0 and recent_end_pos >= 0:
                if recent_start_pos > recent_end_pos:
                    # Case: ...<end>...<start>... - entering think mode
                    absolute_start_pos = check_start_idx + recent_start_pos
                    new_think_count = current_length - (absolute_start_pos + start_len)
                    state["in_think"] = True
                    state["think_count"] = new_think_count
                else:
                    # Case: ...<start>...<end>... - exiting think mode
                    if state["think_count"] >= state["thinking_token_budget_min"]:
                        state["in_think"] = False
                        state["think_count"] = 0

            elif recent_start_pos >= 0:
                # Found think start - entering think mode
                absolute_start_pos = check_start_idx + recent_start_pos
                new_think_count = current_length - (absolute_start_pos + start_len)
                state["in_think"] = True
                state["think_count"] = new_think_count
            elif recent_end_pos >= 0 and state["think_count"]:
                # Found think end - exiting think mode
                state["in_think"] = False
                state["think_count"] = 0
            elif state["in_think"]:
                # Continue thinking mode, increment count by new tokens
                state["think_count"] += len(new_tokens)

            # Set countdown based on current state
            if state["in_think"]:
                remaining_budget = max(
                    0, state["thinking_token_budget_max"] - state["think_count"], 
                    )
                state["check_count_down"] = remaining_budget
            else:
                state["check_count_down"] = state["thinking_token_budget_max"]

            # Check if need to transition to end mode
            if (
                state["in_think"]
                and state["think_count"] >= state["thinking_token_budget_max"]
                and state["think_count"] >= state["thinking_token_budget_min"]
            ):
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["check_count_down"] = state["thinking_token_budget_max"]
        else:
            # In end mode
            state["end_count"] += 1
            if state["end_count"] >= len(self.think_termination_token_ids):
                state.update(
                    {
                        "in_end": False,
                        "end_count": 0,
                        "check_count_down": state["thinking_token_budget_max"],
                    }
                )

    def is_argmax_invariant(self) -> bool:
        """This logits processor can change the outcome of
        greedy sampling by forcing that the thinking section
        ends after a certain number of tokens."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not self.is_enabled:
            return
        if batch_update:
            for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
                thinking_token_budget_max = params.extra_args.get('thinking_token_budget_max', None)
                thinking_token_budget_min = params.extra_args.get('thinking_token_budget_min', None)
                answer_prefix_ids = params.extra_args.get('answer_prefix_ids', None)
                continuation_mode = params.extra_args.get("continuation_mode", "wait")

                if thinking_token_budget_max is not None or thinking_token_budget_min is not None:
                    self._state[index] = self._init_state_entry(
                        prompt_tok_ids, thinking_token_budget_max, thinking_token_budget_min, answer_prefix_ids, continuation_mode
                    )
                    self._state[index]["output_tok_ids"] = output_tok_ids
                else:
                    # Remove state if no thinking budget
                    self._state.pop(index, None)

            for index in batch_update.removed:
                self._state.pop(index, {})

            for i1, i2, direction in batch_update.moved:
                if direction == MoveDirectionality.SWAP:
                    state1 = self._state.get(i1, {})
                    state2 = self._state.get(i2, {})
                    if state1 or state2:
                        self._state[i1] = state2
                        self._state[i2] = state1
                else:
                    self._state[i2] = self._state.pop(i1, {})

        for state in self._state.values():
            self._update_think_state(state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.is_enabled or not self._state:
            return logits

        batch_size = logits.size(0)
        self.mask[:batch_size] = False

        for i in range(batch_size):
            row_state = self._state.get(i)
            if row_state and row_state["in_end"]:
                self.mask[i] = True
                termination_token_ids = row_state["termination_token_ids"]
                self.force_token_ids[i] = termination_token_ids[row_state['end_count']]
                # Advance end_count here so apply() owns the full in_end lifecycle,
                # avoiding the off-by-one where _update_think_state increments before
                # apply() reads end_count.
                if row_state['end_count'] == 0:
                    print(
                        f"[ThinkingBudget] Request index={i}: forcing termination sequence "
                        f"({len(termination_token_ids)} tokens), "
                        f"think_count={row_state['think_count']}, "
                        f"budget=[{row_state['thinking_token_budget_min']}, {row_state['thinking_token_budget_max']}]",
                        flush=True, file=sys.stderr
                    )
                row_state['end_count'] += 1
                if row_state['end_count'] >= len(termination_token_ids):
                    row_state['in_end'] = False
                    row_state['end_count'] = 0
                    row_state['in_think'] = False
                    row_state['think_count'] = 0
                    row_state['check_count_down'] = row_state['thinking_token_budget_max']
                    print(
                        f"[ThinkingBudget] Request index={i}: termination sequence complete, "
                        f"output_tok_ids length={len(row_state['output_tok_ids'])}",
                        flush=True, file=sys.stderr
                    )

            # Either begin continuation sequence if </think> is present; or if already in continuation
            if row_state and (
                (row_state["in_continuation"] == True and row_state['continuation_count'] < len(self.think_continuation_token_ids)
            ) or (
                torch.argmax(logits) == self.think_end_token_ids[0]
                and len(row_state['output_tok_ids']) < row_state['thinking_token_budget_min']
            )):
                row_state["in_continuation"] = True
                self.mask[i] = True
                if (
                    row_state["continuation_mode"] == "wait"
                    and row_state['continuation_count'] < len(self.think_continuation_token_ids)
                ):
                    self.force_token_ids[i] = self.think_continuation_token_ids[row_state['continuation_count']]

            # Suppress </think> if under the required budget
            if row_state and len(row_state['output_tok_ids']) < row_state['thinking_token_budget_min']:
                logits[i, self.think_end_token_ids[0]] = -1e9

        # Check in CPU first not to sync with GPU
        has_active_thinking = any(
            state.get("in_end", False) for state in self._state.values()
        )
        has_active_continuation = any(
            state.get("in_continuation", False) for state in self._state.values()
        )

        if has_active_thinking or has_active_continuation:
            current_mask = self.mask[:batch_size]
            active_indices = current_mask.nonzero(as_tuple=False).view(-1)
            if len(active_indices) > 0:
                force_tokens = self.force_token_ids[active_indices]
                logits[active_indices, force_tokens] = 1e9

        # Increment the tracker on the continuation sequence or reset if done
        for i in range(batch_size):
            row_state = self._state.get(i)
            if row_state is None:
                continue
            if row_state["continuation_mode"] == "suppress_end_think":
                row_state['in_continuation'] = False
                row_state['continuation_count'] = 0
            elif (
                row_state["continuation_mode"] == "wait" and row_state['in_continuation']
                and row_state['continuation_count'] < len(self.think_continuation_token_ids)
            ):
                row_state['continuation_count'] += 1
            elif (
                row_state["continuation_mode"] == "wait"
                and row_state['continuation_count'] >= len(self.think_continuation_token_ids)
            ):
                row_state.update(
                    {
                        "in_continuation": False,
                        "continuation_count": 0,
                        "check_count_down": row_state["thinking_token_budget_max"] - row_state['think_count'],
                    }
                )

        return logits


def _find_sequence_index(target_list: list, token_ids: list) -> int:
    """Returns the index of the last occurrence of token_ids in target_list, or -1."""
    if not token_ids:
        return -1
    for i in range(len(target_list) - len(token_ids), -1, -1):
        if target_list[i : i + len(token_ids)] == token_ids:
            return i
    return -1


class ConstrainedChoiceLogitsProcessor(LogitsProcessor):
    """
    Constrains generation to a fixed set of valid choice strings.

    Activated per-request when 'constrained_choices' is present in SamplingParams.extra_args.

    For thinking models (enable_thinking_for_constrained=True), constraints only activate
    after '</think>' is detected in the output tokens. For non-thinking models, constraints
    apply from the first generated token.
    """

    class TrieNode:
        __slots__ = ("children", "is_terminal", "_valid_token_tensor")

        def __init__(self):
            self.children: dict = {}
            self.is_terminal: bool = False
            self._valid_token_tensor = None  # torch.Tensor, built after trie construction

    def __init__(self, vllm_config: "VllmConfig", device="cuda:0", is_pin_memory=True):
        self.device = device
        self.pin_memory = is_pin_memory

        tokenizer = get_tokenizer(
            vllm_config.model_config.tokenizer if vllm_config.model_config.tokenizer else vllm_config.model_config.model,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            revision=vllm_config.model_config.tokenizer_revision,
        )
        self._tokenizer = tokenizer

        # Pre-tokenize </think> for thinking detection
        self.think_end_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("</think>"))

        # Pre-compute EOS token IDs for use at terminal trie nodes
        eos_ids = []
        for tok in [tokenizer.eos_token_id]:
            if tok is not None:
                eos_ids.append(tok)
        for tok_str in ["<|im_end|>", "<|eot_id|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
                eos_ids.append(tid)
        self.eos_token_ids = eos_ids
        self.eos_token_ids_tensor = torch.tensor(eos_ids, dtype=torch.long, device=device) if eos_ids else None

        self._trie_cache: dict = {}

        self._state: dict = {}

    def _build_trie(self, choice_strings: list) -> "ConstrainedChoiceLogitsProcessor.TrieNode":
        """Build a token-level trie from a list of choice strings. Populates cached tensors on each node."""
        root = self.TrieNode()

        for choice_str in choice_strings:
            token_ids = self._tokenizer.encode(choice_str, add_special_tokens=False)
            assert -1 not in token_ids and self._tokenizer.unk_token_id not in token_ids, \
                f"ConstrainedChoiceLogitsProcessor: choice {repr(choice_str)} produced unknown token id in {token_ids}"
            node = root
            for tid in token_ids:
                if tid not in node.children:
                    node.children[tid] = self.TrieNode()
                node = node.children[tid]
            node.is_terminal = True

        # Populate cached valid-token tensors on every node (BFS)
        from collections import deque
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node.children:
                node._valid_token_tensor = torch.tensor(
                    list(node.children.keys()), dtype=torch.long, device=self.device
                )
                queue.extend(node.children.values())

        sample_encodings = [
            (s, self._tokenizer.encode(s, add_special_tokens=False))
            for s in choice_strings[:3]
        ]
        print(
            f"[ConstrainedDecoding] Trie built: {len(choice_strings)} choices, "
            f"root has {len(root.children)} distinct first-token(s). "
            f"Sample encodings: {[(s, ids) for s, ids in sample_encodings]}",
            flush=True, file=sys.stderr
        )
        return root

    def _scan_for_think_end(self, state: dict):
        """
        For thinking models only: scan output_tok_ids for </think> and activate
        constrained decoding when found. Called from update_state() each step.

        Trie advancement is handled in apply() directly via next_read_idx,
        because output_tok_ids has a vLLM write-ahead sentinel (-1) at the last
        position that makes it unreliable for tracking committed tokens.
        """
        output = state["output_tok_ids"]
        current_length = len(output)
        prev = state["prev_output_length"]

        if current_length <= prev:
            return

        state["prev_output_length"] = current_length
        end_len = len(self.think_end_token_ids)
        check_start = max(0, prev - end_len + 1)
        # Exclude the trailing sentinel when scanning
        recent = [t for t in output[check_start:] if t >= 0]

        if _find_sequence_index(recent, self.think_end_token_ids) >= 0:
            state["constrained_active"] = True
            state["current_node"] = state["trie_root"]
            state["next_read_idx"] = len(output)  # start consuming from here, after </think>
            print(
                f"[ConstrainedDecoding] </think> detected at output token {current_length}. "
                f"Constrained decoding now active.",
                flush=True, file=sys.stderr
            )

    def is_argmax_invariant(self) -> bool:
        """Returns False: this processor changes greedy sampling by constraining valid tokens."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update is None:
            # Scan for </think> on thinking models; trie advancement happens in apply()
            for state in self._state.values():
                if state.get("active") and state.get("enable_thinking") and not state.get("constrained_active"):
                    self._scan_for_think_end(state)
            return

        for index in batch_update.removed:
            self._state.pop(index, None)

        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            extra = params.extra_args or {}
            constrained_choices = extra.get("constrained_choices", None)

            if constrained_choices is None:
                self._state[index] = {"active": False}
                continue

            enable_thinking = extra.get("enable_thinking_for_constrained", False)

            cache_key = tuple(constrained_choices)
            if cache_key not in self._trie_cache:
                self._trie_cache[cache_key] = self._build_trie(constrained_choices)
                print(
                    f"[ConstrainedDecoding] Built trie for {len(constrained_choices)} choices",
                    flush=True, file=sys.stderr
                )
            trie_root = self._trie_cache[cache_key]

            # For instruct models: constrain from token 0. For thinking: wait for </think>.
            constrained_active = not enable_thinking

            # Log once on first activated request: confirm activation mode
            if not hasattr(self, "_logged_activation"):
                mode = "waiting for </think>" if enable_thinking else "active from token 0"
                print(
                    f"[ConstrainedDecoding] First constrained request activated. "
                    f"enable_thinking={enable_thinking}, constrained_active={constrained_active} ({mode}).",
                    flush=True, file=sys.stderr
                )
                self._logged_activation = True

            self._state[index] = {
                "active": True,
                "trie_root": trie_root,
                "current_node": trie_root,
                "constrained_active": constrained_active,
                "enable_thinking": enable_thinking,
                "output_tok_ids": output_tok_ids,  # LIVE reference
                "prev_output_length": len(output_tok_ids),
                "next_read_idx": len(output_tok_ids),  # index of next token to consume from output_tok_ids
                "completed": False,
            }

        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                s1 = self._state.get(i1, {"active": False})
                s2 = self._state.get(i2, {"active": False})
                self._state[i1] = s2
                self._state[i2] = s1
            else:
                self._state[i2] = self._state.pop(i1, {"active": False})

        for state in self._state.values():
            if state.get("active") and state.get("enable_thinking") and not state.get("constrained_active"):
                self._scan_for_think_end(state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._state:
            return logits

        batch_size = logits.size(0)

        for i in range(batch_size):
            state = self._state.get(i)
            if not state or not state["active"] or not state["constrained_active"]:
                continue

            # Advance trie by consuming all newly committed tokens since last apply().
            # next_read_idx tracks the next unread position in output_tok_ids and only
            # advances when a real (non-sentinel) token is successfully consumed. This
            # keeps it in sync with output_tok_ids regardless of timing or batch gaps.
            if not state["completed"]:
                output = state["output_tok_ids"]
                while state["next_read_idx"] < len(output):
                    tok = output[state["next_read_idx"]]
                    if tok < 0:
                        break  # sentinel, not yet committed — stop and retry next step
                    node = state["current_node"]
                    if tok in node.children:
                        state["current_node"] = node.children[tok]
                        state["next_read_idx"] += 1
                        if state["current_node"].is_terminal:
                            state["completed"] = True
                            print(
                                f"[ConstrainedDecoding] Request completed trie traversal. "
                                f"Answer decoded: {repr(self._tokenizer.decode(output[:state['next_read_idx']]))}",
                                flush=True, file=sys.stderr
                            )
                            break
                    else:
                        print(
                            f"[ConstrainedDecoding] WARNING: committed token {tok} "
                            f"({repr(self._tokenizer.decode([tok]))}) not in trie at depth {state['next_read_idx']}. "
                            f"Valid: {list(node.children.keys())}. Forcing EOS.",
                            flush=True, file=sys.stderr
                        )
                        state["completed"] = True
                        break

            # Loop detection: warn if stuck at root for more than 1 consecutive step.
            # Loop detection: warn if stuck at root for multiple steps while output is non-empty.
            # next_read_idx == len(output) means we've consumed all committed tokens so far
            # (waiting for more), which is normal. Being at root with unconsumed tokens is not.
            if not state["completed"]:
                at_root = state["current_node"] is state["trie_root"]
                has_unconsumed = state["next_read_idx"] < len(state["output_tok_ids"])
                if at_root and has_unconsumed:
                    root_steps = state.get("_root_steps", 0) + 1
                    state["_root_steps"] = root_steps
                    if root_steps == 1:
                        output = state["output_tok_ids"]
                        next_tok = output[state["next_read_idx"]] if state["next_read_idx"] < len(output) else None
                        print(
                            f"[ConstrainedDecoding] WARNING: row={i} at trie root with unconsumed tokens. "
                            f"next_read_idx={state['next_read_idx']}, len(output)={len(output)}, "
                            f"next_tok={next_tok} ({repr(self._tokenizer.decode([next_tok])) if next_tok is not None and next_tok >= 0 else next_tok}), "
                            f"root_children={list(state['trie_root'].children.keys())[:8]}",
                            flush=True, file=sys.stderr
                        )
                    elif root_steps % 500 == 0:
                        print(
                            f"[ConstrainedDecoding] WARNING: row={i} still stuck at trie root, "
                            f"{root_steps} steps, next_read_idx={state['next_read_idx']}, "
                            f"len(output)={len(state['output_tok_ids'])}.",
                            flush=True, file=sys.stderr
                        )
                else:
                    state["_root_steps"] = 0

            if state["completed"]:
                logits[i, :] = -1e9
                if self.eos_token_ids_tensor is not None:
                    logits[i, self.eos_token_ids_tensor] = 0.0
                continue

            valid_tensor = state["current_node"]._valid_token_tensor
            if valid_tensor is not None and len(valid_tensor) > 0:
                mask = torch.full_like(logits[i], -1e9)
                mask[valid_tensor] = logits[i, valid_tensor]
                logits[i] = mask
            else:
                state["completed"] = True
                logits[i, :] = -1e9
                if self.eos_token_ids_tensor is not None:
                    logits[i, self.eos_token_ids_tensor] = 0.0

        return logits


def _vllm_mp_worker(
    model_args: dict,
    sampling_params: "list[SamplingParams]",
    requests: list[list[int]],
    lora_request: "LoRARequest",
    result_queue: "Queue",
    dp_size: int,
    local_dp_rank: int,
    dp_master_port: int,
    dp_master_ip: str = "127.0.0.1",
) -> None:
    """
    Worker process for vLLM multiprocessing.
    Initializes a vLLM engine, processes requests, and puts results or errors
    onto the result_queue.
    """

    if not requests:
        result_queue.put((local_dp_rank, []))
        return None

    os.environ["VLLM_DP_RANK"] = os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = str(dp_master_ip)
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    llm = None
    try:
        llm = LLM(**model_args)
        res = llm.generate(
            [TokensPrompt(prompt_token_ids=request) for request in requests],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        # Give engines time to pause their processing loops before exiting."
        sleep(1)
        result_queue.put((local_dp_rank, res))

    except Exception as e:
        error_message = f"Worker {local_dp_rank} failed during generation: {type(e).__name__}: {str(e)}"
        eval_logger.error(error_message, exc_info=True)
        result_queue.put((local_dp_rank, {"error": error_message}))

    finally:
        if llm is not None:
            try:
                del llm
                gc.collect()
            except Exception as e_cleanup:
                eval_logger.warning(
                    f"Worker {local_dp_rank} encountered an error during LLM cleanup: {type(e_cleanup).__name__}: {str(e_cleanup)}",
                    exc_info=True,
                )

    return None


@register_model("vllm")
class VLLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: Union[str, int] = 1,
        max_batch_size=None,
        max_length: int = None,
        max_model_len: int = None,
        seed: int = 1234,
        gpu_memory_utilization: float = 0.9,
        data_parallel_size: int = 1,
        lora_local_path: str = None,
        # VLLM: enable thinking tags in the prompt.
        enable_thinking: bool = True,
        chat_template_args: Optional[dict] = None,
        # End marker for thinking tags - splits to get response after this token (if provided).
        think_start_token: Optional[str] = None,
        think_end_token: Optional[str] = None,
        answer_prefix: Optional[str] = None,
        max_think_tokens: Optional[int] = None,
        min_think_tokens: Optional[int] = None,
        max_lora_rank: int = 16,
        **kwargs,
    ):
        super().__init__()

        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install lm-eval[vllm]` or `pip install -e .[vllm]`"
            )

        assert max_length is None or max_model_len is None, (
            "Either max_length or max_model_len may be provided, but not both"
        )
        kwargs.pop("device", None)
        self.think_end_token = think_end_token
        self.answer_prefix = answer_prefix
        self.V1 = os.environ.get("VLLM_USE_V1", "1") != "0"
        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.data_parallel_size = int(data_parallel_size)
        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "max_num_seqs": kwargs.get("max_num_seqs", max_batch_size),
            "swap_space": int(swap_space),
            "quantization": quantization,
            "seed": int(seed),
            "enable_lora": True if lora_local_path else False,
            "max_lora_rank": int(max_lora_rank),
        }
        
        from transformers import AutoConfig

        self._config = AutoConfig.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code, revision=revision
        )
        self.tokenizer = get_tokenizer(
            tokenizer if tokenizer else pretrained,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            revision=tokenizer_revision,
            add_bos_token=add_bos_token,
        )
        self.tokenizer = configure_pad_token(
            self.tokenizer, model_config=self._config
        )
        self.chat_template_args = chat_template_args or {}
        self.enable_thinking = self.chat_template_args.pop(
            "enable_thinking", enable_thinking
        )
        self.add_bos_token = add_bos_token

        # Register logits processors
        processors = []
        if self.enable_thinking:
            processors.append(ThinkingTokenBudgetLogitsProcessor)
            eval_logger.info(
                "VLLM.__init__: Registered ThinkingTokenBudgetLogitsProcessor"
            )

        # ConstrainedChoiceLogitsProcessor handles constrained decoding for Banking77 and
        # any task that provides 'guided_choice' in gen_kwargs. It is a no-op for requests
        # that do not include 'constrained_choices' in extra_args.
        processors.append(ConstrainedChoiceLogitsProcessor)
        eval_logger.info(
            "VLLM.__init__: Registered ConstrainedChoiceLogitsProcessor"
        )

        eval_logger.info(
            f"VLLM.__init__: Total logits processors: {len(processors)}"
        )

        if processors:
            self.model_args['logits_processors'] = processors
            eval_logger.info(
                f"VLLM.__init__: Stored {len(processors)} processors"
            )

        self.model_args.update(kwargs)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        if self.data_parallel_size <= 1:
            self.model = LLM(
                **self.model_args
            )

            print("vLLM model initialized successfully", flush=True, file=sys.stderr)
            print(f"Logits processors registered: {len(processors)}", flush=True, file=sys.stderr)
        else:
            eval_logger.warning(
                "You might experience occasional issues with model weight downloading when data_parallel is in use. To ensure stable performance, run with data_parallel_size=1 until the weights are downloaded and cached."
            )
            self.model_args["distributed_executor_backend"] = (
                "ray"
                if not self.V1
                else self.model_args.get("distributed_executor_backend", None)
            )
            self.batch_size = "auto"
            eval_logger.info("Manual batching is not compatible with data parallelism.")

        if "gemma" in pretrained.lower():
            add_bos_token = True
            eval_logger.info(
                "Found 'gemma' in model name, a BOS token will be used as Gemma series models underperform without it."
            )

        if parse_version(version("vllm")) >= parse_version("0.8.3"):
            kwargs_resolve_hf_chat_template = {
                "tokenizer": self.tokenizer,
                "chat_template": None,
                "tools": None,
            }

            if parse_version(version("vllm")) >= parse_version("0.9.0"):
                if self.data_parallel_size <= 1:
                    kwargs_resolve_hf_chat_template["model_config"] = (
                        self.model.llm_engine.model_config
                    )
                else:
                    from vllm.engine.arg_utils import EngineArgs

                    engine_args = EngineArgs(**self.model_args)
                    model_config = engine_args.create_model_config()

                    kwargs_resolve_hf_chat_template["model_config"] = model_config
            else:
                kwargs_resolve_hf_chat_template["trust_remote_code"] = trust_remote_code

            self.hf_chat_template = resolve_hf_chat_template(
                **kwargs_resolve_hf_chat_template
            )
        else:
            self.hf_chat_template = None

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

        self._max_gen_toks = max_gen_toks

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version("0.3.0"), (
                "lora adapters only compatible with vllm > v0.3.0."
            )
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                chat_template=self.hf_chat_template,
                enable_thinking=self.enable_thinking,
                **self.chat_template_args,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_templated = self.tokenizer.apply_chat_template(
                [msg for msg in chat_history if msg["role"] != "system"],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                chat_template=self.hf_chat_template,
                enable_thinking=self.enable_thinking,
                **self.chat_template_args,
            )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        encoding: Union[List[List[int]], List[int]] = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]

        return encoding

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        sampling_params: Union[List[SamplingParams], SamplingParams, None] = None,
    ):
        if not generate or sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )
        if not isinstance(sampling_params, List):
            sampling_params = [sampling_params] * len(requests)

        answer_prefix_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.answer_prefix))
        if self.answer_prefix is not None:
            for sample_params in sampling_params:
                if sample_params.extra_args is None:
                    sample_params.extra_args = {}
                sample_params.extra_args['answer_prefix_ids'] = answer_prefix_ids

        if self.data_parallel_size > 1 and not self.V1:
            # vLLM hangs if resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            @ray.remote
            def run_inference_one_model(
                model_args: dict,
                sampling_params: List[SamplingParams],
                requests: List[List[int]],
                lora_request: LoRARequest,
            ):
                llm = LLM(**model_args)
                return llm.generate(
                    [TokensPrompt(prompt_token_ids=request) for request in requests],
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            sampling_params = [
                list(sp) for sp in distribute(self.data_parallel_size, sampling_params)
            ]
            inputs = (
                (self.model_args, sp, req, self.lora_request)
                for req, sp in zip(requests, sampling_params)
            )
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            return undistribute(results)
        elif self.data_parallel_size > 1:
            # based on https://github.com/vllm-project/vllm/blob/a04720bc36401d831cb048c3917b9e58173d9c1d/examples/offline_inference/data_parallel.py
            dp_size = self.data_parallel_size
            dp_master_ip = os.environ.get("VLLM_DP_MASTER_IP", "127.0.0.1")
            dp_master_port = os.environ.get("VLLM_DP_MASTER_PORT") or get_open_port()

            requests = (list(x) for x in distribute(self.data_parallel_size, requests))
            sampling_params = (
                list(sp) for sp in distribute(self.data_parallel_size, sampling_params)
            )
            procs, resq = [], Queue()
            # We use Process as it is non-daemonic
            try:
                for rank, (sp, req) in enumerate(zip(requests, sampling_params)):
                    proc = Process(
                        target=_vllm_mp_worker,
                        args=(
                            self.model_args.copy(),
                            sp,
                            req,
                            self.lora_request,
                            resq,
                            dp_size,
                            rank,
                            dp_master_port,
                            dp_master_ip,
                        ),
                    )
                    proc.start()
                    procs.append(proc)

                # Collect results
                rank_res = {}
                while len(rank_res) < len(procs):
                    try:
                        rank, result = resq.get(timeout=30)
                        if isinstance(result, dict) and "error" in result:
                            raise RuntimeError(result["error"])
                        rank_res[rank] = result
                    except Empty:
                        dead_procs = [
                            idx
                            for idx, p in enumerate(procs)
                            if not p.is_alive() and idx not in rank_res
                        ]
                        if dead_procs:
                            raise RuntimeError(
                                f"Worker processes {dead_procs} died unexpectedly"
                            )
                        continue

                results = [rank_res[i] for i in range(len(procs))]
                return undistribute(results)

            # cleanup
            finally:
                try:
                    resq.close()
                    resq.join_thread()
                except Exception:
                    eval_logger.debug(
                        "Failed to close vllm DP results queue", exc_info=True
                    )
                for proc in procs:
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=5)
                        if proc.is_alive():
                            proc.kill()

        else:
            outputs = self.model.generate(
                [TokensPrompt(prompt_token_ids=request) for request in requests],
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )
            return outputs

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        adaptive_batch_size = None
        if self.batch_size == "auto":
            adaptive_batch_size = len(requests)

        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        # max_seq_len - (1 for context)
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        all_nlls = []
        batch_size = adaptive_batch_size or int(self.batch_size)
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
            )
            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]] = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        re_ords = Collator(
            requests,
            _collate_gen,
            group_by=None,
        )
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        eos = self.tokenizer.decode(self.eot_token_id)
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            context_encoding_truncated = []
            sampling_params = []
            for x, gen_kwargs in zip(context_encoding, all_gen_kwargs):
                # unpack our keyword arguments.
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    # add EOS token to stop sequences
                    until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                    )
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks

                # set the max length in tokens of inputs ("context_enc")
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                if len(x) > max_ctx_len:
                    eval_logger.warning(
                        f"Context length {len(x)} exceeds max length (context + max gen tokens): {max_ctx_len}. Truncating context."
                    )
                    context_encoding_truncated.append(x[-max_ctx_len:])
                else:
                    context_encoding_truncated.append(x)
                # create sampling params
                kwargs = self.modify_gen_kwargs(kwargs)

                sampling_params.append(
                    SamplingParams(max_tokens=max_gen_toks, stop=until, **kwargs)
                )

            # perform batched generation
            cont = self._model_generate(
                requests=context_encoding_truncated,
                generate=True,
                sampling_params=sampling_params
            )

            # cache generations
            for output, context in zip(cont, context):
                generated_text: str = output.outputs[0].text
                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                generated_text = postprocess_generated_text(
                    generated_text, until, self.think_end_token
                )
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                if (
                    full_length := len(context_enc + continuation_enc)
                ) > self.max_length:
                    eval_logger.warning(
                        f"Context length {full_length} exceeds max length ({self.max_length}). Truncating context."
                    )
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)

            for output, ctxlen, (cache_key, _, _), inp in zip(
                outputs, ctxlens, chunk, inputs
            ):
                answer = self._parse_logprobs(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                )

                res.append(answer)

                if cache_key is not None:
                    # special case: loglikelihood_rolling produces a number of loglikelihood requests
                    # all with cache key None. instead do add_partial on the per-example level
                    # in the loglikelihood_rolling() function for those.
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            # vLLM changed the return type of logprobs from float
            # to a Logprob object storing the float value + extra data
            # (https://github.com/vllm-project/vllm/pull/3065).
            # If we are dealing with vllm's Logprob object, return
            # the logprob value stored as an attribute. Otherwise,
            # return the object itself (which should be a float
            # for older versions of vLLM).
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
            for logprob_dict in continuation_logprobs_dicts
        ]

        # Calculate continuation_logprobs
        # assume ctxlen always >= 1
        continuation_logprobs = sum(
            logprob_dict.get(token)
            for token, logprob_dict in zip(
                tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
            )
        )

        # Determine if is_greedy
        is_greedy = True
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        ):
            # Get the token with the maximum log probability from the logprob_dict
            if logprob_dict:  # Ensure the logprob_dict is not None
                top_token = max(logprob_dict, key=logprob_dict.get)
                if top_token != token:
                    is_greedy = False
                    break

        return continuation_logprobs, is_greedy

    def modify_gen_kwargs(self, kwargs: dict) -> dict:
        # sampling_params
        kwargs["temperature"] = kwargs.get("temperature", 0.0)
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False and "temperature" not in kwargs:
            eval_logger.debug(
                "Got `do_sample=False` and no temperature value, setting VLLM temperature to 0.0 ..."
            )
            kwargs["temperature"] = 0.0
        # hf defaults
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get(
            "spaces_between_special_tokens", False
        )

        # Handle guided_choice: inject into extra_args for ConstrainedChoiceLogitsProcessor.
        # Works for both thinking models - constraints activate after </think>
        # and instruct models - constraints active from token 0.
        # Note: extra_args from gen_kwargs.yaml arrives as a list of single-key dicts;
        # normalise it to a flat dict before adding our keys.
        guided_choice = kwargs.pop("guided_choice", None)
        if guided_choice is not None:
            existing = kwargs.get("extra_args", None)
            if isinstance(existing, list):
                # YAML list-of-dicts → flatten into a single dict
                flat: dict = {}
                for item in existing:
                    if isinstance(item, dict):
                        flat.update(item)
                kwargs["extra_args"] = flat
            elif existing is None:
                kwargs["extra_args"] = {}
            kwargs["extra_args"]["constrained_choices"] = guided_choice
            kwargs["extra_args"]["enable_thinking_for_constrained"] = bool(self.enable_thinking)

        return kwargs
