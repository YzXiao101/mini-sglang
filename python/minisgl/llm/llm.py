from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, DetokenizeMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig

if TYPE_CHECKING:
    from minisgl.scheduler.scheduler import ForwardInput


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


@dataclass(frozen=True)
class ProfileSummary:
    phase: Literal["decode", "all"]
    requested_start_step: int
    requested_end_step: int | None
    total_counted_steps: int
    profile_started: bool
    profile_stopped: bool
    actual_start_step: int | None
    actual_end_step: int | None
    stop_reason: str | None

    def as_dict(self) -> Dict[str, int | float | str | bool | None]:
        return {
            "phase": self.phase,
            "requested_start_step": self.requested_start_step,
            "requested_end_step": self.requested_end_step,
            "total_counted_steps": self.total_counted_steps,
            "profile_started": self.profile_started,
            "profile_stopped": self.profile_stopped,
            "actual_start_step": self.actual_start_step,
            "actual_end_step": self.actual_end_step,
            "stop_reason": self.stop_reason,
        }


@dataclass
class _ProfileState:
    profiler: object
    device: torch.device
    phase: Literal["decode", "all"]
    start_step: int
    end_step: int | None
    counted_steps: int = 0
    started: bool = False
    active: bool = False
    stopped: bool = False
    actual_start_step: int | None = None
    actual_end_step: int | None = None
    stop_reason: str | None = None

    def _counts(self, forward_input: ForwardInput) -> bool:
        return self.phase == "all" or forward_input.batch.is_decode

    def _start(self, step: int) -> None:
        self.profiler.start()
        self.started = True
        self.active = True
        self.actual_start_step = step

    def _stop(self, step: int, reason: str) -> None:
        torch.cuda.synchronize(self.device)
        self.profiler.stop()
        self.active = False
        self.stopped = True
        self.actual_end_step = step
        self.stop_reason = reason

    def before_forward(self, forward_input: ForwardInput) -> None:
        if self.stopped or not self._counts(forward_input):
            return
        next_step = self.counted_steps + 1
        if self.active and self.end_step is not None and next_step > self.end_step:
            self._stop(self.counted_steps, "step_limit")
            return
        self.counted_steps = next_step
        if not self.started and self.counted_steps >= self.start_step:
            self._start(self.counted_steps)

    def finish(self) -> ProfileSummary:
        if self.active:
            self._stop(
                self.actual_end_step or self.counted_steps,
                "generate_end",
            )
        elif not self.started:
            self.stop_reason = "not_started"

        return ProfileSummary(
            phase=self.phase,
            requested_start_step=self.start_step,
            requested_end_step=self.end_step,
            total_counted_steps=self.counted_steps,
            profile_started=self.started,
            profile_stopped=self.stopped,
            actual_start_step=self.actual_start_step,
            actual_end_step=self.actual_end_step,
            stop_reason=self.stop_reason,
        )


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0
        self.profile_state: _ProfileState | None = None
        self.last_profile_summary: ProfileSummary | None = None

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    def _before_forward_batch(self, forward_input: ForwardInput) -> None:
        if self.profile_state is not None:
            self.profile_state.before_forward(forward_input)

    def start_profile(
        self,
        profiler: object,
        *,
        start_step: int = 1,
        num_steps: int | None = None,
        phase: Literal["decode", "all"] = "decode",
    ) -> None:
        if self.profile_state is not None:
            raise RuntimeError("profiling is already configured")
        if start_step < 1:
            raise ValueError("start_step must be >= 1")
        if num_steps is not None and num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        self.last_profile_summary = None
        self.profile_state = _ProfileState(
            profiler=profiler,
            device=self.device,
            phase=phase,
            start_step=start_step,
            end_step=None if num_steps is None else start_step + num_steps - 1,
        )

    def stop_profile(self) -> ProfileSummary:
        if self.profile_state is None:
            raise RuntimeError("profiling is not configured")
        summary = self.profile_state.finish()
        self.profile_state = None
        self.last_profile_summary = summary
        return summary

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        self.last_profile_summary = None
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))

        try:
            self.run_forever()
        except RequestAllFinished:
            pass

        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
