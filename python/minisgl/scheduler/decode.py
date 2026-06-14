from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Set

from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.utils import alloc_delta

if TYPE_CHECKING:
    from .cache import CacheManager
    from .metric_sink import SchedulerMetricSink
    from .prefill import PrefillManager
    from .table import TableManager


@dataclass
class _EstimatePolicy:
    page_size: int
    init_new_token_ratio: float = field(init=False)
    min_new_token_ratio: float = field(init=False)
    new_token_ratio_decay: float = field(init=False)
    new_token_ratio: float = field(init=False)
    clip_max_new_tokens: int = field(init=False)
    retract_decode_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.init_new_token_ratio = min(
            ENV.INIT_NEW_TOKEN_RATIO.value * ENV.SCHEDULE_CONSERVATIVENESS.value, 1.0
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * ENV.MIN_NEW_TOKEN_RATIO_FACTOR.value, 1.0
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / ENV.NEW_TOKEN_RATIO_DECAY_STEPS.value
        self.new_token_ratio = self.init_new_token_ratio
        self.clip_max_new_tokens = ENV.CLIP_MAX_NEW_TOKENS.value
        self.retract_decode_steps = ENV.RETRACT_DECODE_STEPS.value

    def reset(self) -> None:
        self.new_token_ratio = self.init_new_token_ratio

    def estimated_inflight_tokens(self, reqs: Iterable[Req]) -> int:
        reserved_size = 0
        for req in reqs:
            if req.sampling_params.ignore_eos:
                tail_est = req.remain_len
            else:
                tail_est = math.ceil(
                    min(req.remain_len, self.clip_max_new_tokens) * self.new_token_ratio
                )
            reserved_size += alloc_delta(req.cached_len, req.extend_len + tail_est, self.page_size)
        return reserved_size

    def on_decode_success(self) -> None:
        self.new_token_ratio = max(
            self.min_new_token_ratio,
            self.new_token_ratio - self.new_token_ratio_decay,
        )

    def on_retract(self, reqs: Iterable[Req]) -> None:
        reqs = list(reqs)
        decoded_tokens = sum(len(req.input_ids) - req.prompt_len for req in reqs)
        total_tokens = sum(req.max_device_len - req.prompt_len for req in reqs)
        self.new_token_ratio = min(
            1.0,
            (decoded_tokens + self.retract_decode_steps * len(reqs)) / total_tokens,
        )


@dataclass
class DecodeManager:
    EstimatePolicy = _EstimatePolicy

    page_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    metric_sink: SchedulerMetricSink
    running_reqs: Set[Req] = field(default_factory=set)
    estimate_policy: "DecodeManager.EstimatePolicy" = field(init=False)

    def __post_init__(self) -> None:
        self.estimate_policy = self.EstimatePolicy(self.page_size)
        self.metric_sink.emit(
            "estimate_policy_init",
            page_size=self.page_size,
            init_new_token_ratio=self.estimate_policy.init_new_token_ratio,
            min_new_token_ratio=self.estimate_policy.min_new_token_ratio,
            new_token_ratio_decay=self.estimate_policy.new_token_ratio_decay,
            clip_max_new_tokens=self.estimate_policy.clip_max_new_tokens,
            retract_decode_steps=self.estimate_policy.retract_decode_steps,
            new_token_ratio=self.estimate_policy.new_token_ratio,
        )

    def reset_new_token_ratio(self) -> None:
        ratio_before = self.estimate_policy.new_token_ratio
        self.estimate_policy.reset()
        self.metric_sink.emit(
            "ratio_update",
            reason="reset",
            ratio_before=ratio_before,
            ratio_after=self.estimate_policy.new_token_ratio,
            running_req_count=len(self.running_reqs),
        )

    @property
    def clip_max_new_tokens(self) -> int:
        return self.estimate_policy.clip_max_new_tokens

    @property
    def estimated_inflight_tokens(self) -> int:
        return self.estimate_policy.estimated_inflight_tokens(self.running_reqs)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    def _decode_mem_need(self, steps: int = 1) -> int:
        return sum(
            alloc_delta(
                req.cached_len,
                min(req.remain_len + req.extend_len, steps),
                self.page_size,
            )
            for req in self.running_reqs
        )

    def _emit_decode_success(
        self,
        *,
        running_req_count: int,
        available_size: int,
        need_next: int,
        ratio_before: float,
    ) -> None:
        self.metric_sink.emit_sampled(
            "decode_schedule",
            key="decode_success",
            decision="success",
            running_req_count=running_req_count,
            available_size=available_size,
            need_next=need_next,
            new_token_ratio=self.estimate_policy.new_token_ratio,
            ratio_before=ratio_before,
        )
        self.metric_sink.emit_sampled(
            "ratio_update",
            key="ratio_success",
            reason="decode_success",
            ratio_before=ratio_before,
            ratio_after=self.estimate_policy.new_token_ratio,
            running_req_count=running_req_count,
        )

    def schedule_next_batch(
        self,
        prefill_manager: PrefillManager,
    ) -> Batch | None:
        if not self.runnable:
            return None
        running_req_count = len(self.running_reqs)
        available_size = self.cache_manager.available_size
        need_next = self._decode_mem_need()
        if need_next <= available_size:
            ratio_before = self.estimate_policy.new_token_ratio
            self.estimate_policy.on_decode_success()
            if self.metric_sink.enabled:
                self._emit_decode_success(
                    running_req_count=running_req_count,
                    available_size=available_size,
                    need_next=need_next,
                    ratio_before=ratio_before,
                )
            return Batch(reqs=list(self.running_reqs), phase="decode")

        retracted_reqs = []
        need_runway_before = self._decode_mem_need(self.estimate_policy.retract_decode_steps)
        need_runway = need_runway_before
        while need_runway > self.cache_manager.available_size:
            if len(self.running_reqs) == 1:
                req = next(iter(self.running_reqs))
                self.metric_sink.emit(
                    "decode_schedule",
                    decision="oom",
                    running_req_count_before=running_req_count,
                    running_req_count_after=len(self.running_reqs),
                    available_size_before=available_size,
                    available_size_after=self.cache_manager.available_size,
                    need_next=need_next,
                    need_runway_before=need_runway_before,
                    need_runway_after=need_runway,
                    retract_decode_steps=self.estimate_policy.retract_decode_steps,
                    retracted_count=len(retracted_reqs),
                    new_token_ratio=self.estimate_policy.new_token_ratio,
                )
                raise RuntimeError(
                    f"Decode OOM ! retract_decode_steps={self.estimate_policy.retract_decode_steps}, "
                    f"cached_len={req.cached_len}"
                )
            req = min(
                self.running_reqs,
                key=lambda req: (len(req.input_ids) - req.prompt_len, -req.prompt_len, req.uid),
            )
            self.running_reqs.remove(req)
            req.is_retracted = True
            self.table_manager.free(req.table_idx)
            self.cache_manager.cache_req(req, finished=True)
            retracted_reqs.append(req)
            need_runway = self._decode_mem_need(self.estimate_policy.retract_decode_steps)
        prefill_manager.requeue_reqs(retracted_reqs)

        batch = Batch(reqs=list(self.running_reqs), phase="decode")
        ratio_before = self.estimate_policy.new_token_ratio
        self.estimate_policy.on_retract(batch.reqs)
        self.metric_sink.emit(
            "ratio_update",
            reason="retract",
            ratio_before=ratio_before,
            ratio_after=self.estimate_policy.new_token_ratio,
            running_req_count=len(self.running_reqs),
            retracted_count=len(retracted_reqs),
        )
        self.metric_sink.emit(
            "decode_schedule",
            decision="retract",
            running_req_count_before=running_req_count,
            running_req_count_after=len(self.running_reqs),
            available_size_before=available_size,
            available_size_after=self.cache_manager.available_size,
            need_next=need_next,
            need_runway_before=need_runway_before,
            need_runway_after=need_runway,
            retract_decode_steps=self.estimate_policy.retract_decode_steps,
            retracted_count=len(retracted_reqs),
            new_token_ratio=self.estimate_policy.new_token_ratio,
        )
        return batch

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
