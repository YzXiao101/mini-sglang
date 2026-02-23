from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    last_fail_reason: str = ""
    last_fail_deficit: int = 0
    last_fail_estimated_len: int = 0

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            self.last_fail_reason = "table"
            return None

        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        needed = estimated_len + self.reserved_size
        available = self.cache_manager.available_size
        if needed > available:
            self.last_fail_reason = "estimate"
            self.last_fail_deficit = needed - available
            self.last_fail_estimated_len = estimated_len
            return None
        self.cache_manager.lock(handle)
        available = self.cache_manager.available_size
        if needed > available:
            self.cache_manager.unlock(handle)
            self.last_fail_reason = "estimate"
            self.last_fail_deficit = needed - available
            self.last_fail_estimated_len = estimated_len
            return None

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(match_indices)

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx][_slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        self.last_fail_reason = ""
        self.last_fail_deficit = 0
        self.last_fail_estimated_len = 0
        if self.token_budget <= 0:
            self.last_fail_reason = "budget"
            return None

        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)
    prompt_len_map: Dict[int, int] = field(default_factory=dict)
    max_tokens_map: Dict[int, int] = field(default_factory=dict)
    prefill_rounds: int = 0
    estimate_reject_count: int = 0
    estimate_reject_deficit_tokens: int = 0
    estimate_reject_estimated_tokens: int = 0
    estimate_hol_blocked_rounds: int = 0
    estimate_hol_blocked_reqs: int = 0
    decode_reserved_tokens: int = 0
    decode_realized_tokens: int = 0
    finished_reqs: int = 0
    aborted_reqs: int = 0

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))
        self.prompt_len_map[req.uid] = len(req.input_ids)
        self.max_tokens_map[req.uid] = req.sampling_params.max_tokens

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None
        self.prefill_rounds += 1

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for i, pending_req in enumerate(self.pending_list):
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                if adder.last_fail_reason == "estimate":
                    self.estimate_reject_count += 1
                    self.estimate_reject_deficit_tokens += adder.last_fail_deficit
                    self.estimate_reject_estimated_tokens += adder.last_fail_estimated_len
                    blocked_reqs = len(self.pending_list) - i - 1
                    if blocked_reqs > 0:
                        self.estimate_hol_blocked_rounds += 1
                        self.estimate_hol_blocked_reqs += blocked_reqs
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                self.on_req_aborted(uid)
                return req.chunked_req
        return None

    def on_req_finished(self, req: Req) -> None:
        prompt_len = self.prompt_len_map.pop(req.uid, len(req.input_ids))
        max_tokens = self.max_tokens_map.pop(req.uid, req.output_len)
        realized = len(req.input_ids) - prompt_len
        realized = max(0, min(realized, max_tokens))
        self.finished_reqs += 1
        self.decode_reserved_tokens += max_tokens
        self.decode_realized_tokens += realized

    def on_req_aborted(self, uid: int) -> None:
        had_prompt = uid in self.prompt_len_map
        had_max_tokens = uid in self.max_tokens_map
        self.prompt_len_map.pop(uid, None)
        self.max_tokens_map.pop(uid, None)
        if had_prompt or had_max_tokens:
            self.aborted_reqs += 1

    def reset_estimation_metrics(self) -> None:
        self.prompt_len_map.clear()
        self.max_tokens_map.clear()
        self.prefill_rounds = 0
        self.estimate_reject_count = 0
        self.estimate_reject_deficit_tokens = 0
        self.estimate_reject_estimated_tokens = 0
        self.estimate_hol_blocked_rounds = 0
        self.estimate_hol_blocked_reqs = 0
        self.decode_reserved_tokens = 0
        self.decode_realized_tokens = 0
        self.finished_reqs = 0
        self.aborted_reqs = 0

    def estimation_metrics(self) -> Dict[str, float]:
        reject_round_ratio = (
            self.estimate_reject_count / self.prefill_rounds if self.prefill_rounds > 0 else 0.0
        )
        avg_reject_deficit = (
            self.estimate_reject_deficit_tokens / self.estimate_reject_count
            if self.estimate_reject_count > 0
            else 0.0
        )
        avg_reject_estimated_tokens = (
            self.estimate_reject_estimated_tokens / self.estimate_reject_count
            if self.estimate_reject_count > 0
            else 0.0
        )
        decode_unused = max(0, self.decode_reserved_tokens - self.decode_realized_tokens)
        decode_unused_ratio = (
            decode_unused / self.decode_reserved_tokens if self.decode_reserved_tokens > 0 else 0.0
        )
        return {
            "prefill_rounds": float(self.prefill_rounds),
            "estimate_reject_count": float(self.estimate_reject_count),
            "estimate_reject_round_ratio": reject_round_ratio,
            "estimate_reject_avg_deficit_tokens": avg_reject_deficit,
            "estimate_reject_avg_estimated_tokens": avg_reject_estimated_tokens,
            "estimate_hol_blocked_rounds": float(self.estimate_hol_blocked_rounds),
            "estimate_hol_blocked_reqs": float(self.estimate_hol_blocked_reqs),
            "decode_reserved_tokens": float(self.decode_reserved_tokens),
            "decode_realized_tokens": float(self.decode_realized_tokens),
            "decode_unused_tokens": float(decode_unused),
            "decode_unused_ratio": decode_unused_ratio,
            "finished_reqs": float(self.finished_reqs),
            "aborted_reqs": float(self.aborted_reqs),
            "tracked_reqs": float(len(self.prompt_len_map)),
        }

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
