# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Environment switches of dense DWDP; no ``llm_args`` field.

* ``TRTLLM_DWDP_DENSE`` -- turn dense DWDP on; default off.  With ``dwdp_config`` it runs with
  expert DWDP over the DWDP group; without, standalone over the attention-DP group of an instance
  with ``enable_attention_dp`` and ``tp_size > 1`` (``gate.validate_dense_gate``), in graph mode
  when the launch also has ``cuda_graph_config`` (``gate.dense_graph_mode``).
* ``TRTLLM_DWDP_DENSE_SEGMENTS`` -- ``attn,moe`` (default) or ``attn`` for attention only.
* ``TRTLLM_DWDP_DENSE_REQUIRED`` -- default on: a failed setup raises on every rank; off: every rank
  drops dense DWDP and keeps serving with replicated dense weights.
* ``TRTLLM_DWDP_DENSE_MIN_BYTES`` -- storages smaller than this stay replicated; default 1 MiB.
* ``TRTLLM_DWDP_DENSE_STORE_CHUNK_MB`` -- 0 (default): one store handle; > 0: whole layers per
  handle of about this many MiB.
* ``TRTLLM_DWDP_DENSE_STRICT`` -- turn demand fetches, leaks, unknown large attributes, shared
  storages and slot watchdog mismatches into errors; default off.  It must be the same on every
  rank: when the ranks differ, setup fails on every rank, which raises unless
  ``TRTLLM_DWDP_DENSE_REQUIRED`` is off.
* ``TRTLLM_DWDP_DENSE_DEBUG`` -- 1: fill the slots with NaN at commit, and name the holders of
  the first 8 moved storages that outlive the commit (each a scan of every live Python object);
  2: also run the per-layer slot checksum watchdog (each layer's H1 checks its slots against the
  layer's checksum; slow: every slot is read once more and the host waits for the GPU once per
  forward); 3: also, in graph mode, assert on the device at the end of every forward (H4) that
  the watchdog found no mismatch, so that a mismatch stops the process without a host
  synchronization (for validation runs); default 0.
* ``TRTLLM_DWDP_DENSE_COPY`` -- the transfer of standalone dense DWDP, on the dense manager's own
  copy stream: ``batched`` (default; one ``cudaMemcpyBatchAsync`` per layer on the copy engines),
  ``per_slice`` or ``gather`` (the SM fabric gather, which holds SMs next to the MoE all-to-all;
  validated on a Rubin context server with ``TRTLLM_DWDP_GATHER_KERNEL=tma`` and
  ``TLLM_ROUTING_DISABLE_COOP_LAUNCH=1``).  The gate rejects an invalid value standalone.  Graph
  mode needs it set, to ``gather`` or ``batched``, and rejects ``per_slice``: with ``batched`` the
  decode CUDA graphs capture copy-engine copies, which hold no SMs, and the gate first checks that
  the CUDA driver captures ``cudaMemcpyBatchAsync``; with ``gather`` they capture the SM gather
  kernel, which holds SMs.  On Rubin (SM107), where generation servers were validated with the
  SM gather, the gate recommends ``gather`` over ``batched``.  With expert DWDP the dense copies
  follow the expert transfer, and the gate warns that any value, valid or not, is ignored.
* ``TRTLLM_DWDP_DENSE_PRIME`` -- when graph mode gathers the first two covered layers of a step:
  ``wrap`` (default; at the last two covered layers of the step before, so that the copies overlap
  the tail of that step: final norm, ``lm_head``, drafter and sampling) or ``entry`` (at the start
  of every forward; a fallback for debugging and A/B runs).  Outside graph mode dense DWDP primes
  at the start of every forward, and the gate warns that any value, valid or not, is ignored.
* ``TRTLLM_DWDP_DENSE_PDL_BREAK`` -- the PDL break (``types.PdlBreak``): H1 of every covered
  layer, right after the compute stream waits for the layer's gather, and H2, after the gather it
  enqueues, launch one tiny kernel without programmatic dependent launch (PDL) on the compute
  stream.  It keeps the PDL-launched kernel after the hook from being dispatched ahead of the
  gather that becomes eligible with it, which would put the gather's clusters on the SMs that
  MegaMoE needs for its 4-CTA clusters and stall MegaMoE until the gather ends.  Default: on with
  the SM gather in graph mode (the closed protocol, where that stall was diagnosed); off with the
  copy engines, which hold no SMs and whose copies a CUDA graph orders before a PDL-launched
  kernel only through full dependencies, and outside graph mode (the legacy protocol); ``0``
  turns it off (A/B runs), ``1`` on everywhere.  It adds one kernel launch per hook; setup logs
  whether it is on.

Booleans accept ``1/true/yes/on`` and ``0/false/no/off`` (any case); unset or empty means the
default.  A flag that defaults to off is on only for an explicit true value; any other value logs
a warning and keeps the default.  ``TRTLLM_DWDP_DENSE_PDL_BREAK`` has no fixed default: unset,
empty or another value (with the warning) leave it to the transfer and the protocol
(:meth:`DenseConfig.pdl_break_enabled`).  Integers that do not parse, or are negative, log a
warning and keep the default.  An invalid segment list raises.  ``TRTLLM_DWDP_DENSE_COPY`` and
``TRTLLM_DWDP_DENSE_PRIME`` are read as set and validated only where they apply: standalone
(:meth:`DenseConfig.standalone_copy_mode`) and in graph mode (:meth:`DenseConfig.prime_policy`).

Standalone dense DWDP serves context servers, normally without CUDA graphs, and generation
servers, with them.  Graph mode runs the ``CLOSED`` protocol, which captures the dense schedule of
a whole step into every decode CUDA graph; expert DWDP and standalone dense DWDP without CUDA
graphs keep the ``LEGACY`` protocol.  Either way every forward gathers every covered layer.  The
gate logs the server role, which a worker sees as ``TRTLLM_DISAGG_ROLE`` only when
``trtllm-serve`` launches it with a role (``--server_role``, or a ctx or gen server in a
disaggregated config), but does not check it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from tensorrt_llm.logger import logger

from .types import (
    DEBUG_FILL_BYTE,
    DEFAULT_MIN_BYTES,
    LOG_PREFIX,
    SEGMENT_ORDER,
    CopyMode,
    DigestSwitches,
    PrimePolicy,
    SegmentKind,
)

DENSE_ENV = "TRTLLM_DWDP_DENSE"
DENSE_SEGMENTS_ENV = "TRTLLM_DWDP_DENSE_SEGMENTS"
DENSE_REQUIRED_ENV = "TRTLLM_DWDP_DENSE_REQUIRED"
DENSE_MIN_BYTES_ENV = "TRTLLM_DWDP_DENSE_MIN_BYTES"
DENSE_STORE_CHUNK_MB_ENV = "TRTLLM_DWDP_DENSE_STORE_CHUNK_MB"
DENSE_STRICT_ENV = "TRTLLM_DWDP_DENSE_STRICT"
DENSE_DEBUG_ENV = "TRTLLM_DWDP_DENSE_DEBUG"
DENSE_COPY_ENV = "TRTLLM_DWDP_DENSE_COPY"
DENSE_PRIME_ENV = "TRTLLM_DWDP_DENSE_PRIME"
DENSE_PDL_BREAK_ENV = "TRTLLM_DWDP_DENSE_PDL_BREAK"

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")


def _bool_env(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = _optional_bool_env(environ, name, str(int(default)))
    return default if value is None else value


def _optional_bool_env(environ: Mapping[str, str], name: str, fallback: str) -> bool | None:
    """``name`` as a boolean; None if unset or empty, or with a warning if not a boolean."""
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    logger.warning(
        f"{LOG_PREFIX} {name}={raw!r} is not one of 1/true/yes/on or 0/false/no/off; "
        f"using {fallback}"
    )
    return None


def _int_env(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        value = -1
    if value < 0:
        logger.warning(
            f"{LOG_PREFIX} {name}={raw!r} is not a non-negative integer; using {default}"
        )
        return default
    return value


def parse_segments(value: str) -> tuple[SegmentKind, ...]:
    """Parse a ``TRTLLM_DWDP_DENSE_SEGMENTS`` value.

    Args:
        value: Comma-separated segment names, any case, surrounding blanks ignored, repeats
            allowed; e.g. ``"attn,moe"`` or ``"attn"``.

    Returns:
        The segments in canonical order (``attn`` before ``moe``).

    Raises:
        ValueError: An unknown name, an empty list, or a list without ``attn``: the attention
            segment drives the per-layer prefetch, so it is always enabled.
    """
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    known = {kind.value for kind in SEGMENT_ORDER}
    unknown = sorted(set(names) - known)
    if unknown or SegmentKind.ATTN.value not in names:
        raise ValueError(
            f"{DENSE_SEGMENTS_ENV}={value!r} must list attn and optionally moe "
            f"(comma-separated); unknown: {unknown}"
        )
    return tuple(kind for kind in SEGMENT_ORDER if kind.value in names)


def parse_copy_mode(value: str) -> CopyMode:
    """Parse a ``TRTLLM_DWDP_DENSE_COPY`` value.

    Args:
        value: ``batched``, ``per_slice`` or ``gather``, any case, surrounding blanks ignored.

    Returns:
        The transfer.

    Raises:
        ValueError: Any other value.
    """
    try:
        return CopyMode(value.strip().lower())
    except ValueError:
        raise ValueError(
            f"{DENSE_COPY_ENV}={value!r} must be one of batched, per_slice, gather"
        ) from None


def parse_prime_policy(value: str) -> PrimePolicy:
    """Parse a ``TRTLLM_DWDP_DENSE_PRIME`` value.

    Args:
        value: ``wrap`` or ``entry``, any case, surrounding blanks ignored.

    Returns:
        The priming policy.

    Raises:
        ValueError: Any other value.
    """
    try:
        return PrimePolicy(value.strip().lower())
    except ValueError:
        raise ValueError(f"{DENSE_PRIME_ENV}={value!r} must be one of wrap, entry") from None


def dense_dwdp_requested(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``TRTLLM_DWDP_DENSE`` asks for dense DWDP; reads no other variable.

    Args:
        environ: Environment to read; ``os.environ`` if None.
    """
    return _bool_env(os.environ if environ is None else environ, DENSE_ENV, False)


@dataclass(frozen=True)
class DenseConfig:
    """Dense DWDP switches, normally read once with :meth:`from_env`.

    Attributes:
        enabled: ``TRTLLM_DWDP_DENSE``.
        segments: ``TRTLLM_DWDP_DENSE_SEGMENTS`` in canonical order; always starts with ``ATTN``.
            Strings are accepted and converted.
        required: ``TRTLLM_DWDP_DENSE_REQUIRED``.
        min_bytes: ``TRTLLM_DWDP_DENSE_MIN_BYTES``.
        store_chunk_mb: ``TRTLLM_DWDP_DENSE_STORE_CHUNK_MB``.
        strict: ``TRTLLM_DWDP_DENSE_STRICT``.
        debug: ``TRTLLM_DWDP_DENSE_DEBUG``.
        copy: ``TRTLLM_DWDP_DENSE_COPY`` as set, blanks stripped and case kept; None when unset or
            blank.  Kept as given and validated only by :meth:`standalone_copy_mode`, so reading
            the switches never fails because of it: with expert DWDP it is ignored.
        prime: ``TRTLLM_DWDP_DENSE_PRIME`` as set, blanks stripped and case kept; None when unset
            or blank.  Kept as given and validated only by :meth:`prime_policy`, like ``copy``:
            outside graph mode it is ignored.
        pdl_break: ``TRTLLM_DWDP_DENSE_PDL_BREAK``; None when unset, blank or not a boolean, so
            that :meth:`pdl_break_enabled` follows the transfer and the protocol.
    """

    enabled: bool = False
    segments: tuple[SegmentKind, ...] = SEGMENT_ORDER
    required: bool = True
    min_bytes: int = DEFAULT_MIN_BYTES
    store_chunk_mb: int = 0
    strict: bool = False
    debug: int = 0
    copy: str | None = None
    prime: str | None = None
    pdl_break: bool | None = None

    def __post_init__(self) -> None:
        segments = _canonical_segments(self.segments)
        object.__setattr__(self, "segments", segments)
        for name in ("min_bytes", "store_chunk_mb", "debug"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"DenseConfig.{name} must be non-negative, got {getattr(self, name)}"
                )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> DenseConfig:
        """Read every ``TRTLLM_DWDP_DENSE*`` variable.

        ``TRTLLM_DWDP_DENSE_COPY`` and ``TRTLLM_DWDP_DENSE_PRIME`` are stored without validation
        (see ``copy`` and ``prime``), so they never make this raise.

        Args:
            environ: Environment to read; ``os.environ`` if None.

        Raises:
            ValueError: ``TRTLLM_DWDP_DENSE_SEGMENTS`` is invalid (see :func:`parse_segments`).
        """
        env = os.environ if environ is None else environ
        raw_segments = env.get(DENSE_SEGMENTS_ENV)
        if raw_segments is None or not raw_segments.strip():
            segments = SEGMENT_ORDER
        else:
            segments = parse_segments(raw_segments)
        copy = (env.get(DENSE_COPY_ENV) or "").strip()
        prime = (env.get(DENSE_PRIME_ENV) or "").strip()
        return cls(
            enabled=_bool_env(env, DENSE_ENV, False),
            segments=segments,
            required=_bool_env(env, DENSE_REQUIRED_ENV, True),
            min_bytes=_int_env(env, DENSE_MIN_BYTES_ENV, DEFAULT_MIN_BYTES),
            store_chunk_mb=_int_env(env, DENSE_STORE_CHUNK_MB_ENV, 0),
            strict=_bool_env(env, DENSE_STRICT_ENV, False),
            debug=_int_env(env, DENSE_DEBUG_ENV, 0),
            copy=copy or None,
            prime=prime or None,
            pdl_break=_optional_bool_env(
                env, DENSE_PDL_BREAK_ENV, "the default (on with the SM gather in graph mode)"
            ),
        )

    def standalone_copy_mode(self) -> CopyMode:
        """The transfer of standalone dense DWDP: ``copy`` parsed, ``BATCHED`` when it is None.

        Only standalone code calls it (the gate and ``DenseDwdpManager.setup``); with expert DWDP
        the dense copies follow the expert transfer whatever ``copy`` holds.

        Raises:
            ValueError: ``copy`` is not one of batched, per_slice, gather (see
                :func:`parse_copy_mode`).
        """
        if self.copy is None:
            return CopyMode.BATCHED
        return parse_copy_mode(self.copy)

    def prime_policy(self) -> PrimePolicy:
        """The priming policy of graph mode: ``prime`` parsed, ``WRAP`` when it is None.

        Only graph-mode code calls it (the gate and the dense manager).  Outside graph mode dense
        DWDP runs the ``LEGACY`` protocol, which primes at the start of every forward, as
        ``ENTRY`` does, whatever ``prime`` holds.

        Raises:
            ValueError: ``prime`` is not one of wrap, entry (see :func:`parse_prime_policy`).
        """
        if self.prime is None:
            return PrimePolicy.WRAP
        return parse_prime_policy(self.prime)

    def pdl_break_enabled(self, *, use_gather: bool, closed: bool) -> bool:
        """Whether H1 and H2 launch the PDL break: ``pdl_break``, else ``use_gather and closed``.

        Setup decides it for the transfer and the protocol it runs: on by default with the SM
        gather in graph mode, where the gather that becomes eligible at a hook must be dispatched
        before the next PDL-launched compute kernel (``types.PdlBreak``); off with the copy
        engines, which hold no SMs, and under the legacy protocol, where that stall was not
        observed.

        Args:
            use_gather: The dense copies use the SM fabric gather (``CopyMode.GATHER``).
            closed: The runtime runs the closed protocol (graph mode).
        """
        if self.pdl_break is not None:
            return self.pdl_break
        return use_gather and closed

    @property
    def slot_fill_byte(self) -> int:
        """Byte the slots are filled with at commit: 0, or ``DEBUG_FILL_BYTE`` (NaN) if debug."""
        return DEBUG_FILL_BYTE if self.debug >= 1 else 0

    @property
    def slot_watchdog(self) -> bool:
        """Whether the per-layer slot checksum watchdog runs (``TRTLLM_DWDP_DENSE_DEBUG >= 2``)."""
        return self.debug >= 2

    @property
    def slot_watchdog_assert(self) -> bool:
        """Whether H4 asserts on the device that the slot watchdog found no mismatch.

        ``TRTLLM_DWDP_DENSE_DEBUG >= 3``; it takes effect in graph mode, the only one with an H4.
        """
        return self.debug >= 3

    def digest_switches(self, num_ranks: int) -> DigestSwitches:
        """The switches the layout depends on, for a group of ``num_ranks`` ranks."""
        return DigestSwitches(
            segments=self.segments,
            min_bytes=self.min_bytes,
            store_chunk_mb=self.store_chunk_mb,
            num_ranks=num_ranks,
        )


def _canonical_segments(segments: Iterable[SegmentKind | str]) -> tuple[SegmentKind, ...]:
    try:
        kinds = {SegmentKind(kind) for kind in segments}
    except ValueError as exc:
        raise ValueError(f"DenseConfig.segments has an unknown segment: {exc}") from None
    if SegmentKind.ATTN not in kinds:
        raise ValueError("DenseConfig.segments must include the attention segment")
    return tuple(kind for kind in SEGMENT_ORDER if kind in kinds)
