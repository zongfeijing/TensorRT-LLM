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
"""Dense DWDP manager: the owner of the committed dense DWDP runtime, in both modes.

The model's dense hooks reach dense DWDP only through the manager that
:func:`get_global_dense_dwdp_manager` returns when the model is built: H1-H3 read
:attr:`DenseDwdpManager.dense`, H0 calls :meth:`DenseDwdpManager.ensure_primed`.  The manager
does not depend on ``DwdpManager``.

* Attached (``DenseMode.ATTACHED``, with expert DWDP).  ``DwdpManager`` creates and registers the
  manager, runs ``setup_dense_dwdp`` over the DWDP group before the expert DWDP setup allocates its
  prefetch ring, on a transfer whose copy stream the expert weight manager then adopts, and hands
  the committed runtime over with :meth:`DenseDwdpManager.attach`, together with the expert
  weight manager.  Priming then interleaves the dense and the expert copies on the shared copy
  stream, ``L(first), E(first), L(second), E(second)``, and H0 also re-records the expert consume
  events after an aborted forward.  ``DwdpManager.cleanup`` releases dense before the expert
  teardown; this manager never releases the weight manager.
* Standalone (``DenseMode.STANDALONE``, an attention-DP instance with ``tp_size > 1`` and no
  expert DWDP).  The executor creator creates and registers the manager before the model is
  built and calls :meth:`DenseDwdpManager.setup` once the weights are loaded.  Setup runs over an
  MPI sub-communicator of the attention-DP group (:func:`standalone_group`), which it frees when
  it ends, with a :class:`StandaloneTransfer`: the manager's own copy stream and
  ``TRTLLM_DWDP_DENSE_COPY``.  Priming enqueues ``L(first), L(second)``.  The final executor
  exits the manager at shutdown.

The module imports no executor or model code.
"""

from __future__ import annotations

import traceback
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from tensorrt_llm._utils import mpi_comm, mpi_disabled
from tensorrt_llm.logger import logger

from .. import fabric_gather
from .config import DenseConfig
from .plan import select_copy_mode
from .setup import (
    DenseTransfer,
    GroupComm,
    LauncherFactory,
    StoreFactory,
    dense_dwdp_active,
    dense_roots_available,
    refuse_dense_dwdp_without_roots,
    setup_dense_dwdp,
)
from .types import LOG_PREFIX, CopyMode, DenseMode, StreamLike

if TYPE_CHECKING:
    from mpi4py import MPI

    from tensorrt_llm.mapping import Mapping

    from .runtime import DenseDWDP

_global_dense_dwdp_manager: DenseDwdpManager | None = None

#: How :meth:`DenseDwdpManager.ensure_primed` names itself when ``DenseDWDP.require_eager``
#: refuses it.
H0_HOOK = "DenseDwdpManager.ensure_primed (H0)"

#: The ``Create_group`` tag of :func:`standalone_group` (``"DW"``).  MPI pairs the
#: ``MPI_Comm_create_group`` calls on one parent communicator by their tag, and ``MPIDist.tp_comm``
#: creates its communicator over the same group with the default tag 0, so with a tag of its own
#: the dense group never pairs with a peer's ``tp_comm``: a rank whose ``TRTLLM_DWDP_DENSE``
#: differs leaves its peers waiting in ``Create_group``, not sharing a communicator with them.  At
#: most 32767, the smallest ``MPI_TAG_UB`` the MPI standard allows.
STANDALONE_GROUP_TAG = 0x4457


def set_global_dense_dwdp_manager(manager: DenseDwdpManager | None) -> None:
    """Register ``manager`` as the process's dense DWDP manager; None unregisters."""
    global _global_dense_dwdp_manager
    _global_dense_dwdp_manager = manager


def get_global_dense_dwdp_manager() -> DenseDwdpManager | None:
    """The registered dense DWDP manager, or None; the model's hooks capture it when built."""
    return _global_dense_dwdp_manager


class ExpertPrefetcher(Protocol):
    """The ``DWDPWeightManager`` calls attached priming and recovery make."""

    def first_moe_layer(self) -> int:
        """The first MoE layer."""
        ...

    def next_moe_layer(self, layer_idx: int) -> int | None:
        """The MoE layer after ``layer_idx``, or None after the last one."""
        ...

    def prefetch_layer(self, layer_idx: int) -> None:
        """Enqueue the expert copy ``E(layer_idx)`` on the expert DWDP copy stream."""
        ...

    def record_compute(self, layer_idx: int) -> None:
        """Record the consume event of ``layer_idx``'s expert slot on the current stream."""
        ...


def new_copy_stream(device_id: int) -> torch.cuda.Stream:
    """A new standalone dense copy stream on ``cuda:<device_id>``.

    Default priority, like the expert DWDP copy stream.  The test seam of the standalone
    transfer: CPU tests monkeypatch ``manager.new_copy_stream``.
    """
    return torch.cuda.Stream(device=torch.device("cuda", device_id))


class StandaloneTransfer:
    """The standalone dense transfer (``setup.DenseTransfer``): its own copy stream and mode.

    Only dense copies run on the stream, so its FIFO is ``L(first), L(second), L(3), ...``, the
    deadline order.  The stream is created by ``stream_factory`` at the first :attr:`copy_stream`
    read, which setup makes in P2 inside the phase's local work: constructing the transfer
    touches no CUDA, and a failure to create the stream becomes this rank's failed P2 report.

    Args:
        mode: The transfer, normally ``DenseConfig.standalone_copy_mode()``.
        stream_factory: Creates the copy stream; called at the first :attr:`copy_stream` read.
    """

    def __init__(self, mode: CopyMode, stream_factory: Callable[[], StreamLike]) -> None:
        self._mode = CopyMode(mode)
        self._stream_factory = stream_factory
        self._stream: StreamLike | None = None

    @classmethod
    def on_device(cls, mode: CopyMode, device_id: int) -> StandaloneTransfer:
        """A transfer whose copy stream is ``new_copy_stream(device_id)``.

        ``new_copy_stream`` is looked up when the stream is created, so a monkeypatched one
        applies.
        """
        return cls(mode, lambda: new_copy_stream(device_id))

    @property
    def mode(self) -> CopyMode:
        """The transfer."""
        return self._mode

    @property
    def copy_stream(self) -> StreamLike:
        """The copy stream: created on the first read, the same object on every later one."""
        if self._stream is None:
            self._stream = self._stream_factory()
        return self._stream

    @property
    def use_gather(self) -> bool:
        """Whether the copies use the SM fabric gather (``CopyMode.GATHER``)."""
        return self._mode is CopyMode.GATHER

    @property
    def use_batched_prefetch(self) -> bool:
        """Whether the copies use one batched copy-engine copy per layer (``CopyMode.BATCHED``)."""
        return self._mode is CopyMode.BATCHED


def standalone_group(mapping: Mapping) -> MPI.Intracomm:
    """The standalone dense group: an MPI sub-communicator over ``mapping.tp_group``.

    Built the way ``MPIDist.tp_comm`` builds its communicator, but a communicator of its own,
    created with :data:`STANDALONE_GROUP_TAG`: the dense setup collectives never interleave with
    the executor's tp collectives, and the caller frees it when setup ends.  Its rank and size are
    ``mapping.tp_rank`` and ``mapping.tp_size``, and it has the ``GroupComm`` calls setup makes
    plus ``Free``.  Creating it is a collective over the tp group: every rank of the group must
    call it.

    Raises:
        RuntimeError: MPI is disabled (``TLLM_DISABLE_MPI=1``); the gate rejects such a launch
            earlier.
    """
    if mpi_disabled():
        raise RuntimeError(
            "standalone dense DWDP needs the MPI backend: its group is an MPI sub-communicator "
            "over the attention-DP group, but TLLM_DISABLE_MPI=1"
        )
    comm = mpi_comm()
    return comm.Create_group(comm.group.Incl(mapping.tp_group), tag=STANDALONE_GROUP_TAG)


class DenseDwdpManager:
    """Owner of the committed dense DWDP runtime in either mode; see the module docstring.

    Neither construction nor registration (``__enter__``) runs a collective or touches CUDA, so
    the manager exists before the model is built, and the model's hooks capture it then.  The
    hooks are no-ops while :attr:`dense` is None: before :meth:`attach` or :meth:`setup` commits a
    runtime, when dense DWDP is off or was abandoned, and after :meth:`cleanup`.

    Args:
        mode: ``DenseMode.ATTACHED`` (with expert DWDP; ``DwdpManager`` sets dense DWDP up and
            attaches the runtime) or ``DenseMode.STANDALONE`` (:meth:`setup` sets it up).
        mapping: Standalone only, and required there: the instance's mapping, with
            ``enable_attention_dp`` and ``tp_size > 1``; the dense group is its tp group.

    Raises:
        ValueError: A standalone manager without such a mapping, or an attached one with a
            mapping.
    """

    def __init__(self, mode: DenseMode, *, mapping: Mapping | None = None) -> None:
        mode = DenseMode(mode)
        if mode is DenseMode.STANDALONE:
            if mapping is None or not mapping.enable_attention_dp or mapping.tp_size <= 1:
                raise ValueError(
                    "a standalone DenseDwdpManager needs the mapping of an attention-DP "
                    "instance with tp_size > 1 (enable_attention_dp=True), got "
                    f"{_describe_mapping(mapping)}"
                )
        elif mapping is not None:
            raise ValueError(
                "an attached DenseDwdpManager takes no mapping: with expert DWDP, DwdpManager "
                f"sets dense DWDP up over the DWDP group; got {_describe_mapping(mapping)}"
            )
        self._mode = mode
        self._mapping = mapping
        # The committed runtime; None before attach() or setup() commits one, when dense DWDP
        # is off or was abandoned, and after cleanup().
        self._dense: DenseDWDP | None = None
        # Attached only: the expert weight manager, for the interleaved priming.
        self._expert: ExpertPrefetcher | None = None
        # Standalone only: the model setup() committed the runtime for.
        self._model_ref: weakref.ReferenceType[nn.Module] | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Global registration
    # ------------------------------------------------------------------

    def __enter__(self) -> DenseDwdpManager:
        """Register this manager globally; refused while any dense DWDP manager is registered."""
        if get_global_dense_dwdp_manager() is not None:
            raise RuntimeError("DenseDwdpManager already registered globally")
        set_global_dense_dwdp_manager(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """:meth:`cleanup`, then unregister this manager (only if it is the registered one)."""
        try:
            self.cleanup()
        finally:
            if get_global_dense_dwdp_manager() is self:
                set_global_dense_dwdp_manager(None)
        return False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def mode(self) -> DenseMode:
        """Attached (with expert DWDP) or standalone."""
        return self._mode

    @property
    def dense(self) -> DenseDWDP | None:
        """The committed dense DWDP runtime, or None when dense DWDP is not active.

        None before :meth:`attach` or :meth:`setup` commits a runtime, when dense DWDP is off or
        was abandoned (``TRTLLM_DWDP_DENSE_REQUIRED=0``), and after :meth:`cleanup`.  The model's
        layer hooks run only while it is not None.
        """
        return self._dense

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def attach(self, runtime: DenseDWDP, *, expert: ExpertPrefetcher) -> None:
        """Take over a runtime ``DwdpManager.setup`` committed (attached mode).

        Keeps ``expert``, the expert weight manager, for the interleaved priming and recovery;
        the manager never releases or tears it down.  Logs nothing.

        Raises:
            RuntimeError: The manager is standalone or closed, or a runtime is already attached.
            ValueError: ``runtime`` is None.
        """
        if self._mode is not DenseMode.ATTACHED:
            raise RuntimeError(
                "DenseDwdpManager.attach() is for dense DWDP with expert DWDP; a standalone "
                "manager sets dense DWDP up itself"
            )
        if self._closed:
            raise RuntimeError("DenseDwdpManager.attach() called after cleanup()")
        if self._dense is not None:
            raise RuntimeError(
                "DenseDwdpManager.attach(): a dense DWDP runtime is already attached"
            )
        if runtime is None:
            raise ValueError("DenseDwdpManager.attach() needs the committed runtime, got None")
        self._dense = runtime
        self._expert = expert

    def setup(
        self,
        model: nn.Module,
        *,
        comm: GroupComm | None = None,
        transfer: DenseTransfer | None = None,
        config: DenseConfig | None = None,
        device_id: int | None = None,
        store_factory: StoreFactory | None = None,
        launcher_factory: LauncherFactory | None = None,
    ) -> DenseDWDP | None:
        """Set standalone dense DWDP up on ``model`` over the attention-DP group.

        Called once from ``py_executor_creator`` after the weights are loaded and before the
        final KV cache is sized, so that the KV cache gets the memory setup frees, on every rank
        of the instance.  In order:

        1. Refused after :meth:`cleanup`.
        2. A runtime is already committed: for the same model it is returned without any
           collective; another model is refused.
        3. A model whose dense weights a released runtime moved is refused: its modules hold
           zero-size tensors.
        4. A model without dense roots: ``refuse_dense_dwdp_without_roots`` (it raises unless
           ``TRTLLM_DWDP_DENSE_REQUIRED`` is off), then None, with no group and no collective.
        5. The switches (``config``, or ``DenseConfig.from_env()``) and the device.
        6. The transfer (``transfer``, or :meth:`StandaloneTransfer.on_device` with
           ``config.standalone_copy_mode()``; no stream yet).
        7. One INFO line, so that ranks waiting for a peer in the next step show where they wait.
        8. The group: ``comm``, which the caller owns, or a new :func:`standalone_group`, the
           first collective of setup.
        9. ``setup_dense_dwdp`` as rank ``mapping.tp_rank`` of ``mapping.tp_size``, without an
           expert layer list.  A committed runtime is stored at once; with a gather transfer,
           ``fabric_gather.apply_compute_side_defaults()`` follows, still before any GEMM is
           tuned.  A setup that commits nothing leaves the process environment as it was.
        10. A group this call created is freed, whatever happened; a failing ``Free`` is logged,
            never raised: the peers would otherwise carry on into the instance's next
            collective, and the group carries no traffic after setup.

        Steps 1-7 are host-only and depend only on the model code and the job environment, which
        every rank shares, so every rank reaches the group or raises before it alike.  Every
        fallible local step after it, the creation of the copy stream included, runs inside
        ``setup_dense_dwdp``'s phases and becomes a report, so every rank reaches the same
        verdict.

        Args:
            model: The loaded model.
            comm: The dense group; a new :func:`standalone_group` if None.  Tests inject one.
            transfer: Copy stream and mode; ``StandaloneTransfer.on_device`` if None.
            config: Switches; ``DenseConfig.from_env()`` if None.
            device_id: CUDA device of this rank; ``torch.cuda.current_device()`` if None.
            store_factory: Passed to ``setup_dense_dwdp``.
            launcher_factory: Passed to ``setup_dense_dwdp``.

        Returns:
            The committed runtime, now :attr:`dense`, or None when dense DWDP is off or was
            abandoned with ``TRTLLM_DWDP_DENSE_REQUIRED`` off.

        Raises:
            RuntimeError: An attached manager; called after :meth:`cleanup`; a runtime is
                committed for another model; or the model's dense weights were already moved.
                Each is raised before any work.
            ValueError: ``TRTLLM_DWDP_DENSE_SEGMENTS`` or ``TRTLLM_DWDP_DENSE_COPY`` is invalid
                (the gate rejects both earlier).
            DenseSetupError: Setup failed or was abandoned with ``TRTLLM_DWDP_DENSE_REQUIRED``
                on, or the model has no dense roots and it is on; raised after the group is
                freed.
        """
        if self._mode is not DenseMode.STANDALONE:
            raise RuntimeError(
                "DenseDwdpManager.setup() is for standalone dense DWDP; with expert DWDP, "
                "DwdpManager.setup sets dense DWDP up and attaches the runtime"
            )
        if self._closed:
            raise RuntimeError("DenseDwdpManager.setup() called after cleanup()")
        if self._dense is not None:
            if self._model_ref is None or self._model_ref() is not model:
                raise RuntimeError(
                    "DenseDwdpManager.setup(): dense DWDP is already committed for another "
                    "model; a manager serves one model"
                )
            return self._dense
        if dense_dwdp_active(model):
            raise RuntimeError(
                "DenseDwdpManager.setup(): dense DWDP already moved this model's dense weights "
                "and its runtime was released, so the modules hold zero-size tensors; load the "
                "model again to set it up"
            )
        if not dense_roots_available(model):
            refuse_dense_dwdp_without_roots(model)
            return None
        if config is None:
            config = DenseConfig.from_env()
        if device_id is None:
            device_id = torch.cuda.current_device()
        if transfer is None:
            transfer = StandaloneTransfer.on_device(config.standalone_copy_mode(), device_id)
        mapping = self._mapping
        copy_mode = select_copy_mode(
            use_gather=transfer.use_gather, use_batched=transfer.use_batched_prefetch
        )
        logger.info(
            f"{LOG_PREFIX} standalone setup: rank {mapping.tp_rank} of the {mapping.tp_size}-rank "
            f"attention-DP group, {copy_mode.value} copies on the dense copy stream"
        )
        group = None
        if comm is None:
            group = comm = standalone_group(mapping)
        try:
            dense = setup_dense_dwdp(
                model,
                transfer,
                comm,
                dwdp_rank=mapping.tp_rank,
                dwdp_size=mapping.tp_size,
                device_id=device_id,
                moe_layers=None,
                config=config,
                store_factory=store_factory,
                launcher_factory=launcher_factory,
            )
            if dense is not None:
                # Stored before the group is freed: whatever Free does, the manager owns the
                # runtime, so cleanup() (or the creator's rollback) releases it.
                self._dense = dense
                self._model_ref = weakref.ref(model)
                if transfer.use_gather:
                    fabric_gather.apply_compute_side_defaults()
        finally:
            if group is not None:
                _free_group(group)
        return dense

    # ------------------------------------------------------------------
    # Priming (H0 and the executor loop)
    # ------------------------------------------------------------------

    def prefetch_first_layers(self) -> None:
        """Executor-loop priming of the first two covered layers before a forward.

        Enqueues ``L(first), L(second)`` (attached: ``L(first), E(first), L(second),
        E(second)``) on the copy stream, then marks the slots primed.  A no-op without a
        runtime, when the slots are already primed, or when a forward is still open: the
        previous forward aborted, and the model's :meth:`ensure_primed` recovers on the execution
        stream.  No read window is open while no forward is, so the stream this runs on needs no
        fence.
        """
        dense = self._dense
        if dense is None or dense.forward_open or dense.primed:
            return
        self._prime_all()

    def ensure_primed(self) -> None:
        """Model-side priming (H0): called at the model's forward entry while ``dense`` is set.

        Runs on the execution stream, so it also covers the forwards the executor loop does not
        prime: warmup, autotuning, KV-cache estimation and every forward of the overlap loop.  It
        refuses a CUDA graph capture and multi-stream execution (``DenseDWDP.require_eager``)
        before it records or enqueues anything, so a refused forward leaves the runtime as it
        found it.  Then, if the previous forward aborted between its first and last covered
        layer, it records a WAR barrier on this stream (every dense consume event; attached, then
        the consume events of both expert slots), so the next copies wait for every read the
        aborted forward enqueued.  Then it primes unless already primed.

        Raises:
            RuntimeError: Dense DWDP is not active.
            DenseProtocolError: The current stream is capturing a CUDA graph, or multi-stream
                execution is on.
        """
        dense = self._dense
        if dense is None:
            raise RuntimeError(
                "DenseDwdpManager.ensure_primed() needs an active dense DWDP runtime"
            )
        dense.require_eager(H0_HOOK)
        if dense.forward_open:
            dense.war_barrier()
            expert = self._expert
            if expert is not None:
                first = expert.first_moe_layer()
                expert.record_compute(first)
                second = expert.next_moe_layer(first)
                if second is not None:
                    expert.record_compute(second)
        if not dense.primed:
            self._prime_all()

    def _prime_all(self) -> None:
        """Enqueue the first two layers in deadline order, then mark the dense slots primed.

        Attached, ``L(l)`` goes before ``E(l)`` on the shared copy stream, which runs them in
        enqueue order, because layer ``l`` reads ``L(l)`` (attention) before ``E(l)`` (experts).
        The covered layers are then the expert MoE layers, so the expert weight manager names
        them.
        """
        dense, expert = self._dense, self._expert
        if expert is not None:
            first = expert.first_moe_layer()
            for layer in (first, expert.next_moe_layer(first)):
                if layer is None:
                    break
                dense.prefetch(layer)
                expert.prefetch_layer(layer)
        else:
            for layer in dense.layers[:2]:
                dense.prefetch(layer)
        dense.mark_primed()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release the runtime; idempotent.

        The manager is closed and its state cleared first, then the runtime is released: it
        synchronizes the device, repoints the model's dense weights to zero-size tensors and
        frees the staging slots and the row store.  The model keeps its ``_dense_dwdp_active``
        mark, so weight reloads stay refused.  A release error propagates with the state already
        cleared.  The expert weight manager is never released here.
        """
        self._closed = True
        dense, self._dense = self._dense, None
        self._expert = None
        self._model_ref = None
        if dense is not None:
            dense.release()


def _free_group(group: MPI.Intracomm) -> None:
    """Free the standalone group setup created; a failure is logged, never raised."""
    try:
        group.Free()
    except Exception:  # the peers carry on; a failed free leaks only a communicator handle
        logger.error(
            f"{LOG_PREFIX} freeing the standalone dense group failed\n{traceback.format_exc()}"
        )


def _describe_mapping(mapping: Mapping | None) -> str:
    if mapping is None:
        return "no mapping"
    return (
        f"a mapping with enable_attention_dp={mapping.enable_attention_dp} and "
        f"tp_size={mapping.tp_size}"
    )
