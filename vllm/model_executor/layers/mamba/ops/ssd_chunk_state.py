# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py

# ruff: noqa: E501

import torch

from vllm.model_executor.layers.mamba.ops.triton_helpers import (
    fast_exp,
    launch_autotuned,
)
from vllm.triton_utils import tl, triton

from .mamba_ssm import softplus


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_H": 2}),
        triton.Config({"BLOCK_SIZE_H": 4}),
        triton.Config({"BLOCK_SIZE_H": 8}),
        triton.Config({"BLOCK_SIZE_H": 16}),
        triton.Config({"BLOCK_SIZE_H": 32}),
        triton.Config({"BLOCK_SIZE_H": 64}),
    ],
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimension
    nheads: tl.constexpr,
    chunk_size: tl.constexpr,
    dt_min: tl.constexpr,
    dt_max: tl.constexpr,
    # Strides
    stride_dt_seqlen: tl.int64,
    stride_dt_head: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_dt_out_head: tl.int64,
    stride_dt_out_chunk: tl.int64,
    stride_dt_out_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    # if dt is long, may cause problems, so use 64 bit
    # https://github.com/triton-lang/triton/issues/1058
    pid_c = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1)

    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

    dt_ptr += chunk_seqlen_start * stride_dt_seqlen
    dt_out_ptr += pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
    )
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)

    dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
    )
    tl.store(
        dt_out_ptrs,
        dt,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(
        dA_cs_ptrs,
        dA_cs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )


@triton.autotune(
    configs=[
        # Small headdim/dstate configs (hdim<=64, dstate<=128) - increased parallelism
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        # Low register pressure configs for large dstate (dstate=128)
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        # original configs for larger headdim/dstate values
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit(do_not_specialize_on_alignment=["final_state_indices_ptr"])
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    seq_idx_ptr,
    last_chunk_indices_ptr,
    initstates_ptr,
    final_states_ptr,
    final_state_indices_ptr,
    # Matrix dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_initstates_batch: tl.int64,
    stride_initstates_head: tl.int64,
    stride_initstates_hdim: tl.int64,
    stride_initstates_dstate: tl.constexpr,
    stride_final_states_slot: tl.int64,
    stride_final_states_head: tl.int64,
    stride_final_states_hdim: tl.int64,
    stride_final_states_dstate: tl.constexpr,
    stride_final_state_indices,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    FUSE_FINAL_STATES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    b_ptr += (
        chunk_seqlen_start * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        scale = fast_exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)

        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    states_ptr += pid_c * stride_states_chunk + pid_h * stride_states_head
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )

    if FUSE_FINAL_STATES:
        # A sequence that fits in one chunk has its final state right here:
        # final = exp(dA_cs_last) * initial + chunk state. Write it into the
        # sequence's cache slot and skip the fp32 per-chunk state, which
        # _state_passing_fwd would only read back to compute the same value.
        seq = tl.load(seq_idx_ptr + pid_c)
        seq_prev = tl.load(seq_idx_ptr + pid_c - 1, mask=pid_c >= 1, other=-1)
        seq_last_chunk = tl.load(last_chunk_indices_ptr + seq)
        if (seq != seq_prev) and (seq_last_chunk == pid_c):
            final = acc
            if HAS_INITSTATES:
                init = tl.load(
                    initstates_ptr
                    + seq * stride_initstates_batch
                    + pid_h * stride_initstates_head
                    + offs_m[:, None] * stride_initstates_hdim
                    + offs_n[None, :] * stride_initstates_dstate,
                    mask=c_mask,
                    other=0.0,
                ).to(tl.float32)
                final = fast_exp(dA_cs_last) * init + acc
            slot = tl.load(
                final_state_indices_ptr + seq * stride_final_state_indices
            ).to(tl.int64)
            final_states_ptrs = (
                final_states_ptr
                + slot * stride_final_states_slot
                + pid_h * stride_final_states_head
                + offs_m[:, None] * stride_final_states_hdim
                + offs_n[None, :] * stride_final_states_dstate
            )
            tl.store(final_states_ptrs, final, mask=c_mask)
        else:
            tl.store(states_ptrs, acc.to(states_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(states_ptrs, acc.to(states_ptr.dtype.element_ty), mask=c_mask)


def _chunk_cumsum_fwd(
    dt,
    A,
    chunk_size,
    cu_chunk_seqlens,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    grid_chunk_cs = lambda META: (nchunks, triton.cdiv(nheads, META["BLOCK_SIZE_H"]))
    launch_autotuned(
        _chunk_cumsum_fwd_kernel,
        grid_chunk_cs,
        dt_ptr=dt,
        A_ptr=A,
        dt_bias_ptr=dt_bias,
        dt_out_ptr=dt_out,
        dA_cumsum_ptr=dA_cumsum,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        nheads=nheads,
        chunk_size=chunk_size,
        dt_min=dt_limit[0],
        dt_max=dt_limit[1],
        stride_dt_seqlen=dt.stride(0),
        stride_dt_head=dt.stride(1),
        stride_A_head=A.stride(0),
        stride_dt_bias_head=dt_bias.stride(0) if dt_bias is not None else 0,
        stride_dt_out_head=dt_out.stride(0),
        stride_dt_out_chunk=dt_out.stride(1),
        stride_dt_out_csize=dt_out.stride(2),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        DT_SOFTPLUS=dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
    )
    return dA_cumsum, dt_out


def _chunk_state_fwd(
    B,
    x,
    dt,
    dA_cumsum,
    cu_chunk_seqlens,
    states=None,
    states_in_fp32=True,
    seq_idx=None,
    last_chunk_indices=None,
    initial_states=None,
    final_states=None,
    final_state_indices=None,
):
    """If `final_states` is given (with `seq_idx`, `last_chunk_indices` and
    `final_state_indices`), each sequence that fits in a single chunk gets its
    final state written to
    `final_states[final_state_indices[seq]]` here, and its entry in `states` is
    left unwritten. `_state_passing_fwd(..., skip_single_chunk_seqs=True)` then
    skips those sequences."""
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape

    if states is not None:
        assert states.shape == (nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype
        )

    fuse = final_states is not None
    if fuse:
        assert seq_idx is not None and final_state_indices is not None
        assert last_chunk_indices is not None
        assert final_states.shape[1:] == (nheads, headdim, dstate)
    init_strides = (
        initial_states.stride() if fuse and initial_states is not None else (0,) * 4
    )
    final_strides = final_states.stride() if fuse else (0,) * 4

    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )
    launch_autotuned(
        _chunk_state_fwd_kernel,
        grid,
        x_ptr=x,
        b_ptr=B,
        states_ptr=states,
        dt_ptr=dt,
        dA_cumsum_ptr=dA_cumsum,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        seq_idx_ptr=seq_idx if fuse else None,
        last_chunk_indices_ptr=last_chunk_indices if fuse else None,
        initstates_ptr=initial_states if fuse else None,
        final_states_ptr=final_states,
        final_state_indices_ptr=final_state_indices if fuse else None,
        hdim=headdim,
        dstate=dstate,
        chunk_size=chunk_size,
        nheads_ngroups_ratio=nheads // ngroups,
        stride_x_seqlen=x.stride(0),
        stride_x_head=x.stride(1),
        stride_x_hdim=x.stride(2),
        stride_b_seqlen=B.stride(0),
        stride_b_head=B.stride(1),
        stride_b_dstate=B.stride(2),
        stride_states_chunk=states.stride(0),
        stride_states_head=states.stride(1),
        stride_states_hdim=states.stride(2),
        stride_states_dstate=states.stride(3),
        stride_dt_head=dt.stride(0),
        stride_dt_chunk=dt.stride(1),
        stride_dt_csize=dt.stride(2),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        stride_initstates_batch=init_strides[0],
        stride_initstates_head=init_strides[1],
        stride_initstates_hdim=init_strides[2],
        stride_initstates_dstate=init_strides[3],
        stride_final_states_slot=final_strides[0],
        stride_final_states_head=final_strides[1],
        stride_final_states_hdim=final_strides[2],
        stride_final_states_dstate=final_strides[3],
        stride_final_state_indices=final_state_indices.stride(0) if fuse else 0,
        HAS_INITSTATES=fuse and initial_states is not None,
        FUSE_FINAL_STATES=fuse,
    )
    return states
