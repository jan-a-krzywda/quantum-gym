"""Backward-compat shim to analysis_processing.fid_data_io."""

from ML.analysis_processing.fid_data_io import (
    FidStackResult,
    build_stack_like_single_job,
    infer_num_qubits,
    load_memory_pickle,
    stack_to_vae_tensors,
)
