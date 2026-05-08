import sys
import pytest

collect_ignore_glob = []

try:
    import torch  # noqa: F401
except ImportError:
    collect_ignore_glob += [
        "test_checkpointing.py",
        "test_cluster_backends.py",
        "test_cluster_head.py",
        "test_mask_aware_discovery.py",
        "test_pipeline_reports.py",
        "test_prepare_unimodal_dataset.py",
    ]
