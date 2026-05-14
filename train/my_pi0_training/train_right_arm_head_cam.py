#!/usr/bin/env python3
"""Offline training launcher for pi0_right_arm_head_cam."""

"""
cd /share/0xyj/model3_openpi0.5/my_pi0_training && /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/python train_right_arm_head_cam.py pi0_right_arm_head_cam --exp-name train_125_jiazhuastate_true 2>&1 | tee train.log &
echo "PID: $!"
  
  
可以用 tail -f train.log 监控进度
  
  """
  
  
  
  
import json
import os
import pathlib
import sys

_MY_DIR = pathlib.Path("/share/0xyj/model3_openpi0.5/my_pi0_training")
_OPENPI_SRC = pathlib.Path("/share/0xyj/model3_openpi0.5/openpi-main/src")
_OPENPI_SCRIPTS = pathlib.Path("/share/0xyj/model3_openpi0.5/openpi-main/scripts")
for _p in [str(_MY_DIR), str(_OPENPI_SRC), str(_OPENPI_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
from my_config_right_arm_head_cam import get_my_configs


_OFFLINE_PATCHED = False
_local_root_registry: dict[str, str] = {}


def apply_patches() -> None:
    global _OFFLINE_PATCHED
    if _OFFLINE_PATCHED:
        return
    _OFFLINE_PATCHED = True

    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets import utils as ds_utils

    def patched_get_safe_version(repo_id: str, version: str | None) -> str:
        del repo_id, version
        return "main"

    ds_utils.get_safe_version = patched_get_safe_version
    if hasattr(lerobot_dataset, "get_safe_version"):
        lerobot_dataset.get_safe_version = patched_get_safe_version

    def _class_get_video_file_path(self_meta, ep_index: int, vid_key: str):
        cs = self_meta.info.get("chunks_size", 1000)
        ep_chunk = ep_index // cs
        file_index = ep_index % cs
        return pathlib.Path(f"videos/{vid_key}/chunk-{ep_chunk:03d}/file-{file_index:03d}.mp4")

    def _class_get_data_file_path(self_meta, ep_index: int):
        cs = self_meta.info.get("chunks_size", 1000)
        ep_chunk = ep_index // cs
        file_index = ep_index % cs
        return pathlib.Path(f"data/chunk-{ep_chunk:03d}/file-{file_index:03d}.parquet")

    lerobot_dataset.LeRobotDatasetMetadata.get_video_file_path = _class_get_video_file_path
    lerobot_dataset.LeRobotDatasetMetadata.get_data_file_path = _class_get_data_file_path

    def patched_query(self, query_indices):
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            selected = self.hf_dataset.select(q_idx)[key]
            if hasattr(selected, "__class__") and selected.__class__.__name__ == "Column":
                result[key] = np.array(selected)
            else:
                result[key] = np.stack(selected)
        return result

    lerobot_dataset.LeRobotDataset._query_hf_dataset = patched_query

    def patched_create_torch_dataset(data_config, action_horizon, model_config):
        repo_id = data_config.repo_id
        if repo_id is None:
            raise ValueError("Repo ID is not set.")
        if repo_id == "fake":
            return _data_loader.FakeDataset(model_config, num_samples=1024)

        root = _local_root_registry.get(repo_id)
        root = pathlib.Path(root) if root else None
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
            video_backend="pyav",
        )
        if data_config.prompt_from_task:
            dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
        return dataset

    _data_loader.create_torch_dataset = patched_create_torch_dataset


def register_configs() -> None:
    for cfg in get_my_configs():
        if cfg.name not in _config._CONFIGS_DICT:
            _config._CONFIGS.append(cfg)
            _config._CONFIGS_DICT[cfg.name] = cfg

    # Register local dataset roots after config registration.
    for cfg in get_my_configs():
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        if hasattr(cfg.data, "local_root") and getattr(cfg.data, "local_root"):
            _local_root_registry[dc.repo_id] = cfg.data.local_root


def _patch_dataloader_workers_for_spawn() -> None:
    """Spawn workers get a fresh interpreter; re-apply LeRobot patches there."""
    os.environ["OPENPI_LEROOT_JSON"] = json.dumps(dict(_local_root_registry))
    _orig_worker_init = _data_loader._worker_init_fn

    def _worker_init_fn(worker_id: int) -> None:
        _orig_worker_init(worker_id)
        global _OFFLINE_PATCHED
        _OFFLINE_PATCHED = False
        for k, v in json.loads(os.environ["OPENPI_LEROOT_JSON"]).items():
            _local_root_registry[k] = v
        apply_patches()

    _data_loader._worker_init_fn = _worker_init_fn


def main():
    apply_patches()
    register_configs()
    _patch_dataloader_workers_for_spawn()
    import train_pytorch as official_train_pytorch

    official_train_pytorch.init_logging()
    config = _config.cli()
    official_train_pytorch.train_loop(config)


if __name__ == "__main__":
    main()
