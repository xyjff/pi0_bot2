"""Offline-friendly norm stats computation using official OpenPI flow."""

import os
import pathlib
import sys

_MY_DIR = pathlib.Path("/share/0xyj/model3_openpi0.5/my_pi0_training")
_OPENPI_SRC = pathlib.Path("/share/0xyj/model3_openpi0.5/openpi-main/src")
for _p in [str(_MY_DIR), str(_OPENPI_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

# Ensure JAX doesn't compete for CUDA while loading stats.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_DEFAULT_REPO_ID = "local/right_arm_head_cam"
_DEFAULT_DATASET_PATH = (
    "/share/0xyj/model3_openpi0.5/my_pi0_training/hdf5_to_lerobot_data/lerobot_dataset_headcam_rightarm"
)
_OFFLINE_PATCHED = False
_local_root_registry: dict[str, str] = {}


def _apply_offline_patches() -> None:
    """Patch data loading so local datasets work in offline environments."""
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
            raise ValueError("Repo ID is not set. Cannot create dataset.")
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


class RemoveStrings(_transforms.DataTransformFn):
    """Remove string fields not needed for norm stats."""

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """Create a PyTorch dataloader for computing normalization stats."""
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    repo_id: str = _DEFAULT_REPO_ID,
    dataset_path: str = _DEFAULT_DATASET_PATH,
    action_horizon: int = 50,
    action_dim: int = 8,
    batch_size: int = 32,
    num_workers: int = 4,
    max_frames: int | None = None,
    output_dir: str = "/share/0xyj/model3_openpi0.5/openpi-main/assets/pi0_right_arm_head_cam",
):
    """Compute normalization statistics fully offline from local dataset."""
    _apply_offline_patches()
    model_config = pi0_config.Pi0Config(
        # Keep this aligned with my_config_right_arm_head_cam.py and pi0_base weights.
        pi05=False,
        action_dim=action_dim,
        action_horizon=action_horizon,
        discrete_state_input=False,
    )

    data_transforms = _transforms.Group(
        inputs=[_transforms.DeltaActions(_transforms.make_bool_mask(7, -1))],
        outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1))],
    )
    data_config = _config.DataConfig(
        repo_id=repo_id,
        asset_id=repo_id,
        norm_stats=None,
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.cam_high"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        ),
        data_transforms=data_transforms,
        model_transforms=_transforms.Group(inputs=[], outputs=[]),
        action_sequence_keys=("action",),
        prompt_from_task=False,
    )

    # Register local root so patched loader never relies on HF Hub.
    _local_root_registry[data_config.repo_id] = dataset_path

    assets_dir = pathlib.Path(output_dir)

    print(f"Dataset: {data_config.repo_id}")
    print(f"Dataset root: {dataset_path}")
    print(f"Assets dir: {assets_dir}")
    print(f"Action horizon: {action_horizon}")
    print(f"Action dim: {action_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")

    data_loader, num_batches = create_torch_dataloader(
        data_config,
        model_config.action_horizon,
        batch_size,
        model_config,
        num_workers,
        max_frames,
    )

    print(f"Total batches to process: {num_batches}")

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = assets_dir / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done!")

    # Print the computed stats
    print("\nComputed normalization statistics:")
    for key, stat in norm_stats.items():
        print(f"\n{key}:")
        print(f"  mean: {stat.mean}")
        print(f"  std: {stat.std}")


if __name__ == "__main__":
    tyro.cli(main)
