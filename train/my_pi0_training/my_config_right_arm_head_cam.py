"""Custom Pi0.5 config for right-arm + head camera training."""

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

OPENPI_DIR = Path("/share/0xyj/model3_openpi0.5/openpi-main")
DATASET_PATH = "/share/0xyj/model3_openpi0.5/my_pi0_training/hdf5_to_lerobot_data/lerobot_dataset_headcam_rightarm125"
WEIGHT_PATH = "/share/0xyj/model3_openpi0.5/pi0_base"
REPO_ID = "local/right_arm_head_cam"


def _to_numpy_hwc_uint8(img) -> np.ndarray:
    """LeRobot video pipeline may yield torch.Tensor CHW float; OpenPI resize expects HWC uint8."""
    try:
        import torch

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
    except ImportError:
        pass
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


@dataclasses.dataclass(frozen=True)
class MyRobotInputs(_transforms.DataTransformFn):
    """Map dataset fields to OpenPI canonical fields."""

    include_gripper_state_input: bool = False

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        state = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        if state.ndim > 1:
            state = state[..., :8]
        else:
            state = state[:8]
        if not self.include_gripper_state_input and state.shape[-1] > 7:
            state = state.copy()
            state[..., 7] = 0.0
        if actions.ndim >= 2:
            actions = actions[..., :8]

        cam = _to_numpy_hwc_uint8(images["cam_high"])
        placeholder = np.zeros_like(cam)
        image = {
            "base_0_rgb": cam,
            "left_wrist_0_rgb": placeholder,
            "right_wrist_0_rgb": placeholder,
        }
        image_mask = {
            "base_0_rgb": True,
            "left_wrist_0_rgb": False,
            "right_wrist_0_rgb": False,
        }

        out = {
            "state": state,
            "actions": actions,
            "image": image,
            "image_mask": image_mask,
        }
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class MyRobotOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": data["actions"][..., :8]}


@dataclasses.dataclass(frozen=True)
class MyDataConfig(_config.DataConfigFactory):
    repo_id: str = REPO_ID
    local_root: str | None = DATASET_PATH
    use_delta_joint_actions: bool = True
    include_gripper_state_input: bool = False
    default_prompt: str | None = "right arm pick and place task"
    prompt_from_task: bool = False
    action_sequence_keys: Sequence[str] = ("action",)

    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.cam_high"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    def create(self, assets_dirs, model_config) -> _config.DataConfig:
        data_transforms = _transforms.Group(
            inputs=[MyRobotInputs(include_gripper_state_input=self.include_gripper_state_input)],
            outputs=[MyRobotOutputs()],
        )
        if self.use_delta_joint_actions:
            mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(mask)],
                outputs=[_transforms.AbsoluteActions(mask)],
            )

        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=self.prompt_from_task,
        )


def get_my_configs() -> list[_config.TrainConfig]:
    return [
        _config.TrainConfig(
            name="pi0_right_arm_head_cam",
            checkpoint_base_dir=str(OPENPI_DIR / "checkpoints"),
            assets_base_dir=str(OPENPI_DIR / "assets"),
            model=pi0_config.Pi0Config(
                # Must match pytorch_weight_path: pi0_base is π0, not π0.5 (different expert MLP / norms).
                pi05=False,
                action_dim=32,
                action_horizon=50,
                dtype="bfloat16",
                discrete_state_input=False,
            ),
            data=MyDataConfig(),
            pytorch_weight_path=WEIGHT_PATH,
            num_train_steps=50_000,
            batch_size=4,
            log_interval=50,
            save_interval=2000,
            num_workers=0,  # must be 0 unless train_right_arm_head_cam patches workers (HF Column bug)
            keep_period=0,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=5e-5,
                decay_steps=50_000,
                decay_lr=5e-6,
            ),
            ema_decay=0.999,
            wandb_enabled=False,
            overwrite=False,
        )
    ]
