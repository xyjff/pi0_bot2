#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPI 右臂8维离线推理脚本 - 仅使用头部相机，可选校正

功能：
1. 仅从数据集读取头部相机图像（cam_high）
2. 从 parquet 文件读取右臂8维关节状态和动作
3. 使用 OpenPI 右臂8维模型进行推理
4. 可选应用校正值来补偿系统偏差

使用方式：
    cd /home/dmh/xyj_zhipingfang/model3_openpi0.5/my_bot2_deployment/offline_inference
    source /home/dmh/xyj_zhipingfang/model3_openpi0.5/openpi/.venv/bin/activate

    # 不校正
    python run_right_arm_inference_headcam_only.py \
        --checkpoint-dir /home/dmh/xyj_zhipingfang/model3_openpi0.5/pi0_checkpoint_xin129tiao_headcam/7000 \
        --dataset-dir /home/dmh/xyj_zhipingfang/model3_openpi0.5/lerobot_dataset \
        --episode-index 1 \
        --no-correction \
        --output-csv ./results_headcam_ep1_raw.csv

    # 使用校正（推荐）
    python run_right_arm_inference_headcam_only.py \
        --checkpoint-dir /home/dmh/xyj_zhipingfang/model3_openpi0.5/pi0_checkpoint_xin129tiao_headcam/7000 \
        --dataset-dir /home/dmh/xyj_zhipingfang/model3_openpi0.5/lerobot_dataset \
        --episode-index 1 \
        --use-correction \
        --output-csv ./results_headcam_ep1_corrected.csv
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import pathlib
import sys
import time
from typing import Optional

# ════════════════════════════════════════════════════════════════
# 【关键】设置 Python 路径
# ════════════════════════════════════════════════════════════════
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_OPENPI_ROOT = _SCRIPT_DIR.parent.parent / "openpi"
sys.path.insert(0, str(_OPENPI_ROOT / "src"))

# 设置 OPENPI_DATA_HOME
if "OPENPI_DATA_HOME" not in os.environ:
    os.environ["OPENPI_DATA_HOME"] = str(_OPENPI_ROOT / ".openpi_cache")

import cv2
import numpy as np
import pyarrow.parquet as pq

# ════════════════════════════════════════════════════════════════
# 【关键】禁用 Triton 编译
# ════════════════════════════════════════════════════════════════
os.environ["PYTORCH_DISABLE_TRITON"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


# ════════════════════════════════════════════════════════════════
# 【关键】校正参数 - 针对30000权重的头部相机模型
# 从片段1数据统计得出
# ════════════════════════════════════════════════════════════════
FIXED_CORRECTION_DEG = np.array([
    18.7941,   # right_joint_0
     5.0680,   # right_joint_1
    -3.5635,   # right_joint_2
    -7.1282,   # right_joint_3
    -1.8197,   # right_joint_4
     2.3149,   # right_joint_5
     1.8379,   # right_joint_6
    28.1949,   # right_dexterous_hand
])

FIXED_CORRECTION_RAD = FIXED_CORRECTION_DEG * np.pi / 180


# ════════════════════════════════════════════════════════════════
# 数据集配置 - 仅使用头部相机
# ════════════════════════════════════════════════════════════════

CAMERA_KEYS = {
    "cam_high": "observation.images.cam_high",
}

DEFAULT_PROMPT = "right arm pick and place task"

JOINT_NAMES_8D = [
    "right_joint_0", "right_joint_1", "right_joint_2",
    "right_joint_3", "right_joint_4", "right_joint_5",
    "right_joint_6", "right_dexterous_hand"
]


# ════════════════════════════════════════════════════════════════
# 数据转换类 - 仅使用头部相机
# ════════════════════════════════════════════════════════════════

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class RightArmHeadcamInputs(_transforms.DataTransformFn):
    """仅使用头部相机的输入转换"""
    def __call__(self, data: dict) -> dict:
        import einops

        state = np.asarray(data["state"])
        if state.ndim > 1:
            state = state[..., :8]
        else:
            state = state[:8]

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).clip(0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        images_dict = data.get("images", data.get("image", {}))

        # 仅提取头部相机图像
        extracted = []
        for key in ["cam_high"]:
            if key in images_dict:
                extracted.append(convert_image(images_dict[key]))
            else:
                raise ValueError(
                    f"图像 key '{key}' 未找到。可用 key: {list(images_dict.keys())}"
                )

        # 仅使用头部相机，禁用其他相机
        openpi_images = {
            "base_0_rgb": extracted[0],
            "left_wrist_0_rgb": np.zeros_like(extracted[0]),
            "right_wrist_0_rgb": np.zeros_like(extracted[0]),
        }
        openpi_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.False_,
            "right_wrist_0_rgb": np.False_,
        }

        result = {
            "state": state.copy(),
            "observation/state": state.copy(),
            "image": openpi_images,
            "image_mask": openpi_masks,
        }

        if "prompt" in data:
            result["prompt"] = data["prompt"]

        return result


@dataclasses.dataclass(frozen=True)
class PadStateTo32(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] < 32:
            state = np.pad(state, (0, 32 - state.shape[-1]))
        data = dict(data)
        data["state"] = state
        return data


@dataclasses.dataclass(frozen=True)
class RightArmOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data.get("actions", np.zeros((50, 8), dtype=np.float32))
        return {"actions": np.asarray(actions[:, :8], dtype=np.float32)}


# ════════════════════════════════════════════════════════════════
# 模型构建
# ════════════════════════════════════════════════════════════════

def build_policy(
    checkpoint_dir: pathlib.Path,
    prompt: str,
    device: str,
    fixed_noise_seed: Optional[int] = None,
) -> _policy.Policy:
    print("\n" + "=" * 70)
    print("【构建策略 - 右臂8维，仅头部相机】")
    print("=" * 70)

    train_config = _config.TrainConfig(
        name="pi0_right_arm_infer",
        model=pi0_config.Pi0Config(action_dim=32, pytorch_compile_mode=None),
        data=_config.FakeDataConfig(),
        policy_metadata={"checkpoint": str(checkpoint_dir)},
        assets_base_dir=str(checkpoint_dir.parent),
        checkpoint_base_dir=str(checkpoint_dir.parent),
        wandb_enabled=False,
    )

    print(f"加载权重: {checkpoint_dir / 'model.safetensors'}")
    weight_path = checkpoint_dir / "model.safetensors"
    model = train_config.model.load_pytorch(train_config, str(weight_path))
    print(f"✓ 权重加载完成")

    print(f"加载 norm_stats...")
    norm_stats = _normalize.load(checkpoint_dir / "assets" / "local" / "right_arm_head_cam")
    print(f"✓ norm_stats 加载完成: {list(norm_stats.keys())}")

    # 使用仅头部相机的输入转换
    transforms = [
        RightArmHeadcamInputs(),
        _transforms.Normalize(norm_stats, use_quantiles=True),
        PadStateTo32(),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(
            tokenizer=_tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)
        ),
    ]

    delta_mask = _transforms.make_bool_mask(7, -1)

    output_transforms = [
        _transforms.Unnormalize(norm_stats, use_quantiles=True, strict=False),
        _transforms.AbsoluteActions(delta_mask),
        RightArmOutputs(),
    ]

    policy_sample_kwargs = {}
    if fixed_noise_seed is not None:
        rng = np.random.default_rng(fixed_noise_seed)
        fixed_noise_np = rng.standard_normal((1, 50, 32), dtype=np.float32)
        if device.startswith("cuda") and torch.cuda.is_available():
            policy_sample_kwargs["noise"] = torch.from_numpy(fixed_noise_np).to(device)
        else:
            policy_sample_kwargs["noise"] = torch.from_numpy(fixed_noise_np)

    print(f"创建 Policy 对象...")
    policy = _policy.Policy(
        model,
        transforms=transforms,
        output_transforms=output_transforms,
        sample_kwargs=policy_sample_kwargs,
        metadata=train_config.policy_metadata,
        pytorch_device=device,
        is_pytorch=True,
    )
    print(f"✓ Policy 创建完成")

    return policy


# ════════════════════════════════════════════════════════════════
# 校正函数
# ════════════════════════════════════════════════════════════════

def apply_correction(model_delta: np.ndarray, use_correction: bool = True) -> np.ndarray:
    """
    应用校正

    原理：
    - 模型输出的是 "当前状态 + 固定偏差"
    - 减去偏差后，输出接近 "当前状态"
    - 期望的增量 = action - state ≈ 0
    """
    if not use_correction:
        return model_delta

    return model_delta - FIXED_CORRECTION_RAD


# ════════════════════════════════════════════════════════════════
# 视频帧提取器
# ════════════════════════════════════════════════════════════════

class VideoFrameExtractor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __len__(self) -> int:
        return self.total_frames

    def read_frame(self, frame_index: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"无法读取帧 {frame_index}")
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self):
        self.cap.release()


# ════════════════════════════════════════════════════════════════
# Parquet 数据读取器
# ════════════════════════════════════════════════════════════════

class ParquetReader:
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self.table = pq.read_table(parquet_path)
        self.nrows = self.table.num_rows
        self.state_data = self.table.column("observation.state").to_numpy()
        self.action_data = self.table.column("action").to_numpy()

    def __len__(self) -> int:
        return self.nrows

    def get_state(self, frame_index: int) -> np.ndarray:
        state = self.state_data[frame_index]
        return np.asarray(state[:8], dtype=np.float32)

    def get_action(self, frame_index: int) -> np.ndarray:
        action = self.action_data[frame_index]
        return np.asarray(action[:8], dtype=np.float32)

    def get_frame_info(self, frame_index: int) -> dict:
        return {
            "episode_index": self.table.column("episode_index")[frame_index].as_py(),
            "frame_index": self.table.column("frame_index")[frame_index].as_py(),
            "timestamp": self.table.column("timestamp")[frame_index].as_py(),
            "state": self.get_state(frame_index),
            "action": self.get_action(frame_index),
        }


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def format_angle(val_deg: float) -> str:
    """格式化角度值，带符号"""
    sign = "+" if val_deg >= 0 else ""
    return f"{sign}{val_deg:.2f}°"


def print_frame_summary(
    frame_idx: int, total_frames: int,
    state_deg: np.ndarray, model_delta_deg: np.ndarray,
    corrected_delta_deg: np.ndarray, action_deg: np.ndarray,
    use_correction: bool, frame_mae_deg: float
):
    """打印简洁的帧摘要表格"""
    import sys

    if frame_idx == 0:
        # 表头
        print("\n" + "─" * 107)
        print("│ 关节名称            │         输入角度 │          模型增量 │           校正后 │          目标角度 │          预测误差 │")
        print("├" + "─" * 23 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┤")

    # 数据行
    for i, name in enumerate(JOINT_NAMES_8D):
        # 计算校正后的预测状态和误差
        if use_correction:
            predicted_deg = state_deg[i] + corrected_delta_deg[i]
            error_deg = predicted_deg - action_deg[i]
        else:
            predicted_deg = state_deg[i] + model_delta_deg[i]
            error_deg = predicted_deg - action_deg[i]

        print(f"│ {name:<21} │ {format_angle(state_deg[i]):>14} │ {format_angle(model_delta_deg[i]):>14} │ {format_angle(corrected_delta_deg[i]):>15} │ {format_angle(action_deg[i]):>14} │ {format_angle(error_deg):>14} │")

    # 底部横线
    if frame_idx == total_frames - 1:
        print("─" * 107)
    sys.stdout.flush()


# ════════════════════════════════════════════════════════════════
# 主推理函数
# ════════════════════════════════════════════════════════════════

def run_inference(
    checkpoint_dir: pathlib.Path,
    dataset_dir: pathlib.Path,
    episode_index: int,
    device: str,
    use_correction: bool = False,
    save_frames: bool = True,
    max_frames: Optional[int] = None,
    fixed_noise_seed: Optional[int] = None,
    output_csv: Optional[str] = None,
    verbose: bool = True,
):
    """运行离线推理 - 仅使用头部相机，可选校正"""

    # 静默模式 = 不输出详细信息
    quiet = not verbose

    print("\n" + "=" * 70)
    print("【OpenPI 右臂8维离线推理 - 仅头部相机】")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"数据集: {dataset_dir}")
    print(f"Episode: {episode_index}")
    print(f"设备: {device}")
    print("相机配置: 仅使用 cam_high (头部相机)")
    print(f"使用校正: {use_correction}")
    if use_correction:
        print(f"校正参数 (deg): {FIXED_CORRECTION_DEG.tolist()}")
    print("=" * 70)

    # 1. 构建策略模型
    t_start_total = time.time()
    policy = build_policy(checkpoint_dir, DEFAULT_PROMPT, device, fixed_noise_seed)
    t_model_load = time.time()
    if verbose:
        print(f"\n模型加载耗时: {t_model_load - t_start_total:.2f} 秒")

    # 2. 确定视频和 parquet 文件路径
    video_dir = dataset_dir / "videos"
    data_dir = dataset_dir / "data"

    # 3. 初始化视频提取器 - 仅头部相机
    if verbose:
        print("\n【初始化视频提取器 - 仅头部相机】")
    extractors = {}
    chunks_size = 1000
    ep_chunk = episode_index // chunks_size
    ep_file = episode_index % chunks_size
    
    for cam_name, cam_key in CAMERA_KEYS.items():
        video_path = video_dir / cam_key / f"chunk-{ep_chunk:03d}" / f"file-{ep_file:03d}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        extractors[cam_name] = VideoFrameExtractor(str(video_path))
        if verbose:
            print(f"  {cam_name}: {video_path} ({len(extractors[cam_name])} 帧)")

    # 4. 初始化 parquet 读取器
    parquet_path = data_dir / f"chunk-{ep_chunk:03d}" / f"file-{ep_file:03d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")
    parquet_reader = ParquetReader(str(parquet_path))
    if verbose:
        print(f"  Parquet: {parquet_path} ({len(parquet_reader)} 条记录)")

    # 获取总帧数
    total_frames = min(len(ext) for ext in extractors.values())
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    if verbose:
        print(f"\n总帧数: {total_frames}")

    # 5. 创建图像保存目录
    frames_save_dir = None
    if save_frames:
        frames_save_dir = checkpoint_dir / "right_arm_frames_headcam_only" / f"episode_{episode_index}" / ("corrected" if use_correction else "raw")
        for cam_name in CAMERA_KEYS.keys():
            (frames_save_dir / cam_name).mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"图像保存目录: {frames_save_dir}")

    # 6. 开始推理
    if verbose:
        print("\n" + "=" * 70)
        print("【开始推理】")
        print("=" * 70)

    total_inference_time = 0.0
    all_results = []
    total_mae = 0.0
    per_joint_errors = {name: [] for name in JOINT_NAMES_8D}

    for frame_idx in range(total_frames):
        # 读取图像 - 仅头部相机
        frames = {}
        for cam_name, extractor in extractors.items():
            frame = extractor.read_frame(frame_idx)
            frames[cam_name] = frame

            if save_frames and frames_save_dir is not None:
                save_path = frames_save_dir / cam_name / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(save_path), frame)

        # 读取关节状态和目标动作
        frame_info = parquet_reader.get_frame_info(frame_idx)
        state_full = frame_info["state"]
        action_gt = frame_info["action"]

        # 构造模型输入 - 仅头部相机
        obs = {
            "state": state_full,
            "images": {
                "cam_high": frames["cam_high"],
            },
            "prompt": DEFAULT_PROMPT,
        }

        # 执行推理
        t_infer_start = time.time()
        result = policy.infer(obs)
        inference_time = time.time() - t_infer_start
        total_inference_time += inference_time

        # 解析输出 - 模型输出的是增量动作
        model_delta = result["actions"][0]

        # 应用校正
        corrected_delta = apply_correction(model_delta, use_correction)

        # 计算预测状态
        predicted_state = state_full + model_delta
        predicted_corrected = state_full + corrected_delta

        # 计算误差
        error_before = predicted_state - action_gt
        error_after = predicted_corrected - action_gt

        if use_correction:
            error = error_after
            frame_mae = np.mean(np.abs(error_after))
        else:
            error = error_before
            frame_mae = np.mean(np.abs(error_before))

        total_mae += frame_mae

        # 记录每个关节的误差
        for i, name in enumerate(JOINT_NAMES_8D):
            per_joint_errors[name].append(abs(error[i]))

        # 转换为角度
        state_deg = state_full * 180 / math.pi
        model_delta_deg = model_delta * 180 / math.pi
        corrected_delta_deg = corrected_delta * 180 / math.pi
        action_deg = action_gt * 180 / math.pi

        print_frame_summary(
            frame_idx, total_frames,
            state_deg, model_delta_deg, corrected_delta_deg, action_deg,
            use_correction, frame_mae
        )

        # 保存结果
        all_results.append({
            "frame_index": frame_idx,
            "timestamp": frame_info["timestamp"],
            "episode_index": frame_info["episode_index"],
            # 当前状态
            "state_current_rad": state_full.tolist(),
            "state_current_deg": (state_full * 180 / math.pi).tolist(),
            # 模型原始输出
            "model_delta_rad": model_delta.tolist(),
            "model_delta_deg": (model_delta * 180 / math.pi).tolist(),
            # 校正后的输出
            "corrected_delta_rad": corrected_delta.tolist(),
            "corrected_delta_deg": (corrected_delta * 180 / math.pi).tolist(),
            # 预测状态（未校正）
            "predicted_state_rad": predicted_state.tolist(),
            "predicted_state_deg": (predicted_state * 180 / math.pi).tolist(),
            # 预测状态（校正后）
            "predicted_corrected_rad": predicted_corrected.tolist(),
            "predicted_corrected_deg": (predicted_corrected * 180 / math.pi).tolist(),
            # 数据集目标动作
            "action_gt_rad": action_gt.tolist(),
            "action_gt_deg": (action_gt * 180 / math.pi).tolist(),
            # 误差
            "error_before_rad": error_before.tolist(),
            "error_before_deg": (error_before * 180 / math.pi).tolist(),
            "error_after_rad": error_after.tolist(),
            "error_after_deg": (error_after * 180 / math.pi).tolist(),
            "mae_before_deg": np.mean(np.abs(error_before)) * 180 / math.pi,
            "mae_after_deg": np.mean(np.abs(error_after)) * 180 / math.pi,
            "inference_time_ms": inference_time * 1000,
        })

    # 计算平均误差
    avg_mae = total_mae / total_frames
    per_joint_mae = {name: np.mean(errors) * 180 / math.pi for name, errors in per_joint_errors.items()}

    # 8. 保存结果到 CSV (与终端表格格式一致)
    if output_csv:
        csv_path = pathlib.Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # 写入表头
            header = ["帧索引", "关节名称", "输入角度", "模型增量", "校正后", "目标角度", "预测误差"]
            writer.writerow(header)

            # 写入数据行 (每帧8行，每行一个关节)
            for r in all_results:
                frame_idx = r["frame_index"]
                for i, name in enumerate(JOINT_NAMES_8D):
                    row = [
                        frame_idx,
                        name,
                        f"{r['state_current_deg'][i]:.2f}",
                        f"{r['model_delta_deg'][i]:.2f}",
                        f"{r['corrected_delta_deg'][i]:.2f}",
                        f"{r['action_gt_deg'][i]:.2f}",
                        f"{r['error_after_deg'][i]:.2f}",
                    ]
                    writer.writerow(row)

        if verbose:
            print(f"\n结果已保存到 CSV: {csv_path}")

        # 保存完整 JSON
        json_path = csv_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        if verbose:
            print(f"完整结果已保存到 JSON: {json_path}")

    # 计算校正前后的MAE
    mae_before = np.mean([np.mean(np.abs(np.array(r['error_before_rad']))) for r in all_results])
    mae_after = np.mean([np.mean(np.abs(np.array(r['error_after_rad']))) for r in all_results])

    # 9. 总结
    t_total = time.time() - t_start_total

    if verbose:
        print("\n" + "=" * 70)
        print("【推理完成 - 仅头部相机】")
        print("=" * 70)
        print(f"处理帧数: {total_frames}")
        print(f"模型加载耗时: {t_model_load - t_start_total:.2f} 秒")
        print(f"推理总耗时: {total_inference_time:.2f} 秒")
        print(f"平均每帧推理: {total_inference_time / total_frames:.2f} 秒")
        print(f"总耗时: {t_total:.2f} 秒")
        print()
        print("【校正效果】")
        print(f"  校正前 MAE: {mae_before * 180 / math.pi:.4f}°")
        print(f"  校正后 MAE: {mae_after * 180 / math.pi:.4f}°")
        if use_correction:
            improvement = (mae_before - mae_after) / mae_before * 100
            print(f"  改善比例: {improvement:.1f}%")

        if save_frames:
            print(f"\n图像保存目录: {frames_save_dir}")

    # 关闭视频提取器
    for extractor in extractors.values():
        extractor.close()

    return all_results, avg_mae, per_joint_mae


# ════════════════════════════════════════════════════════════════
# 主程序入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OpenPI 右臂8维离线推理 - 仅头部相机，无校正",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint 目录路径"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="LeRobot 数据集目录路径"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=1,
        help="Episode 索引（片段1，默认: 1）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备（默认: cuda）"
    )
    parser.add_argument(
        "--use-correction",
        action="store_true",
        default=True,
        help="使用校正（默认开启）"
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="不使用校正"
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="不保存提取的图像"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最多处理帧数（默认: 全部）"
    )
    parser.add_argument(
        "--fixed-noise-seed",
        type=int,
        default=None,
        help="固定推理噪声 seed（用于调试重复性）"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="输出 CSV 文件路径"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式，减少输出"
    )

    args = parser.parse_args()

    # 校正逻辑：默认使用校正，除非明确指定 --no-correction
    use_correction = args.use_correction and not args.no_correction

    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    dataset_dir = pathlib.Path(args.dataset_dir).resolve()

    if not checkpoint_dir.exists():
        print(f"错误: Checkpoint 目录不存在: {checkpoint_dir}")
        sys.exit(1)

    if not dataset_dir.exists():
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，降级到 CPU")
        args.device = "cpu"

    run_inference(
        checkpoint_dir=checkpoint_dir,
        dataset_dir=dataset_dir,
        episode_index=args.episode_index,
        device=args.device,
        use_correction=use_correction,
        save_frames=not args.no_save_frames,
        max_frames=args.max_frames,
        fixed_noise_seed=args.fixed_noise_seed,
        output_csv=args.output_csv,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
