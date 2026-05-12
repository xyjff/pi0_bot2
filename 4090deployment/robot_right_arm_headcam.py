#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ════════════════════════════════════════════════════════════════
# OpenPI 右臂8维真机实时推理脚本
# ════════════════════════════════════════════════════════════════
"""
功能：
1. TCP服务器接收机器人端观测数据
2. 仅使用头部相机图像进行推理
3. 推理完成后打印并下发50步动作序列给机器人端执行
4. 支持自定义动作处理器（ActionHandler）扩展
5. 自动保存每次推理的输入图片和动作数据

保存文件：
    {save_dir}/
        infer0001_input_image.jpg    # 输入图片
        infer0001_actions.json       # 动作数据
        infer0002_input_image.jpg
        infer0002_actions.json
        ...

使用方式：
cd /home/dmh/xyj_zhipingfang/model3_openpi0.5/my_bot2_deployment/4090deployment
/home/dmh/xyj_zhipingfang/model3_openpi0.5/openpi-main/.venv/bin/python robot_right_arm_headcam.py \
  --checkpoint-dir /home/dmh/xyj_zhipingfang/model3_openpi0.5/checkpoints/pi0_checkpoint_xin128tiao_headcam/28000 \
  --listen-port 9000

键盘操作：
    按 Enter/回车: 启动连续推理并持续下发动作给机器人端
    输入 s: 停止连续推理
    输入 q: 退出程序

扩展方式：
    继承 ActionHandler 类，实现 on_action_ready() 方法处理动作，
    例如发送到机器人执行、保存日志等。
"""

import argparse
import base64
import dataclasses
import json
import math
import os
import pathlib
import socket
import struct
import sys
import threading
import time
from typing import Optional

# ════════════════════════════════════════════════════════════════
# 【关键】设置 Python 路径
# ════════════════════════════════════════════════════════════════
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_OPENPI_ROOT = _SCRIPT_DIR.parent.parent / "openpi-main"
sys.path.insert(0, str(_OPENPI_ROOT / "src"))
sys.path.insert(0, str(_OPENPI_ROOT / "packages" / "openpi-client" / "src"))

if "OPENPI_DATA_HOME" not in os.environ:
    os.environ["OPENPI_DATA_HOME"] = str(_OPENPI_ROOT / ".openpi_cache")

# ════════════════════════════════════════════════════════════════
# 【关键】禁用 Triton 编译
# ════════════════════════════════════════════════════════════════
os.environ["PYTORCH_DISABLE_TRITON"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import cv2
import numpy as np
import torch
torch._dynamo.config.suppress_errors = True

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


# ════════════════════════════════════════════════════════════════
# 常量配置
# ════════════════════════════════════════════════════════════════

DEFAULT_PROMPT = "right arm pick and place task"

JOINT_NAMES_8D = [
    "right_joint_0", "right_joint_1", "right_joint_2",
    "right_joint_3", "right_joint_4", "right_joint_5",
    "right_joint_6", "right_dexterous_hand"
]


# ════════════════════════════════════════════════════════════════
# 数据转换类
# ════════════════════════════════════════════════════════════════

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

        extracted = []
        for key in ["cam_high"]:
            if key in images_dict:
                extracted.append(convert_image(images_dict[key]))
            else:
                raise ValueError(
                    f"图像 key '{key}' 未找到。可用 key: {list(images_dict.keys())}"
                )

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
    """构建 OpenPI 右臂8维策略模型"""
    print("\n" + "=" * 70)
    print("【构建策略 - 右臂8维，仅头部相机】")
    print("=" * 70)

    train_config = _config.TrainConfig(
        name="pi0_right_arm_infer",
        model=pi0_config.Pi0Config(action_dim=32, pi05=False, pytorch_compile_mode=None),
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

    transforms = [
        RightArmHeadcamInputs(),
        _transforms.Normalize(norm_stats, use_quantiles=False),
        PadStateTo32(),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(
            tokenizer=_tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)
        ),
    ]

    delta_mask = _transforms.make_bool_mask(7, -1)

    output_transforms = [
        _transforms.Unnormalize(norm_stats, use_quantiles=False),
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
# 网络通信函数
# ════════════════════════════════════════════════════════════════

def send_msg(sock: socket.socket, obj: dict) -> None:
    """发送JSON消息: [4字节长度(big-endian)][JSON字节串]"""
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = struct.pack(">I", len(payload))
    sock.sendall(header + payload)


def recv_msg(sock: socket.socket, timeout: float = 10.0) -> Optional[dict]:
    """接收JSON消息"""
    sock.settimeout(timeout)
    try:
        header = _recv_exact(sock, 4)
        if header is None:
            return None
        length = struct.unpack(">I", header)[0]

        if length > 10 * 1024 * 1024:
            print(f"[RECV] 非法消息长度: {length}")
            return None

        payload = _recv_exact(sock, length)
        if payload is None:
            return None

        msg = json.loads(payload.decode("utf-8"))
        return msg
    except json.JSONDecodeError as e:
        print(f"[RECV] JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"[RECV] 接收消息异常: {e}")
        return None


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """接收指定字节数"""
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[RECV] _recv_exact 失败: {e}")
            return None
    return buf


def decode_image(img_b64: str) -> Optional[np.ndarray]:
    """解码base64图像"""
    if img_b64 is None:
        return None
    try:
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[DECODE] 图像解码失败: {e}")
        return None


def print_separator(title: str = ""):
    """打印分隔符"""
    if title:
        print(f"\n{'='*70}")
        print(f"【{title}】")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")


# ════════════════════════════════════════════════════════════════
# ActionHandler 基类 - 动作处理器接口
# ════════════════════════════════════════════════════════════════

class ActionHandler:
    """
    动作处理器基类
    
    扩展方式：
        继承此类并实现 on_action_ready() 方法，
        即可自定义动作处理逻辑（如发送到机器人、保存日志等）。
    
    示例：
        class RobotActionHandler(ActionHandler):
            def on_action_ready(self, action_msg: dict):
                # 发送动作到机器人
                send_msg(self.robot_socket, action_msg)
    """

    def on_action_ready(self, action_msg: dict):
        """
        推理完成后的回调
        
        Args:
            action_msg: 包含以下字段的字典
                - msg_type: 消息类型 ("action_sequence")
                - obs_seq: 观测序列号
                - actions: 机器人端执行的50步动作序列
                - actions_50: 50步动作序列 (list of list, 单位: 度/原始值)
                - infer_ms: 推理耗时 (毫秒)
        """
        pass

    def on_error(self, error: Exception):
        """错误处理回调"""
        pass


class PrintOnlyActionHandler(ActionHandler):
    """
    打印/保存动作处理器
    
    默认用于记录推理结果；动作下发由 RealtimeInferenceClient 完成。
    """

    def __init__(self, save_dir: pathlib.Path = None):
        super().__init__()
        self.save_dir = save_dir
        self._infer_count = 0

    def on_action_ready(self, action_msg: dict):
        self._infer_count += 1
        infer_num = self._infer_count
        
        # 保存动作数据
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            action_path = self.save_dir / f"infer{infer_num:04d}_actions.json"
            with open(action_path, 'w') as f:
                json.dump(action_msg, f, indent=2)
        
        # 打印动作信息
        print(f"\n{'='*70}")
        print(f"[动作信息 - 已保存/准备下发给机器人]")
        print(f"{'='*70}")
        print(f"  obs_seq: {action_msg.get('obs_seq', 'N/A')}")
        print(f"  infer_ms: {action_msg.get('infer_ms', 0):.1f}")
        if self.save_dir:
            print(f"  动作已保存: {action_path}")
        actions_50 = action_msg.get('actions_50', [])
        print(f"  50步动作序列 (deg): 共 {len(actions_50)} 步")
        print(f"\n  {'步骤':<6} │ {'j0':<10} │ {'j1':<10} │ {'j2':<10} │ {'j3':<10} │ {'j4':<10} │ {'j5':<10} │ {'j6':<10} │ {'gripper':<8}")
        print(f"  {'-'*6}─┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*9}")
        for step, action in enumerate(actions_50[:10]):
            j_vals = [f"{action[i]:>+8.3f}" for i in range(7)]
            g_val = f"{action[7]:>7.4f}"
            print(f"  {step:<6} │ {' │ '.join(j_vals)} │ {g_val}")
        if len(actions_50) > 10:
            print(f"  ... (共 {len(actions_50)} 步)")
        print(f"{'='*70}")


# ════════════════════════════════════════════════════════════════
# RealtimeInferenceClient 类
# ════════════════════════════════════════════════════════════════

class RealtimeInferenceClient:
    """
    实时推理客户端
    
    功能：
    1. TCP服务器监听，接受机器人端连接
    2. 接收线程持续接收观测数据
    3. 按键触发推理
    4. 通过 ActionHandler 处理推理结果
    
    Args:
        policy: OpenPI策略模型
        action_handler: 动作处理器 (默认: PrintOnlyActionHandler)
        save_dir: 保存目录，保存输入图片和动作
    """

    def __init__(self, policy, action_handler: ActionHandler = None,
                 save_dir: pathlib.Path = None,
                 gripper_threshold: float = 0.5):
        self.policy = policy
        self.action_handler = action_handler or PrintOnlyActionHandler(save_dir)
        self.save_dir = save_dir
        self.gripper_threshold = gripper_threshold
        
        # 如果 action_handler 是默认的 PrintOnlyActionHandler 且没有指定 save_dir
        if isinstance(self.action_handler, PrintOnlyActionHandler) and self.action_handler.save_dir is None:
            self.action_handler.save_dir = save_dir

        self._sock: Optional[socket.socket] = None
        self._running = False
        self._client_connected = False

        self._latest_obs = None
        self._obs_lock = threading.Lock()
        self._obs_count = 0

        self._infer_count = 0
        self._action_count = 0
        self._continuous_infer = False
        self._continuous_thread: Optional[threading.Thread] = None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _accept_connection(self, server_sock: socket.socket):
        """接受机器人端连接"""
        print(f"\n[等待] 等待机器人端连接...")
        server_sock.settimeout(1.0)

        while self._running:
            try:
                client_sock, addr = server_sock.accept()
                print(f"[连接] 机器人端已连接: {addr}")
                self._sock = client_sock
                self._client_connected = True
                return
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[错误] 接受连接失败: {e}")
                time.sleep(1)

    def _receive_obs_loop(self):
        """接收观测数据线程"""
        print("[接收] 观测接收线程已启动")

        while self._running:
            if self._sock is None:
                time.sleep(0.1)
                continue

            try:
                msg = recv_msg(self._sock, timeout=1.0)
                if msg is None:
                    continue

                if msg.get("msg_type") == "observation":
                    with self._obs_lock:
                        self._latest_obs = msg
                        self._obs_count += 1

                    if self._obs_count % 30 == 0:
                        obs_seq = msg.get("obs_seq", 0)
                        joints_r = msg.get("joints_right", [])
                        images = msg.get("images", {})
                        head_ok = images.get("head") is not None

                        print(f"[接收 #{self._obs_count}] seq={obs_seq}, "
                              f"joints_r={joints_r[:3] if joints_r else 'N/A'}..., "
                              f"head={'OK' if head_ok else 'FAIL'}")

            except Exception as e:
                print(f"[接收] 接收异常: {e}")
                time.sleep(0.1)

    def get_latest_obs(self):
        """获取最新观测数据"""
        with self._obs_lock:
            return self._latest_obs

    def _format_robot_actions(self, actions_50_deg: np.ndarray) -> list[dict]:
        """转换为 robot_executor.py 期望的 action_sequence 单步格式。"""
        robot_actions = []
        for action in actions_50_deg:
            action_list = [float(x) for x in action.tolist()]
            gripper_raw = action_list[7]
            gripper_cmd = 1.0 if gripper_raw >= self.gripper_threshold else 0.0
            robot_actions.append({
                "joints_right": action_list[:7],
                "dexhand_right": gripper_cmd,
                "dexhand_right_raw": gripper_raw,
            })
        return robot_actions

    def _send_action_sequence(self, action_msg: dict) -> bool:
        """下发动作序列到机器人端。"""
        if self._sock is None or not self._client_connected:
            print("[错误] 机器人端未连接，动作未下发")
            return False

        send_msg(self._sock, action_msg)
        self._action_count += 1
        print(f"[下发] 动作序列已发送给机器人: obs_seq={action_msg.get('obs_seq')}, "
              f"步数={len(action_msg.get('actions', []))}, "
              f"累计下发={self._action_count}")
        return True

    def _continuous_inference_loop(self):
        """连续推理并持续下发动作序列。"""
        print("[连续推理] 已启动：每次推理完成后立即下发下一段50步动作")
        while self._running and self._continuous_infer:
            try:
                action_msg = self.run_inference()
                if action_msg is None:
                    time.sleep(0.1)
            except Exception as e:
                print(f"[连续推理] 推理/下发异常: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        print("[连续推理] 已停止")

    def _start_continuous_inference(self):
        """启动连续推理线程。"""
        if self._continuous_thread is not None and self._continuous_thread.is_alive():
            print("[连续推理] 已经在运行，无需重复启动")
            return

        self._continuous_infer = True
        self._continuous_thread = threading.Thread(
            target=self._continuous_inference_loop,
            name="CONTINUOUS_INFER",
            daemon=True,
        )
        self._continuous_thread.start()

    def _stop_continuous_inference(self):
        """停止连续推理线程。"""
        if not self._continuous_infer:
            print("[连续推理] 当前未运行")
            return
        print("[连续推理] 正在停止，当前推理完成后生效...")
        self._continuous_infer = False

    def run_inference(self):
        """使用最新观测执行推理并处理动作"""
        self._infer_count += 1
        infer_num = self._infer_count

        print(f"\n{'='*70}")
        print(f"【推理 #{infer_num}】")
        print(f"{'='*70}")

        obs = self.get_latest_obs()
        if obs is None:
            print("[错误] 等待机器人端观测数据...")
            return None

        obs_seq = obs.get("obs_seq", 0)
        ts_ns = obs.get("ts_ns", 0)
        joints_right = obs.get("joints_right", [])
        dexhand_right = obs.get("dexhand_right", 0.0)
        images_raw = obs.get("images", {})

        print(f"\n[观测原始数据]")
        print(f"  obs_seq: {obs_seq}")
        print(f"  时间戳: {ts_ns} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_ns/1e9)) if ts_ns else 'N/A'})")
        print(f"  joints_right (deg): {[f'{x:.2f}' for x in joints_right]}")
        print(f"  dexhand_right: {dexhand_right}")
        print(f"  图像keys: {list(images_raw.keys())}")

        head_img = decode_image(images_raw.get("head"))
        print(f"\n[图像解码] head_img: {'成功' if head_img is not None else '失败'} {head_img.shape if head_img is not None else ''}")

        if head_img is None:
            print("[错误] 头部图像数据无效")
            return None

        # 保存输入图片
        if self.save_dir:
            img_path = self.save_dir / f"infer{infer_num:04d}_input_image.jpg"
            cv2.imwrite(str(img_path), head_img)
            print(f"\n[保存] 输入图片已保存: {img_path}")

        if len(joints_right) < 7:
            print(f"[错误] 关节数据不足: {joints_right}")
            return None

        joints_right_rad = [deg * math.pi / 180.0 for deg in joints_right[:7]]
        state_full = np.array(joints_right_rad + [float(dexhand_right)], dtype=np.float32)

        print(f"\n[模型输入状态 - 8维右臂]")
        for i, name in enumerate(JOINT_NAMES_8D):
            deg = state_full[i] if i == 7 else state_full[i] * 180.0 / math.pi
            print(f"  {name}: {state_full[i]:.4f} rad ({deg:.3f}°)")

        model_obs = {
            "state": state_full,
            "images": {
                "cam_high": cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB),
            },
            "prompt": DEFAULT_PROMPT,
        }

        print(f"\n[推理] 执行推理中...")
        t_infer_start = time.time()
        result = self.policy.infer(model_obs)
        inference_time = time.time() - t_infer_start
        print(f"[推理] 完成，耗时: {inference_time*1000:.1f}ms")

        actions_50 = np.asarray(result["actions"], dtype=np.float32)
        if actions_50.ndim != 2 or actions_50.shape[1] < 8:
            print(f"[错误] 模型输出动作维度异常: {actions_50.shape}")
            return None

        # 将输出动作转换为角度 (与离线脚本一致: 前7维转度，第8维保持原始值)
        actions_50_deg = np.zeros_like(actions_50)
        actions_50_deg[:, :7] = actions_50[:, :7] * 180.0 / math.pi
        actions_50_deg[:, 7] = actions_50[:, 7]

        if not np.isfinite(actions_50_deg).all():
            print("[错误] 模型输出包含 NaN/Inf，动作未下发")
            return None

        # 打印推理结果 (与离线脚本一致的格式)
        print(f"\n{'='*70}")
        print(f"[推理结果 #{infer_num} - 50步动作序列]")
        print(f"{'='*70}")

        print(f"\n  {'步骤':<6} │ {'j0(°)':<10} │ {'j1(°)':<10} │ {'j2(°)':<10} │ {'j3(°)':<10} │ {'j4(°)':<10} │ {'j5(°)':<10} │ {'j6(°)':<10} │ {'gripper':<8}")
        print(f"  {'-'*6}─┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*9}")

        for step in range(min(10, len(actions_50_deg))):
            action = actions_50_deg[step]
            j_vals = [f"{action[i]:>+8.3f}" for i in range(7)]
            g_val = f"{action[7]:>7.4f}"
            print(f"  {step:<6} │ {' │ '.join(j_vals)} │ {g_val}")

        if len(actions_50_deg) > 10:
            print(f"  ... (共 {len(actions_50_deg)} 步)")

        print(f"\n  推理耗时: {inference_time*1000:.1f}ms")

        robot_actions = self._format_robot_actions(actions_50_deg[:, :8])
        gripper_raw = actions_50_deg[:, 7]
        gripper_close_count = sum(a["dexhand_right"] >= 0.5 for a in robot_actions)
        print(f"  夹爪raw范围: min={float(gripper_raw.min()):.6f}, "
              f"max={float(gripper_raw.max()):.6f}, "
              f"threshold={self.gripper_threshold:.6f}, "
              f"闭合步数={gripper_close_count}/{len(robot_actions)}")
        action_msg = {
            "msg_type": "action_sequence",
            "obs_seq": obs_seq,
            "actions": robot_actions,
            "actions_50": actions_50_deg[:, :8].tolist(),
            "infer_ms": inference_time * 1000,
        }

        try:
            self.action_handler.on_action_ready(action_msg)
        except Exception as e:
            print(f"[错误] ActionHandler 处理失败: {e}")
            self.action_handler.on_error(e)

        try:
            self._send_action_sequence(action_msg)
        except Exception as e:
            print(f"[错误] 动作下发失败: {e}")
            self.action_handler.on_error(e)

        return action_msg

    def run(self, listen_port: int):
        """运行推理客户端"""
        print("\n" + "=" * 70)
        print("【OpenPI 真机实时推理客户端 - 右臂8维，仅头部相机】")
        print("=" * 70)
        print(f"监听端口: {listen_port}")
        print(f"使用校正: 无（模型输出直接使用）")
        print(f"夹爪阈值: 模型输出 >= {self.gripper_threshold} 时下发闭合(1)，否则张开(0)")
        print(f"动作处理: {self.action_handler.__class__.__name__}")
        print("=" * 70)

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", listen_port))
        server_sock.listen(1)
        print(f"[启动] TCP服务器已启动，监听端口 {listen_port}")

        self._running = True

        recv_thread = threading.Thread(target=self._receive_obs_loop, daemon=True)
        recv_thread.start()

        self._accept_connection(server_sock)

        print_separator("键盘控制模式")
        print("操作说明:")
        print("  按 Enter/回车: 启动连续推理，并持续下发动作给机器人端")
        print("  输入 s: 停止连续推理")
        print("  输入 q: 退出程序")
        print("=" * 70)

        last_obs_time = "无"

        try:
            while self._running:
                obs = self.get_latest_obs()
                if obs:
                    ts_ns = obs.get("ts_ns", 0)
                    if ts_ns > 0:
                        last_obs_time = time.strftime("%H:%M:%S", time.localtime(ts_ns / 1e9))

                infer_status = "运行中" if self._continuous_infer else "未运行"
                print(f"\n[等待按键] 最新观测时间: {last_obs_time}, 已接收: {self._obs_count}, "
                      f"已推理: {self._infer_count}, 连续推理: {infer_status}")

                try:
                    user_input = input("按回车启动连续推理，输入 s 停止，输入 q 退出: ").strip()
                except EOFError:
                    break

                if user_input.lower() == 'q':
                    print("\n[退出] 正在关闭连接...")
                    break
                if user_input.lower() == 's':
                    self._stop_continuous_inference()
                    continue

                self._start_continuous_inference()

        except KeyboardInterrupt:
            print("\n[中断] 收到 Ctrl+C")
        finally:
            self._continuous_infer = False
            self._running = False
            if self._sock:
                self._sock.close()
            server_sock.close()
            print("[完成] 程序已退出")


# ════════════════════════════════════════════════════════════════
# 主程序入口
# ════════════════════════════════════════════════════════════════

def main():
    # 默认保存目录
    DEFAULT_SAVE_DIR = pathlib.Path("/home/dmh/xyj_zhipingfang/model3_openpi0.5/my_bot2_deployment/4090deployment/realtime_inference_results")

    parser = argparse.ArgumentParser(
        description="OpenPI 右臂8维真机实时推理 - 推理并下发动作",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint 目录路径"
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=9000,
        help="本地监听端口（默认: 9000）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备（默认: cuda）"
    )
    parser.add_argument(
        "--fixed-noise-seed",
        type=int,
        default=42,
        help="固定推理噪声 seed（用于调试重复性）"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(DEFAULT_SAVE_DIR),
        help=f"保存目录（默认: {DEFAULT_SAVE_DIR}）"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存推理结果"
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="夹爪二值化阈值：模型第8维输出 >= 阈值时下发闭合(1)，否则张开(0)（默认: 0.5）"
    )

    args = parser.parse_args()

    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()

    if not checkpoint_dir.exists():
        print(f"错误: Checkpoint 目录不存在: {checkpoint_dir}")
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，降级到 CPU")
        args.device = "cpu"

    # 保存目录
    save_dir = None if args.no_save else pathlib.Path(args.save_dir)

    # 1. 构建策略模型
    t_start_total = time.time()
    policy = build_policy(checkpoint_dir, DEFAULT_PROMPT, args.device, args.fixed_noise_seed)
    t_model_load = time.time()
    print(f"\n模型加载耗时: {t_model_load - t_start_total:.2f} 秒")

    # 2. 创建推理客户端
    print("\n" + "=" * 70)
    print("【启动真机实时推理模式】")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"监听端口: {args.listen_port}")
    print(f"保存目录: {save_dir if save_dir else '不保存'}")
    print(f"夹爪阈值: {args.gripper_threshold}")
    print("=" * 70)

    action_handler = PrintOnlyActionHandler(save_dir)
    client = RealtimeInferenceClient(
        policy=policy,
        action_handler=action_handler,
        save_dir=save_dir,
        gripper_threshold=args.gripper_threshold,
    )
    client.run(args.listen_port)


if __name__ == "__main__":
    main()
