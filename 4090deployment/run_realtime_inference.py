#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_realtime_inference.py — 真机实时推理客户端（仅头部相机版）

功能：
1. TCP服务器模式，接收机器人端发送的实时观测数据
2. 仅使用头部摄像头图像进行推理
3. 按键触发推理，使用最新观测数据
4. 发送动作到机器人端执行

使用方式：
    cd /home/dmh/xyj_zhipingfang/model3_openpi0.5/my_bot2_deployment/offline_inference
    source /home/dmh/xyj_zhipingfang/model3_openpi0.5/openpi/.venv/bin/activate
    python run_realtime_inference.py \
        --checkpoint-dir /home/dmh/xyj_zhipingfang/model3_openpi0.5/pi0_checkpoint_xin129tiao_headcam/30000 \
        --robot-host 192.168.1.100 \
        --listen-port 9000

键盘操作：
    按 Enter/回车: 按键触发持续推理模式（持续使用最新观测执行推理）
    输入 q: 退出程序

持续推理模式：
    按 Enter 后，程序会持续执行推理，每次使用最新的观测数据。
    新动作序列到达机器人端时会立即替换旧序列，从第0步开始执行。
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

import cv2
import numpy as np

# ════════════════════════════════════════════════════════════════
# 【关键】设置 Python 路径（必须在其他导入之前）
# ════════════════════════════════════════════════════════════════
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_OPENPI_ROOT = _SCRIPT_DIR.parent.parent / "openpi"
sys.path.insert(0, str(_OPENPI_ROOT / "src"))

# 设置 OPENPI_DATA_HOME（如果未设置）
if "OPENPI_DATA_HOME" not in os.environ:
    os.environ["OPENPI_DATA_HOME"] = str(_OPENPI_ROOT / ".openpi_cache")

# ════════════════════════════════════════════════════════════════
# 【关键】禁用 Triton 编译
# ════════════════════════════════════════════════════════════════
os.environ["PYTORCH_DISABLE_TRITON"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


DEFAULT_PROMPT = "right arm pick and place task"

JOINT_NAMES_8D = [
    "right_joint_0", "right_joint_1", "right_joint_2",
    "right_joint_3", "right_joint_4", "right_joint_5",
    "right_joint_6", "right_dexterous_hand"
]


# ════════════════════════════════════════════════════════════════
# 数据转换类（与原脚本相同）
# ════════════════════════════════════════════════════════════════

@dataclasses.dataclass(frozen=True)
class RightArmInputs(_transforms.DataTransformFn):
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

        # 仅使用头部相机，禁用腕部相机
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
    print("【构建策略 - 右臂8维】")
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

    transforms = [
        RightArmInputs(),
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
# 网络通信
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
        # 先读取4字节长度头
        header = _recv_exact(sock, 4)
        if header is None:
            return None
        length = struct.unpack(">I", header)[0]

        if length > 10 * 1024 * 1024:  # 超过10MB，认为是非法数据
            print(f"[RECV] 非法消息长度: {length}")
            return None

        # 读取payload
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


# ════════════════════════════════════════════════════════════════
# 打印函数
# ════════════════════════════════════════════════════════════════

def print_separator(title: str = ""):
    """打印分隔符"""
    if title:
        print(f"\n{'='*70}")
        print(f"【{title}】")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")


def print_state_info(state_full: np.ndarray):
    """打印状态信息"""
    print("\n[输入状态 - 右臂8维] (弧度)")
    for i, name in enumerate(JOINT_NAMES_8D):
        deg = state_full[i] * 180.0 / math.pi
        print(f"  {name}: {state_full[i]:.4f} ({deg:.3f}°)")


def print_inference_result(
    state_deg: np.ndarray,
    model_delta_deg: np.ndarray,
    target_action_deg: np.ndarray,
    inference_time: float,
):
    """打印推理结果"""
    print("\n" + "=" * 70)
    print("[推理结果 - 右臂8维]")
    print("=" * 70)

    print("  关节名称            │         当前状态 │          模型增量 │          目标角度")
    print("  " + "─" * 23 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14)

    for i, name in enumerate(JOINT_NAMES_8D):
        def fmt(val):
            return f"{val:+.3f}°"
        print(f"  {name:<21} │ {fmt(state_deg[i]):>10} │ {fmt(model_delta_deg[i]):>10} │ {fmt(target_action_deg[i]):>10}")

    print()
    print(f"  推理耗时: {inference_time*1000:.1f}ms")
    print("=" * 70)


# ════════════════════════════════════════════════════════════════
# 实时推理客户端
# ════════════════════════════════════════════════════════════════

class RealtimeInferenceClient:
    """实时推理客户端 - 接收机器人端观测，按键触发推理"""

    def __init__(self, policy, robot_host: str, robot_port: int,
                 save_debug_images: bool = True):
        self.policy = policy
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.save_debug_images = save_debug_images

        self._sock: Optional[socket.socket] = None
        self._running = False
        self._client_connected = False

        # 最新观测数据
        self._latest_obs = None
        self._obs_lock = threading.Lock()
        self._obs_count = 0

        # 推理统计
        self._infer_count = 0
        self._action_count = 0

        # 调试保存目录
        self._debug_dir = pathlib.Path("debug_inference")
        if self.save_debug_images:
            self._debug_dir.mkdir(exist_ok=True)

    def _accept_connection(self, server_sock: socket.socket):
        """接受机器人端连接"""
        print(f"\n[等待] 等待机器人端连接 {self.robot_host}:{self.robot_port}...")
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

                    # 每30帧打印一次状态
                    if self._obs_count % 30 == 0:
                        obs_seq = msg.get("obs_seq", 0)
                        ts_ns = msg.get("ts_ns", 0)
                        ts_sec = ts_ns / 1e9
                        joints_r = msg.get("joints_right", [])
                        images = msg.get("images", {})
                        head_ok = images.get("head") is not None
                        right_arm_ok = images.get("right_arm") is not None
                        left_arm_ok = images.get("left_arm") is not None

                        print(f"[接收 #{self._obs_count}] seq={obs_seq}, "
                              f"joints_r={joints_r[:3] if joints_r else 'N/A'}..., "
                              f"head={'OK' if head_ok else 'FAIL'}, "
                              f"right_arm={'OK' if right_arm_ok else 'FAIL'}, "
                              f"left_arm={'OK' if left_arm_ok else 'FAIL'}")

            except Exception as e:
                print(f"[接收] 接收异常: {e}")
                time.sleep(0.1)

    def get_latest_obs(self):
        """获取最新观测数据"""
        with self._obs_lock:
            return self._latest_obs

    def run_inference(self):
        """使用最新观测执行推理并发送动作"""
        self._infer_count += 1
        infer_num = self._infer_count

        print(f"\n{'='*70}")
        print(f"【推理 #{infer_num}】")
        print(f"{'='*70}")

        obs = self.get_latest_obs()
        if obs is None:
            print("[错误] 等待机器人端观测数据...")
            return None

        # ========== 调试信息：原始观测数据 ==========
        obs_seq = obs.get("obs_seq", 0)
        ts_ns = obs.get("ts_ns", 0)
        joints_right = obs.get("joints_right", [])
        dexhand_right = obs.get("dexhand_right", 0.0)
        images_raw = obs.get("images", {})

        print(f"\n[观测原始数据]")
        print(f"  obs_seq: {obs_seq}")
        print(f"  时间戳: {ts_ns} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_ns/1e9))})")
        print(f"  joints_right (度): {[f'{x:.2f}' for x in joints_right]}")
        print(f"  dexhand_right: {dexhand_right}")
        print(f"  图像keys: {list(images_raw.keys())}")
        print(f"  head图像: {'有' if images_raw.get('head') else '无'} ({len(images_raw.get('head', ''))//1024}KB)")
        print(f"  right_arm图像: {'有' if images_raw.get('right_arm') else '无'} ({len(images_raw.get('right_arm', ''))//1024}KB)")
        print(f"  left_arm图像: {'有' if images_raw.get('left_arm') else '无'}")

        # ========== 解码图像 ==========
        head_img = decode_image(images_raw.get("head"))

        print(f"\n[图像解码]")
        print(f"  head_img: {'成功' if head_img is not None else '失败'} {head_img.shape if head_img is not None else ''}")

        if head_img is None:
            print("[错误] 头部图像数据无效")
            return None

        # ========== 保存输入图像 ==========
        if self.save_debug_images:
            head_path = self._debug_dir / f"infer{infer_num:03d}_head.jpg"
            cv2.imwrite(str(head_path), head_img)
            print(f"\n[保存] 输入图像已保存到 {self._debug_dir}/")
            print(f"  - {head_path.name}")

        # ========== 获取关节状态 ==========
        if len(joints_right) < 7:
            print(f"[错误] 关节数据不足: {joints_right}")
            return None

        # 机械臂发送的是度数，需要转换为弧度给模型
        joints_right_rad = [deg * math.pi / 180.0 for deg in joints_right[:7]]

        # 构造8维状态（弧度）
        state_full = np.array(joints_right_rad + [float(dexhand_right)], dtype=np.float32)

        # ========== 打印输入状态 ==========
        print(f"\n[模型输入状态 - 8维右臂]")
        for i, name in enumerate(JOINT_NAMES_8D):
            deg = state_full[i] if i == 7 else state_full[i] * 180.0 / math.pi
            print(f"  {name}: {state_full[i]:.4f} rad ({deg:.3f}°)")

        # ========== 构造模型输入 ==========
        # 注意：cv2.imdecode 返回 BGR，转为 RGB
        # 只使用头部相机图像
        model_obs = {
            "state": state_full,
            "images": {
                "cam_high": cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB),
            },
            "prompt": DEFAULT_PROMPT,
        }

        # ========== 调试：验证图像内容 ==========
        if self.save_debug_images:
            # 保存 resize 前的图像
            head_orig = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)

            # 模拟 resize (使用 PIL like the model does)
            from PIL import Image as PILImage
            def resize_with_pad_pil(img, h, w):
                pil_img = PILImage.fromarray(img)
                ratio = max(pil_img.width / w, pil_img.height / h)
                new_w = int(pil_img.width / ratio)
                new_h = int(pil_img.height / ratio)
                pil_resized = pil_img.resize((new_w, new_h), PILImage.BILINEAR)
                zero_img = PILImage.new('RGB', (w, h), 0)
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                zero_img.paste(pil_resized, (pad_x, pad_y))
                return np.array(zero_img)

            head_resized = resize_with_pad_pil(head_orig, 224, 224)

            # 保存 resize 后的图像用于对比
            cv2.imwrite(str(self._debug_dir / f"infer{infer_num:03d}_head_224.jpg"), cv2.cvtColor(head_resized, cv2.COLOR_RGB2BGR))

            # 打印图像统计信息
            print(f"\n[图像统计]")
            print(f"  head原始: shape={head_orig.shape}, mean={head_orig.mean():.1f}, min={head_orig.min()}, max={head_orig.max()}")
            print(f"  head_224: shape={head_resized.shape}, mean={head_resized.mean():.1f}, min={head_resized.min()}, max={head_resized.max()}")

        # ========== 执行推理 ==========
        print(f"\n[推理] 执行推理中...")
        t_infer_start = time.time()
        result = self.policy.infer(model_obs)
        inference_time = time.time() - t_infer_start
        print(f"[推理] 完成，耗时: {inference_time*1000:.1f}ms")

        # ========== 解析输出 ==========
        # 注意：result["actions"] shape 为 (50, 8)，包含50步动作序列
        all_actions = result["actions"]  # shape: (50, 8)

        # 构建50步动作序列
        action_sequence = []
        for i in range(50):
            model_action = np.asarray(all_actions[i], dtype=np.float32)
            target_action_deg = model_action * 180.0 / np.pi
            target_action_deg[7] = model_action[7]  # 夹爪是0-1值

            action_sequence.append({
                "joints_right": target_action_deg[:7].tolist(),
                "dexhand_right": 1.0 if float(np.clip(model_action[7], 0.0, 1.0)) > 0.5 else 0.0,
            })

        # ========== 打印推理结果 ==========
        print(f"\n{'='*70}")
        print(f"[推理结果 #{infer_num}] - 50步动作序列")
        print(f"{'='*70}")
        print(f"\n  第1步目标关节角度: {[f'{x:.2f}' for x in action_sequence[0]['joints_right']]}")
        print(f"  第25步目标关节角度: {[f'{x:.2f}' for x in action_sequence[24]['joints_right']]}")
        print(f"  第50步目标关节角度: {[f'{x:.2f}' for x in action_sequence[49]['joints_right']]}")
        print(f"\n  推理耗时: {inference_time*1000:.1f}ms")
        print(f"  动作序列长度: {len(action_sequence)} 步")

        # ========== 构建动作序列消息 ==========
        action_msg = {
            "msg_type": "action_sequence",
            "obs_seq": obs_seq,
            "actions": action_sequence,
            "infer_ms": inference_time * 1000,
        }

        # ========== 打印即将发送的动作序列 ==========
        print(f"\n{'='*70}")
        print(f"[发送动作序列 #{infer_num}]")
        print(f"{'='*70}")
        print(f"  obs_seq: {action_msg['obs_seq']}")
        print(f"  动作步数: {len(action_msg['actions'])}")
        print(f"  第1步: joints_right={[f'{x:.2f}' for x in action_msg['actions'][0]['joints_right']]}")
        print(f"  第1步: dexhand_right={action_msg['actions'][0]['dexhand_right']}")
        print(f"  infer_ms: {action_msg['infer_ms']:.1f}")
        print(f"  (左臂保持原位，不执行)")

        # ========== 发送动作序列 ==========
        if self._sock:
            try:
                send_msg(self._sock, action_msg)
                self._action_count += 1
                print(f"\n[发送成功] ✓ (动作序列#{self._action_count})")
                print(f"{'='*70}")
            except Exception as e:
                print(f"[发送失败] {e}")
                return None
        else:
            print("[错误] 未连接到机器人端")
            return None

        return action_msg

    def run(self, listen_port: int):
        """运行推理客户端"""
        print("\n" + "=" * 70)
        print("【OpenPI 真机实时推理客户端】")
        print("=" * 70)
        print(f"监听端口: {listen_port}")
        print(f"机器人地址: {self.robot_host}:{self.robot_port}")
        print("=" * 70)

        # 创建TCP服务器
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", listen_port))
        server_sock.listen(1)
        print(f"[启动] TCP服务器已启动，监听端口 {listen_port}")

        self._running = True

        # 启动接收线程
        recv_thread = threading.Thread(target=self._receive_obs_loop, daemon=True)
        recv_thread.start()

        # 等待机器人端连接
        self._accept_connection(server_sock)

        # 键盘控制循环
        print_separator("键盘控制模式")
        print("操作说明:")
        print("  按 Enter/回车: 触发持续推理模式（持续使用最新观测执行推理）")
        print("  输入 q: 退出程序")
        print("=" * 70)

        last_obs_time = "无"
        last_action = None
        continuous_mode = False  # 是否处于持续推理模式
        continuous_count = 0    # 持续推理计数

        try:
            while self._running:
                # 显示最新观测状态
                obs = self.get_latest_obs()
                if obs:
                    ts_ns = obs.get("ts_ns", 0)
                    if ts_ns > 0:
                        last_obs_time = time.strftime("%H:%M:%S", time.localtime(ts_ns / 1e9))

                if continuous_mode:
                    print(f"\n[持续推理模式] 推理#{continuous_count+1}, 观测时间: {last_obs_time}, 已发送: {self._action_count}")
                else:
                    print(f"\n[等待按键] 最新观测时间: {last_obs_time}, 已接收: {self._obs_count}, 已推理: {self._infer_count}, 已发送: {self._action_count}")

                try:
                    user_input = input("按回车触发/停止持续推理，或输入 q 退出: ").strip()
                except EOFError:
                    break

                # 退出
                if user_input.lower() == 'q':
                    print("\n[退出] 正在关闭连接...")
                    break

                # 切换持续推理模式
                if not continuous_mode:
                    continuous_mode = True
                    continuous_count = 0
                    print("\n[持续推理模式] 已启动，按 Enter 停止...")

                    # 持续推理循环
                    while continuous_mode and self._running:
                        # 执行一次推理
                        try:
                            action_msg = self.run_inference()
                            if action_msg:
                                last_action = action_msg
                                continuous_count += 1
                                # 不等待，持续推理
                                if continuous_count % 10 == 0:
                                    print(f"[持续推理] 已完成 {continuous_count} 次推理")
                        except Exception as e:
                            print(f"[错误] 推理失败: {e}")
                            import traceback
                            traceback.print_exc()
                            time.sleep(0.5)  # 失败后短暂休眠

                        # 检查是否还有等待输入（非阻塞）
                        import select
                        if select.select([sys.stdin], [], [], 0.0)[0]:
                            try:
                                user_input = input().strip()
                                if user_input.lower() == 'q':
                                    continuous_mode = False
                                    self._running = False
                                    break
                                else:
                                    # 其他输入停止持续推理
                                    continuous_mode = False
                                    print("\n[持续推理模式] 已停止")
                                    break
                            except EOFError:
                                break
                else:
                    continuous_mode = False
                    print("\n[持续推理模式] 已停止")

        except KeyboardInterrupt:
            print("\n[中断] 收到 Ctrl+C")
        finally:
            self._running = False
            if self._sock:
                self._sock.close()
            server_sock.close()
            print("[完成] 程序已退出")


# ════════════════════════════════════════════════════════════════
# 主程序入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OpenPI 真机实时推理客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint 目录路径"
    )
    parser.add_argument(
        "--robot-host",
        type=str,
        default="192.168.1.100",
        help="机器人IP（默认: 192.168.1.100）"
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

    args = parser.parse_args()

    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()

    if not checkpoint_dir.exists():
        print(f"错误: Checkpoint 目录不存在: {checkpoint_dir}")
        sys.exit(1)

    # GPU 检查
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，降级到 CPU")
        args.device = "cpu"

    # 1. 构建策略模型
    t_start_total = time.time()
    policy = build_policy(checkpoint_dir, DEFAULT_PROMPT, args.device)
    t_model_load = time.time()
    print(f"\n模型加载耗时: {t_model_load - t_start_total:.2f} 秒")

    # 2. 创建推理客户端
    client = RealtimeInferenceClient(
        policy=policy,
        robot_host=args.robot_host,
        robot_port=args.listen_port,  # 注意：端口复用
    )

    # 3. 运行客户端
    client.run(args.listen_port)


if __name__ == "__main__":
    main()