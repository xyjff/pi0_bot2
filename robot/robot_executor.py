# -*- coding: utf-8 -*-
"""
robot_executor.py — 机器人端执行器（实时动作序列版）

功能：
  - 三线程架构: 观测发送/动作接收/执行
  - 观测发送线程: 60Hz采集摄像头+关节数据，主动发送到4090端
  - 动作接收线程: 接收4090端推理结果（50步动作序列）
  - 执行线程: 每10ms取一步动作执行（1度限幅，仅右臂）

关键特性：
  - 观测发送频率: 60Hz
  - 动作执行频率: 100Hz (每10ms一步)
  - 动作序列长度: 50步
  - 新动作覆盖机制: 收到新动作时立即丢弃旧序列
  - 单步限幅: 1度
  - 仅控制右臂

运行（机器人端）:
  cd /path/to/robot
  pip install -r requirements.txt
  python robot_executor.py --inference-host 192.168.1.101 --inference-port 9000

使用说明:
  1. 先启动4090端推理脚本
  2. 再启动此脚本（机器人端主动连接4090）
  3. 机器人端持续发送60Hz观测数据
  4. 4090端按Enter触发持续推理后，机器人执行动作
"""
import argparse
import base64
import json
import logging
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2
import numpy as np

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "robot_executor.log"

# 配置日志：同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)

# 抑制第三方库的冗余日志
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ============================================================
# 工具函数
# ============================================================
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
            logger.error(f"[RECV] 非法消息长度: {length}")
            return None

        # 读取payload
        payload = _recv_exact(sock, length)
        if payload is None:
            return None

        msg = json.loads(payload.decode("utf-8"))
        return msg
    except json.JSONDecodeError as e:
        logger.error(f"[RECV] JSON解析失败: {e}")
        return None
    except Exception as e:
        logger.warning(f"[RECV] 接收消息异常: {e}")
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
            # 没有数据，连接正常，继续
            continue
        except Exception as e:
            logger.error(f"[RECV] _recv_exact 失败: {e}")
            return None
    return buf

def encode_image(image, quality: int = 85) -> Optional[str]:
    """将numpy图像编码为base64 JPEG"""
    if image is None:
        return None
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")

def format_joints(joints, precision=1):
    """格式化关节角度为短字符串"""
    if joints is None:
        return "None"
    return "[" + ", ".join(f"{x:.1f}" for x in joints[:3]) + ", ...]"

def format_images_info(images):
    """格式化图像状态信息"""
    if not images:
        return "empty"
    parts = []
    for k, v in images.items():
        if v is not None:
            parts.append(f"{k}=OK({len(v)//1024}KB)")
        else:
            parts.append(f"{k}=FAIL")
    return ", ".join(parts)

# ============================================================
# 线程安全的数据缓冲区
# ============================================================
class DataBuffer:
    """线程安全的数据缓冲区（FIFO队列）"""
    def __init__(self, name: str = "buffer", max_size: int = 100):
        self._queue = []  # 使用队列存储多个数据
        self._lock = threading.Lock()
        self._name = name
        self._put_count = 0
        self._get_count = 0
        self._max_size = max_size

    def put(self, data) -> bool:
        """存入数据，返回是否成功"""
        with self._lock:
            self._queue.append(data)
            # 如果队列太长，移除最旧的数据
            while len(self._queue) > self._max_size:
                self._queue.pop(0)
            self._put_count += 1
            return True

    def get(self):
        """取出最早的数据，返回(data, timestamp_ns)"""
        with self._lock:
            if not self._queue:
                self._get_count += 1
                return None, 0
            data = self._queue.pop(0)
            self._get_count += 1
            return data, time.time_ns()

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def size(self) -> int:
        """返回队列中的数据数量"""
        with self._lock:
            return len(self._queue)

    def get_stats(self):
        """获取缓冲区统计信息"""
        with self._lock:
            return {
                "name": self._name,
                "size": len(self._queue),
                "put_count": self._put_count,
                "get_count": self._get_count,
            }

    def get_latest(self):
        """获取最新存入的数据"""
        with self._lock:
            if not self._queue:
                return None, 0
            data = self._queue[-1]
            return data, time.time_ns()

    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._queue.clear()


# ============================================================
# 动作序列缓存池 - 支持新动作实时覆盖
# ============================================================
class ActionSequenceBuffer:
    """
    50步动作序列缓存池 - 支持新动作实时覆盖

    核心设计：
    1. 存储50步动作序列
    2. 每10ms取一步执行
    3. 新动作序列到达时，立即清除旧序列，从第0步开始执行
    4. 无论旧序列执行到哪一步，都丢弃
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._actions = []  # 存储50步动作
        self._current_step = 0  # 当前执行到第几步
        self._has_new_sequence = False  # 标记是否有新序列

    def set_sequence(self, actions):
        """
        设置新的动作序列（立即清除旧序列，即使50步未执行完）

        关键：无论旧序列执行到哪一步，都丢弃，从新序列第0步开始执行
        """
        with self._lock:
            self._actions = list(actions)  # 深拷贝，避免引用问题
            self._current_step = 0
            self._has_new_sequence = True

    def get_next_action(self):
        """获取下一步动作"""
        with self._lock:
            if self._current_step < len(self._actions):
                action = self._actions[self._current_step]
                self._current_step += 1
                return action
            self._has_new_sequence = False
            return None

    def has_pending_actions(self):
        """检查是否还有待执行的动作"""
        with self._lock:
            return self._current_step < len(self._actions)

    def get_remaining_steps(self):
        """获取剩余步数"""
        with self._lock:
            return len(self._actions) - self._current_step

    def get_stats(self):
        """获取统计信息"""
        with self._lock:
            return {
                "total_steps": len(self._actions),
                "current_step": self._current_step,
                "remaining": len(self._actions) - self._current_step,
                "has_sequence": self._has_new_sequence,
            }

# ============================================================
# 机械臂客户端
# ============================================================
class ArmClient:
    """机械臂客户端 - TCP控制 + UDP状态监听"""
    def __init__(self, key: str, ip: str, udp_port: int):
        self.key = key
        self.ip = ip
        self.udp_port = udp_port
        self._lock = threading.Lock()
        self._tcp = None
        self._udp = None
        self._running = False
        self._thread = None
        self._buf = ""
        self._joints_deg: Optional[List[float]] = None
        self._gripper: Optional[float] = None
        self._gripper_online = False
        self._dexhand_dof: int = 6  # 灵巧手自由度，默认6

    def _recv_json(self, timeout: float) -> Optional[dict]:
        self._tcp.settimeout(timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = self._tcp.recv(4096).decode("utf-8", errors="ignore")
            except socket.timeout:
                continue
            if not chunk:
                continue
            self._buf += chunk
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip().rstrip("\r")
                if not line:
                    continue
                try:
                    return json.loads(line)
                except Exception:
                    pass
        return None

    def _send(self, cmd: dict, timeout: float = 3.0) -> Optional[dict]:
        self._tcp.send((json.dumps(cmd) + "\r\n").encode("utf-8"))
        return self._recv_json(timeout)

    def _udp_loop(self):
        while self._running:
            try:
                data, _ = self._udp.recvfrom(65535)
                msg = json.loads(data.decode("utf-8", errors="ignore"))
                joints_raw = msg["joint_status"]["joint_position"]
                joints_deg = [x / 1000.0 for x in joints_raw]
                grip = None
                online = False
                dof = 6
                rp = msg.get("rm_plus_state")
                if isinstance(rp, dict) and rp.get("sys_state") != "offline":
                    online = True
                    pos = rp.get("pos", [])
                    if isinstance(pos, list) and len(pos) > 0:
                        grip = float(pos[0])
                        dof = len(pos)
                with self._lock:
                    self._joints_deg = joints_deg
                    self._gripper_online = online
                    self._dexhand_dof = dof
                    if grip is not None:
                        self._gripper = grip
            except socket.timeout:
                continue
            except Exception:
                continue

    def connect(self, local_ip: str, arm_tcp_port: int, wait_s: float = 6.0):
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp.settimeout(1.0)
        self._udp.bind((local_ip, self.udp_port))
        self._running = True
        self._thread = threading.Thread(target=self._udp_loop, daemon=True)
        self._thread.start()
        self._tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp.settimeout(3.0)
        self._tcp.connect((self.ip, arm_tcp_port))
        self._send({
            "command": "set_realtime_push",
            "cycle": 5,
            "enable": True,
            "port": self.udp_port,
            "force_coordinate": 2,
            "ip": local_ip,
            "custom": {
                "joint_speed": True,
                "arm_current_status": True,
                "hand": True,
                "rm_plus_base": True,
                "rm_plus_state": True,
            },
        }, timeout=2.0)
        # 设置灵巧手波特率
        self._send({"command": "set_rm_plus_mode", "mode": 115200}, timeout=2.0)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.joints() is not None:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"[{self.key}] 超时：未收到关节UDP数据")

    def disconnect(self):
        self._running = False
        for s in (self._udp, self._tcp):
            try:
                if s:
                    s.close()
            except Exception:
                pass

    def joints(self) -> Optional[List[float]]:
        with self._lock:
            return list(self._joints_deg) if self._joints_deg else None

    def gripper(self) -> Optional[float]:
        with self._lock:
            return float(self._gripper) if self._gripper is not None else None

    def gripper_online(self) -> bool:
        with self._lock:
            return self._gripper_online

    def dexhand_dof(self) -> int:
        with self._lock:
            return self._dexhand_dof

    def movej_nowait(self, joints_deg: List[float], repeats: int = 3) -> None:
        """发送 movej_canfd 命令，可选择重复发送多次以确保命令被接收"""
        cmd = json.dumps({
            "command": "movej_canfd",
            "joint": [int(round(x * 1000.0)) for x in joints_deg],
            "follow": True,
            "expand": 0,
            "trajectory_mode": 0,
            "radio": 0,
        }) + "\r\n"
        encoded_cmd = cmd.encode("utf-8")
        
        def _fire():
            try:
                # 连续发送多次，确保命令被机械臂接收
                for i in range(repeats):
                    self._tcp.send(encoded_cmd)
                    # 发送间隔极短（微秒级），连续下发
                    if i < repeats - 1:
                        time.sleep(0.002)  # 2ms 间隔
            except Exception:
                pass
        threading.Thread(target=_fire, daemon=True).start()

    def set_gripper_nowait(self, pos: int) -> None:
        dof = self.dexhand_dof()
        # 灵巧手控制逻辑：
        # ch0 (拇指) = 0 → 张开
        # ch1-4 (食指~小指) = pos → 闭合/张开
        # ch5 (拇指旋转) = 0 → 张开
        hand_pos = [0] * dof
        if dof >= 5:
            hand_pos[0] = 0  # 拇指始终张开
            for i in range(1, min(dof - 1, 5)):  # ch1-ch4
                hand_pos[i] = int(max(0, min(65535, pos)))
            hand_pos[min(dof - 1, 5)] = 0  # 拇指旋转始终张开
        cmd = json.dumps({
            "command": "hand_follow_pos",
            "hand_pos": hand_pos,
        }) + "\r\n"
        def _fire():
            try:
                self._tcp.send(cmd.encode("utf-8"))
            except Exception:
                pass
        threading.Thread(target=_fire, daemon=True).start()

# ============================================================
# 相机管理
# ============================================================
# 机器人相机配置（硬编码）
CAMERA_CONFIG = {
    # 相机SN映射
    "camera_sn": {
        "left_arm": "353322271325",    # 左臂 D405
        "right_arm": "353322271272",   # 右臂 D405
        "head": "346222070837",        # 头部 D435
    },
    # 默认相机参数
    "resolution": {"width": 640, "height": 480},
    "fps": 30,
}


class CameraRig:
    """四路RealSense相机"""
    def __init__(self, config_json: str = None):
        self.config_json = config_json
        self.cameras = []
        self._rs = None

    def start(self) -> int:
        import pyrealsense2 as rs
        self._rs = rs

        # 使用硬编码的相机配置
        role_sn = CAMERA_CONFIG["camera_sn"]
        width = CAMERA_CONFIG["resolution"]["width"]
        height = CAMERA_CONFIG["resolution"]["height"]
        fps = CAMERA_CONFIG["fps"]

        logger.info(f"[Camera] 分辨率: {width}x{height}, FPS: {fps}")
        logger.info(f"[Camera] 相机SN映射: {role_sn}")

        # 获取当前连接的设备
        ctx = rs.context()
        devices = []
        for d in ctx.query_devices():
            try:
                sn = d.get_info(rs.camera_info.serial_number)
                name = d.get_info(rs.camera_info.name)
                devices.append({"sn": sn, "name": name})
                logger.info(f"[Camera] 发现设备: SN={sn}, Name={name}")
            except Exception:
                continue

        # 启动配置的相机
        visible = {x["sn"] for x in devices}
        used = set()

        for role, sn in role_sn.items():
            if sn not in visible:
                logger.warning(f"[Camera] 角色 {role} 的相机 SN={sn} 未连接，跳过")
                continue

            if sn in used:
                continue

            try:
                pipeline = rs.pipeline()
                rs_cfg = rs.config()
                rs_cfg.enable_device(sn)
                rs_cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                pipeline.start(rs_cfg)
                self.cameras.append({"name": role, "sn": sn, "pipeline": pipeline})
                used.add(sn)
                logger.info(f"[Camera] 相机启动成功: {role} (SN={sn})")
            except Exception as e:
                logger.error(f"[Camera] 相机启动失败: {role} (SN={sn}): {e}")

        logger.info(f"[Camera] 共启动 {len(self.cameras)}/{len(role_sn)} 个相机")
        return fps

    def capture_once(self) -> Dict[str, Optional[np.ndarray]]:
        images = {}
        lock = threading.Lock()
        def _grab(cam):
            try:
                fs = cam["pipeline"].wait_for_frames(timeout_ms=1200)
                color = fs.get_color_frame()
                with lock:
                    images[cam["name"]] = np.asanyarray(color.get_data()) if color else None
            except Exception:
                with lock:
                    images[cam["name"]] = None
        threads = [threading.Thread(target=_grab, args=(c,), daemon=True) for c in self.cameras]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=1.5)
        for cam in self.cameras:
            if cam["name"] not in images:
                images[cam["name"]] = None
        return images

    def stop(self):
        for cam in self.cameras:
            try:
                cam["pipeline"].stop()
            except Exception:
                pass

# ============================================================
# 机器人执行器 - 三线程架构（实时观测版）
# ============================================================
class RobotExecutor:
    """机器人端执行器 - 实时动作序列执行器

    线程分工：
    1. OBS_SENDER: 60Hz采集观测数据，发送到4090端
    2. ACTION_RECV: 接收4090端推理结果（50步动作序列）
    3. EXEC: 每10ms取一步动作执行（1度限幅，仅右臂）

    关键特性：
    - 观测发送频率: 60Hz
    - 动作执行频率: 100Hz (每10ms一步)
    - 动作序列长度: 50步
    - 新动作覆盖机制: 收到新动作时立即丢弃旧序列
    - 单步限幅: 1度
    - 仅控制右臂
    """
    def __init__(
        self,
        inference_host: str = "192.168.1.101",
        inference_port: int = 9000,
        local_ip: str = "192.168.1.100",
        arm_right_ip: str = "192.168.1.19",
        arm_tcp_port: int = 8080,
        arm_right_udp_port: int = 18089,
        max_step_deg: float = 1.0,
        jpeg_quality: int = 85,
        connect_delay: int = 60,
        send_obs_hz: float = 60.0,
    ):
        self.inference_host = inference_host
        self.inference_port = inference_port
        self.local_ip = local_ip
        self.arm_right_ip = arm_right_ip
        self.arm_tcp_port = arm_tcp_port
        self.arm_right_udp_port = arm_right_udp_port
        self.max_step_deg = max_step_deg  # 限幅角度，默认1度
        self.jpeg_quality = jpeg_quality
        self.connect_delay = connect_delay
        self.send_obs_hz = send_obs_hz  # 观测发送频率，默认60Hz

        # 硬件客户端
        self._right_arm: Optional[ArmClient] = None  # 仅右臂
        self._rig: Optional[CameraRig] = None
        self._sock: Optional[socket.socket] = None

        # 动作序列缓存池（关键！）
        self._action_sequence_buffer = ActionSequenceBuffer()

        # 序列号
        self._obs_seq = 0
        self._action_seq = 0

        # 线程安全缓冲区
        self._obs_buffer = DataBuffer("obs")

        # 状态
        self._running = False
        self._threads = []

        # 统计
        self._send_count = 0
        self._recv_count = 0
        self._exec_count = 0
        self._obs_send_count = 0  # 观测发送计数
        self._sequence_count = 0  # 动作序列计数

    def connect_hardware(self):
        """连接机械臂、相机和推理服务器"""
        logger.info("=" * 60)
        logger.info("连接硬件设备...")
        logger.info("=" * 60)

        # 仅连接右臂
        c = ArmClient("right", self.arm_right_ip, self.arm_right_udp_port)
        logger.info(f"[CONNECT] 连接右臂 ({self.arm_right_ip})...")
        c.connect(self.local_ip, self.arm_tcp_port)
        self._right_arm = c
        logger.info(f"[CONNECT] 右臂连接成功")

        # 初始化相机（使用硬编码配置）
        self._rig = CameraRig()
        fps = self._rig.start()
        logger.info(f"[CONNECT] 相机启动完成, FPS={fps}")

        logger.info("[CONNECT] 等待夹爪初始化...")
        time.sleep(2.0)

        # 获取右臂状态
        joints = self._right_arm.joints()
        gripper = self._right_arm.gripper()
        online = self._right_arm.gripper_online()
        logger.info(f"[CONNECT] [右臂] 关节={format_joints(joints)}, 夹爪online={online}, gripper={gripper}")

        # 连接推理服务器
        logger.info(f"[CONNECT] 连接推理服务器 {self.inference_host}:{self.inference_port}...")
        self._connect_to_inference_server()
        logger.info("[CONNECT] 推理服务器连接成功")

    def _connect_to_inference_server(self):
        """连接到4090推理服务器（带重试机制）"""
        retry_count = 0
        start_time = time.time()

        while True:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(5.0)
                self._sock.connect((self.inference_host, self.inference_port))
                logger.info(f"[CONNECT] 推理服务器连接成功! (重试次数: {retry_count})")
                return
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                retry_count += 1
                elapsed = time.time() - start_time
                if elapsed >= self.connect_delay:
                    logger.error(f"[CONNECT] 推理服务器连接超时 ({self.connect_delay}秒): {e}")
                    raise TimeoutError(f"无法连接到推理服务器 {self.inference_host}:{self.inference_port}")
                logger.warning(f"[CONNECT] 推理服务器连接失败 (已等待 {elapsed:.0f}s/{self.connect_delay}s)，重试中... (#{retry_count})")
                time.sleep(1.0)

    # ============================================================
    # 工具方法：1度限幅
    # ============================================================
    def _clamp_joints(self, target_joints: List[float], current_joints: List[float]) -> List[float]:
        """
        限幅：确保每个关节变化不超过 max_step_deg 度

        Args:
            target_joints: 目标关节角度
            current_joints: 当前关节角度

        Returns:
            限幅后的目标关节角度
        """
        clamped = []
        for target, current in zip(target_joints, current_joints):
            diff = target - current
            if diff > self.max_step_deg:
                clamped.append(current + self.max_step_deg)
            elif diff < -self.max_step_deg:
                clamped.append(current - self.max_step_deg)
            else:
                clamped.append(target)
        return clamped

    def _execute_single_action(self, action: dict) -> dict:
        """
        执行单步动作（带1度限幅，仅右臂）

        Args:
            action: 单步动作，包含 joints_right 和 dexhand_right

        Returns:
            执行结果
        """
        exec_result = {
            "joints_r_sent": None,
            "gripper_r_sent": None,
        }

        # 获取当前关节位置
        current_r = self._right_arm.joints()
        if current_r is None:
            return exec_result

        # 获取目标关节
        target_joints_r = action.get("joints_right")
        if target_joints_r is None or len(target_joints_r) < 7:
            return exec_result

        # 1度限幅
        clamped_joints = self._clamp_joints(target_joints_r[:7], current_r)

        # 执行右臂动作
        self._right_arm.movej_nowait(clamped_joints)
        exec_result["joints_r_sent"] = clamped_joints

        # 控制右灵巧手
        dexhand_r = action.get("dexhand_right")
        if dexhand_r is not None and self._right_arm.gripper_online():
            grip_raw = 65535 if float(dexhand_r) >= 0.5 else 0
            self._right_arm.set_gripper_nowait(grip_raw)
            exec_result["gripper_r_sent"] = grip_raw

        return exec_result

    def _cleanup(self):
        """清理资源"""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def stop(self):
        """停止执行器"""
        self._running = False
        if self._right_arm:
            self._right_arm.disconnect()
            self._right_arm = None
        if self._rig:
            self._rig.stop()
        self._cleanup()
        logger.info("[STOP] 机器人执行器已停止")

    # ============================================================
    # 线程1: 观测发送线程 - 60Hz采集并发送观测数据到4090端
    # ============================================================
    def _obs_sender_thread_worker(self):
        """观测发送线程: 60Hz采集摄像头+关节数据，发送到4090端"""
        logger.info(f"[OBS_SENDER] 观测发送线程启动 (目标频率: {self.send_obs_hz}Hz)")
        period_s = 1.0 / self.send_obs_hz

        while self._running:
            try:
                # 采集右臂关节数据
                joints_r = self._right_arm.joints()

                if joints_r is None:
                    logger.warning("[OBS_SENDER] 关节数据不可用，等待...")
                    time.sleep(period_s)
                    continue

                # 采集相机图像
                images = {}
                if self._rig:
                    images = self._rig.capture_once()
                else:
                    images = {"head": None, "right_arm": None, "left_arm": None}

                # 获取灵巧手状态
                rp = self._right_arm.gripper()
                dexhand_right = 1.0 if (rp is not None and rp >= 32767.5) else 0.0

                self._obs_seq += 1
                obs_msg = {
                    "msg_type": "observation",
                    "obs_seq": self._obs_seq,
                    "ts_ns": int(time.time_ns()),
                    "joints_right": joints_r,
                    "dexhand_right": float(dexhand_right),
                    "images": {
                        "head": encode_image(images.get("head"), self.jpeg_quality),
                        "right_arm": encode_image(images.get("right_arm"), self.jpeg_quality),
                    },
                }

                # 发送到4090端
                send_msg(self._sock, obs_msg)
                self._obs_send_count += 1

                if self._obs_send_count % 100 == 0:
                    logger.info(f"[OBS_SENDER] 已发送 {self._obs_send_count} 帧观测数据")

                time.sleep(period_s)

            except Exception as e:
                logger.error(f"[OBS_SENDER] 采集/发送观测失败: {e}")
                time.sleep(0.1)

    # ============================================================
    # 线程2: 动作接收线程 - 接收50步动作序列
    # ============================================================
    def _recv_thread_worker(self):
        """动作接收线程: 接收4090端推理结果（50步动作序列）"""
        logger.info("[ACTION_RECV] 动作接收线程启动（50步动作序列模式）")

        # 等待连接建立
        while self._running and self._sock is None:
            time.sleep(0.1)

        # 循环接收动作
        while self._running:
            if self._sock is None:
                time.sleep(0.1)
                continue

            try:
                msg = recv_msg(self._sock, timeout=2.0)
                if msg is None:
                    continue

                if msg.get("msg_type") == "action_sequence":
                    # 收到动作序列，立即更新缓存池（覆盖旧序列）
                    actions = msg.get("actions", [])
                    self._action_sequence_buffer.set_sequence(actions)
                    self._recv_count += 1
                    self._sequence_count += 1

                    # 打印日志
                    stats = self._action_sequence_buffer.get_stats()
                    logger.info(f"[ACTION_RECV] 收到动作序列 #{self._sequence_count}, "
                                f"obs_seq={msg.get('obs_seq', 0)}, "
                                f"步数={len(actions)}, "
                                f"infer_ms={msg.get('infer_ms', 0):.1f}")

            except Exception as e:
                logger.error(f"[ACTION_RECV] 接收动作异常: {e}")
                time.sleep(0.05)

    # ============================================================
    # 线程3: 执行线程 - 每10ms取一步动作执行（1度限幅）
    # ============================================================
    def _exec_thread_worker(self):
        """执行线程: 从动作序列缓存池取动作，每10ms执行一步（1度限幅）"""
        logger.info("[EXEC] 执行线程启动（每10ms一步，1度限幅，仅右臂）")
        exec_period_ms = 0.01  # 10ms

        while self._running:
            try:
                # 从缓存池取下一步动作
                action = self._action_sequence_buffer.get_next_action()

                if action is None:
                    # 无动作时短暂休眠
                    time.sleep(0.005)
                    continue

                # 执行动作（带1度限幅）
                exec_result = self._execute_single_action(action)
                self._exec_count += 1

                # 每50步打印一次状态
                if self._exec_count % 50 == 0:
                    stats = self._action_sequence_buffer.get_stats()
                    logger.info(f"[EXEC] 执行 #{self._exec_count}, "
                                f"剩余待执行: {stats['remaining']}/{stats['total_steps']}步")

                # 固定10ms周期
                time.sleep(exec_period_ms)

            except Exception as e:
                logger.error(f"[EXEC] 执行动作异常: {e}")
                time.sleep(0.005)

    # ============================================================
    # 主循环 - 三线程架构
    # ============================================================
    def run_loop(self):
        """主循环 - 启动三线程架构"""
        logger.info("=" * 60)
        logger.info("【机器人执行器 - 实时动作序列版】")
        logger.info(f"推理服务器: {self.inference_host}:{self.inference_port}")
        logger.info(f"本地IP: {self.local_ip}")
        logger.info(f"右臂: {self.arm_right_ip}")
        logger.info(f"观测发送频率: {self.send_obs_hz}Hz")
        logger.info(f"动作执行频率: 100Hz (每10ms一步)")
        logger.info(f"动作序列长度: 50步")
        logger.info(f"单步限幅: ±{self.max_step_deg}°")
        logger.info("=" * 60)

        self._running = True

        # 启动三线程（观测发送/动作接收/执行）
        threads_info = [
            (self._obs_sender_thread_worker, "OBS_SENDER"),
            (self._recv_thread_worker, "ACTION_RECV"),
            (self._exec_thread_worker, "EXEC"),
        ]

        for worker, name in threads_info:
            t = threading.Thread(target=worker, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            logger.info(f"[START] {name} 线程已启动")

        # 无限期等待
        logger.info("[MAIN] 所有线程已启动，等待运行...")
        try:
            for t in self._threads:
                t.join()
        except KeyboardInterrupt:
            self._running = False
        finally:
            self._running = False
            self._cleanup()
            logger.info(f"[MAIN] 执行器已停止: 观测发送={self._obs_send_count}, "
                        f"动作序列接收={self._recv_count}, 执行={self._exec_count}")

# ============================================================
# 主入口
# ============================================================
def main():
    print("=" * 60)
    print(f"机器人执行器 - 实时动作序列版")
    print(f"日志文件: {LOG_FILE}")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="机器人执行器 (实时动作序列版)")
    parser.add_argument("--inference-host", default="192.168.1.101", help="推理服务器IP")
    parser.add_argument("--inference-port", type=int, default=9000, help="推理服务器端口")
    parser.add_argument("--local-ip", default="192.168.1.100", help="本地IP")
    parser.add_argument("--arm-right-ip", default="192.168.1.19", help="右臂IP")
    parser.add_argument("--arm-tcp-port", type=int, default=8080, help="机械臂TCP端口")
    parser.add_argument("--arm-right-udp-port", type=int, default=18089, help="右臂UDP端口")
    parser.add_argument("--max-step", type=float, default=1.0, help="单步限幅(度)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG质量")
    parser.add_argument("--connect-delay", type=int, default=60, help="连接推理服务器超时(秒)")
    parser.add_argument("--send-obs-hz", type=float, default=60.0, help="观测发送频率(Hz)")
    args = parser.parse_args()

    executor = RobotExecutor(
        inference_host=args.inference_host,
        inference_port=args.inference_port,
        local_ip=args.local_ip,
        arm_right_ip=args.arm_right_ip,
        arm_tcp_port=args.arm_tcp_port,
        arm_right_udp_port=args.arm_right_udp_port,
        max_step_deg=args.max_step,
        jpeg_quality=args.jpeg_quality,
        connect_delay=args.connect_delay,
        send_obs_hz=args.send_obs_hz,
    )

    try:
        executor.connect_hardware()
        executor.run_loop()
    finally:
        executor.stop()

if __name__ == "__main__":
    main()