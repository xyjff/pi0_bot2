"""将智平方双臂机器人 HDF5 数据转换为 LeRobot (OpenPI) 数据集格式的一体化脚本。

1、将HDF5格式的文件，转为lerobot的2.1版本的格式。
2、涉及到机器人的4路摄像头，分别是头部，胸部，左手，右手。
双臂14自由度。
灵巧手2自由度，完全闭合1000，完全打开0。


特点：
1. 内置且修正了“先右臂后左臂”的 16维状态拼接（8+8），完美匹配 ALOHA/OpenPI 控制器逻辑。
2. 内部直接融合了 VLA 模型所需的多语言指令（Multi-Language Instructions）映射。
3. 一步生成涵盖 Parquet, MP4, stats.json, info.json 特征的全套数据集，无需前后的辅助补丁脚本。
"""

import h5py
import numpy as np
import cv2
import os
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import glob
import time

# =============================================================================
# [核心配置区] 请根据您的任务场景随意修改以下参数   3步
# =============================================================================
# 1、进入环境：       source /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/activate
# 2、路径修改
INPUT_DIR = r'/share/0xyj/datasate_baocun/task_657_14双臂+4路摄像头+2灵巧手4.22新采集数据/ds_635_31_657_14双臂+4路摄像头+2灵巧手4.22新采集数据'          # 输入 HDF5 文件所在目录 (请确保此目录下全是您要转换的 .h5 文件)
OUTPUT_DIR = r'/share/0xyj/model3_openpi0.5/hdf5_to_lerobot_data/lerobot_dataset'      # 输出 LeRobot 数据集的根目录 (转换完成后会在此目录下生成 data/, meta/, videos/ 等子目录)
FPS = 30
CHUNK_SIZE = 1000

# 3、【重中之重】：您的 VLA (Pi0.5、pi0) 语言指令集合，这里存放了您此次数据对应的不同自然语言表述 (增加模型泛化能力)
# 目前的任务是：使用右臂的灵巧手去夹起箱子然后放下
TASK_INSTRUCTIONS = [
    "use the right dexterous hand to pick up the box and place it down",
    "grasp the crate with your right hand and put it on the table",
    "use the right gripper to pick up the box, then release it",
    "right arm dexterous hand grasp the box and set it down",
    "pick up the box using the right hand and carefully place it down",
    "use right dexterous hand to grip the box and release it",
    "grab the box with your right arm's hand, then ungrasp it",
    "right hand grasp the crate and lower it down gently",
    "use the right arm hand to clamp the box and release it",
    "pick the box up with the right dexterous hand, then set it down"
]

# 4、一键启动转换脚本       python /share/0xyj/model3_openpi0.5/hdf5_to_lerobot_data/convert_zhipingfang_to_lerobot.py




# 在这个单任务脚本中，所有数据的任务 ID 都默认归一为 0
DEFAULT_TASK_INDEX = 0

# =============================================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def convert_deg_to_rad(deg):
    return deg * np.pi / 180.0

def format_seconds(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def process_dataset():
    total_start_time = time.time()
    ensure_dir(OUTPUT_DIR)

    # 1. 创建 LeRobot V2 标准目录结构
    data_dir = Path(OUTPUT_DIR) / 'data/chunk-000'
    meta_dir = Path(OUTPUT_DIR) / 'meta'
    episodes_dir = meta_dir / 'episodes/chunk-000'
    videos_dir = Path(OUTPUT_DIR) / 'videos'

    ensure_dir(data_dir)
    ensure_dir(meta_dir)
    ensure_dir(episodes_dir)

    # 智平方 3 路相机的映射（去掉胸部相机）
    cam_keys = {
        'head': 'observation.images.cam_high',
        'left': 'observation.images.cam_left_wrist',
        'right': 'observation.images.cam_right_wrist'
    }

    for key, name in cam_keys.items():
        ensure_dir(videos_dir / name / 'chunk-000')

    # 2. 扫描输入 HDF5 文件
    files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.h5')))
    if not files:
        print(f"❌ 错误: 在 {INPUT_DIR} 中未找到任何 HDF5 文件")
        return
    total_episodes = len(files)
    print(f"✅ 找到 {total_episodes} 个hdf5文件待处理: {[os.path.basename(f) for f in files]}")
    print("⏱️ 已开启耗时统计：将显示每集耗时、总耗时与预计剩余时间 ETA")

    # 3. 初始化全局统计累加器
    stats_accum = {}
    def init_stats(key, shape):
        stats_accum[key] = {
            'sum': np.zeros(shape, dtype=np.float64),
            'sq_sum': np.zeros(shape, dtype=np.float64),
            'min': np.full(shape, np.inf, dtype=np.float64),
            'max': np.full(shape, -np.inf, dtype=np.float64),
            'count': 0
        }

    episodes_metadata = []
    global_frame_counter = 0
    total_frames = 0

    # 4. 遍历处理每个剧集
    for episode_idx, file_path in enumerate(files):
        ep_start_time = time.time()
        print(f"\n🚀 正在处理第 {episode_idx} 个剧集: {file_path}")

        with h5py.File(file_path, 'r') as f:
            # -------------------------------------------------------------
            # A. 读取本体传感器数据
            # -------------------------------------------------------------
            left_joints_deg = f['observations/arm/left/joints'][:]
            right_joints_deg = f['observations/arm/right/joints'][:]
            # 弧度转换
            left_joints_rad = convert_deg_to_rad(left_joints_deg)
            right_joints_rad = convert_deg_to_rad(right_joints_deg)

            left_hand = f['observations/effector/left/position'][:].astype(np.float32)
            right_hand = f['observations/effector/right/position'][:].astype(np.float32)

            # 灵巧手语义：完全打开=0，完全闭合=1000，归一化到 [0, 1]
            left_hand_norm = np.clip(left_hand / 1000.0, 0.0, 1.0)
            right_hand_norm = np.clip(right_hand / 1000.0, 0.0, 1.0)

            # 严格遵循“先右臂后左臂”拼接顺序。
            # [右7关节, 右1灵巧手, 左7关节, 左1灵巧手] -> 总共 8 + 8 = 16维
            state = np.concatenate([
                right_joints_rad, right_hand_norm,
                left_joints_rad, left_hand_norm
            ], axis=1).astype(np.float32)

            num_frames = state.shape[0]

            # 生成 action 与 effort
            action = np.zeros_like(state)
            action[:-1] = state[1:]  # 用 state 移位充当 action
            action[-1] = state[-1]
            effort = np.zeros_like(state)
            timestamps = (np.arange(num_frames) / FPS).astype(np.float32)

            # -------------------------------------------------------------
            # B. 记录统计数据
            # -------------------------------------------------------------
            def update_stats(key, data):
                if key not in stats_accum:
                    init_stats(key, data.shape[1:] if data.ndim > 1 else ())
                s = stats_accum[key]
                s['sum'] += np.sum(data, axis=0)
                s['sq_sum'] += np.sum(data ** 2, axis=0)
                s['min'] = np.minimum(s['min'], np.min(data, axis=0))
                s['max'] = np.maximum(s['max'], np.max(data, axis=0))
                s['count'] += data.shape[0]

            update_stats('observation.state', state)
            update_stats('observation.effort', effort)
            update_stats('action', action)
            update_stats('timestamp', timestamps)

            # -------------------------------------------------------------
            # C. 处理相机与视频并收集图像统计
            # -------------------------------------------------------------
            selected_codec = 'mp4v'
            for key, name in cam_keys.items():
                img_ds = f[f'observations/camera/rgb/{key}/images']
                video_path = videos_dir / name / 'chunk-000' / f'file-{episode_idx:03d}.mp4'

                first_img = cv2.imdecode(np.frombuffer(img_ds[0], np.uint8), cv2.IMREAD_COLOR)
                height, width = first_img.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*selected_codec)
                out = cv2.VideoWriter(str(video_path), fourcc, FPS, (width, height))

                if name not in stats_accum:
                    init_stats(name, (3,))

                pixel_sum, pixel_sq_sum = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
                batch_min, batch_max = np.full(3, 255.0, dtype=np.float64), np.full(3, 0.0, dtype=np.float64)

                for i in range(num_frames):
                    img = cv2.imdecode(np.frombuffer(img_ds[i], np.uint8), cv2.IMREAD_COLOR)
                    out.write(img)

                    if i % 10 == 0:
                        img_norm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        pixel_sum += img_norm.sum(axis=(0, 1))
                        pixel_sq_sum += (img_norm ** 2).sum(axis=(0, 1))
                        batch_min = np.minimum(batch_min, img_norm.min(axis=(0, 1)))
                        batch_max = np.maximum(batch_max, img_norm.max(axis=(0, 1)))

                out.release()

                s = stats_accum[name]
                s['sum'] += pixel_sum
                s['sq_sum'] += pixel_sq_sum
                s['min'] = np.minimum(s['min'], batch_min)
                s['max'] = np.maximum(s['max'], batch_max)
                s['count'] += ((num_frames + 9) // 10) * height * width

            # -------------------------------------------------------------
            # D. 保存具体每一帧的 Parquet 数据 (包含文字表述 ID)
            # -------------------------------------------------------------
            data_dict = {
                'observation.state': list(state),
                'observation.effort': list(effort),
                'action': list(action),
                'episode_index': np.full(num_frames, episode_idx, dtype=np.int64),
                'frame_index': np.arange(num_frames, dtype=np.int64),
                'timestamp': timestamps,
                'next.done': np.full(num_frames, False, dtype=bool),
                'index': np.arange(global_frame_counter, global_frame_counter + num_frames, dtype=np.int64),
                
                # 🎉【关键修正点 2】：分配唯一的 task_index 关联到所有 language instruction
                'task_index': np.full(num_frames, DEFAULT_TASK_INDEX, dtype=np.int64)
            }
            data_dict['next.done'][-1] = True

            update_stats('episode_index', data_dict['episode_index'])
            update_stats('frame_index', data_dict['frame_index'])
            update_stats('index', data_dict['index'])
            update_stats('task_index', data_dict['task_index'])
            update_stats('next.done', data_dict['next.done'].astype(float))

            fields = [
                pa.field('observation.state', pa.list_(pa.float32())),
                pa.field('observation.effort', pa.list_(pa.float32())),
                pa.field('action', pa.list_(pa.float32())),
                pa.field('episode_index', pa.int64()),
                pa.field('frame_index', pa.int64()),
                pa.field('timestamp', pa.float32()),
                pa.field('next.done', pa.bool_()),
                pa.field('index', pa.int64()),
                pa.field('task_index', pa.int64())
            ]
            pq.write_table(pa.Table.from_pydict(data_dict, schema=pa.schema(fields)), str(data_dir / f'file-{episode_idx:03d}.parquet'))

            # -------------------------------------------------------------
            # E. 记录此集片段的大纲统计
            # -------------------------------------------------------------
            episode_dict = {
                'episode_index': episode_idx,
                # 🎉【关键修正点 3】：直接在此处注入该剧集的全部多种语言指令！
                # 免去了用 add_language_instructions 后处理修改 episodes.parquet 的烦恼。
                'tasks': TASK_INSTRUCTIONS,  
                'length': num_frames,
                'dataset_from_index': global_frame_counter,
                'dataset_to_index': global_frame_counter + num_frames - 1,
                'data/chunk_index': 0,
                'data/file_index': episode_idx,
                'meta/episodes/chunk_index': 0,
                'meta/episodes/file_index': 0,
            }

            for key, name in cam_keys.items():
                episode_dict[f'videos/{name}/chunk_index'] = 0
                episode_dict[f'videos/{name}/file_index'] = episode_idx
                episode_dict[f'videos/{name}/from_timestamp'] = float(timestamps[0])
                episode_dict[f'videos/{name}/to_timestamp'] = float(timestamps[-1])

            for col in ['observation.state', 'observation.effort', 'action']:
                data = np.stack(data_dict[col])
                episode_dict[f'stats/{col}/min'] = data.min(axis=0).tolist()
                episode_dict[f'stats/{col}/max'] = data.max(axis=0).tolist()
                episode_dict[f'stats/{col}/mean'] = data.mean(axis=0).tolist()
                episode_dict[f'stats/{col}/std'] = data.std(axis=0).tolist()
                episode_dict[f'stats/{col}/count'] = [num_frames]

            episode_dict[f'stats/timestamp/min'] = [float(timestamps.min())]
            episode_dict[f'stats/timestamp/max'] = [float(timestamps.max())]
            episode_dict[f'stats/timestamp/mean'] = [float(timestamps.mean())]
            episode_dict[f'stats/timestamp/std'] = [float(timestamps.std())]
            episode_dict[f'stats/timestamp/count'] = [num_frames]

            episodes_metadata.append(episode_dict)
            global_frame_counter += num_frames
            total_frames += num_frames

        # -------------------------------------------------------------
        # F. 打印进度与剩余时间估算
        # -------------------------------------------------------------
        done_episodes = episode_idx + 1
        elapsed_total = time.time() - total_start_time
        elapsed_ep = time.time() - ep_start_time
        avg_ep = elapsed_total / done_episodes
        remain_episodes = len(files) - done_episodes
        eta_seconds = avg_ep * remain_episodes
        expected_finish_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + eta_seconds))

        print(
            f"✅ 第 {episode_idx} 集完成 | 本集耗时: {format_seconds(elapsed_ep)} | "
            f"累计耗时: {format_seconds(elapsed_total)} | "
            f"平均每集: {format_seconds(avg_ep)} | "
            f"预计剩余: {format_seconds(eta_seconds)} | "
            f"预计完成: {expected_finish_str}"
        )

    # =========================================================================
    # 5. 写入各个核心元数据文件
    # =========================================================================
    print("\n⏳ 正在计算并生成最终数据集元数据...")
    
    # 5.1 生成 Stats statistics (stats.json)
    final_stats = {}
    for key, s in stats_accum.items():
        if s['count'] > 0:
            mean = s['sum'] / s['count']
            std = np.sqrt(np.maximum((s['sq_sum'] / s['count']) - (mean ** 2), 0))
            if key in cam_keys.values():
                final_stats[key] = {
                    'min': s['min'].reshape(3, 1, 1).tolist(), 'max': s['max'].reshape(3, 1, 1).tolist(),
                    'mean': mean.reshape(3, 1, 1).tolist(), 'std': std.reshape(3, 1, 1).tolist(),
                    'count': [int(s['count'])]
                }
            else:
                if np.isscalar(mean) or mean.ndim == 0:
                    final_stats[key] = {
                        'min': [s['min'].item() if hasattr(s['min'], 'item') else s['min']],
                        'max': [s['max'].item() if hasattr(s['max'], 'item') else s['max']],
                        'mean': [mean.item() if hasattr(mean, 'item') else mean],
                        'std': [std.item() if hasattr(std, 'item') else std],
                        'count': [int(total_frames)]
                    }
                else:
                    final_stats[key] = {'min': s['min'].tolist(), 'max': s['max'].tolist(), 'mean': mean.tolist(), 'std': std.tolist(), 'count': [int(total_frames)]}
    with open(meta_dir / 'stats.json', 'w') as f:
        json.dump(final_stats, f, indent=4)

    # 5.2 生成 episodes.parquet
    pd.DataFrame(episodes_metadata).to_parquet(episodes_dir / 'file-000.parquet', engine='pyarrow')

    # 5.3 🌟 生成跨任务语言映射文件 (tasks.parquet) 🌟
    # 根据 LeRobot 规范，每条 instruction 将作为 Index，而 task_index 在此列
    tasks_records = [{'instruction': inst, 'task_index': DEFAULT_TASK_INDEX} for inst in TASK_INSTRUCTIONS]
    df_new_tasks = pd.DataFrame(tasks_records)
    df_new_tasks.set_index('instruction', inplace=True)
    df_new_tasks.to_parquet(meta_dir / 'tasks.parquet', engine='pyarrow')

    # 5.4 生成 info.json
    motor_names = [
        "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3",
        "right_joint_4", "right_joint_5", "right_joint_6", "right_dexterous_hand",
        "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3",
        "left_joint_4", "left_joint_5", "left_joint_6", "left_dexterous_hand"
    ]
    info = {
        "codebase_version": "v3.0",
        "robot_type": "aloha",
        "total_episodes": len(files),
        "total_frames": total_frames,
        "total_tasks": 1, 
        "fps": FPS,
        "chunks_size": CHUNK_SIZE,
        "splits": {"train": f"0:{len(files)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [16], "names": {"motors": motor_names}, "fps": FPS},
            "observation.effort": {"dtype": "float32", "shape": [16], "names": {"motors": motor_names}, "fps": FPS},
            "action": {"dtype": "float32", "shape": [16], "names": {"motors": motor_names}, "fps": FPS},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": FPS},
            "next.done": {"dtype": "bool", "shape": [1], "names": None, "fps": FPS},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
            "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS}
        }
    }
    for key, name in cam_keys.items():
        info['features'][name] = {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"],
            "video_info": {"video.fps": float(FPS), "video.codec": "mp4v", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
    with open(meta_dir / 'info.json', 'w') as f:
        json.dump(info, f, indent=4)

    total_elapsed = time.time() - total_start_time
    print(f"\n✅ 恭喜！智平方 {total_episodes} 组剧集转换全流程完成！")
    print(
        f"📊 总耗时: {format_seconds(total_elapsed)} | "
        f"平均每组: {format_seconds(total_elapsed / max(total_episodes, 1))}"
    )

if __name__ == '__main__':
    try:
        process_dataset()
    except Exception as e:
        import traceback
        traceback.print_exc()
