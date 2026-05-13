import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

# XLA 및 MuJoCo 관련 환경 변수 설정
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import logging as python_logging
LOGGER = python_logging.getLogger()
LOGGER.setLevel(python_logging.INFO)

from absl import logging
logging.set_verbosity(logging.INFO)

from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import tqdm
import torch

# 사용자 환경에 맞게 임포트 경로 유지
from src.envs.g1.g1_tracking_env import default_config
from src.envs.g1.play_g1_tracking_env import PlayG1TrackingEnv


@dataclass
class Args:
    exp_name: str
    play_ref_motion: bool = False
    use_viewer: bool = False  # passive viewer (with display)
    use_renderer: bool = False  # renderer with video (headless mode)


@dataclass
class State:
    info: dict
    obs: dict


def get_latest_ckpt(path: Path) -> Path | None:
    ckpts = [ckpt for ckpt in path.glob("*") if not ckpt.name.endswith(".json")]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None


# =============================================================================
# [서브 프로세스] 4개의 독립된 창, 각각 3x4 배열의 12개 서브플롯을 그리는 함수
# =============================================================================
def plot_process(times_list, angles_list, velocities_list, torques_list, powers_list, joint_names):
    times = np.array(times_list)
    angles = np.array(angles_list)
    velocities = np.array(velocities_list)
    torques = np.array(torques_list)
    powers = np.array(powers_list)
    
    num_leg_joints = 12 
    
    # --- 1. 각도(Angle) 창 ---
    fig_angle, axs_angle = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    fig_angle.canvas.manager.set_window_title('Joint Angles')
    fig_angle.suptitle('Leg Joints: Angle (rad)', fontsize=18, fontweight='bold')
    axs_angle = axs_angle.flatten()

    # --- 2. 각속도(Velocity) 창 ---
    fig_vel, axs_vel = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    fig_vel.canvas.manager.set_window_title('Joint Angular Velocities')
    fig_vel.suptitle('Leg Joints: Angular Velocity (rad/s)', fontsize=18, fontweight='bold')
    axs_vel = axs_vel.flatten()

    # --- 3. 토크(Torque) 창 ---
    fig_torque, axs_torque = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    fig_torque.canvas.manager.set_window_title('Joint Torques')
    fig_torque.suptitle('Leg Joints: Torque (Nm)', fontsize=18, fontweight='bold')
    axs_torque = axs_torque.flatten()

    # --- 4. 파워(Power) 창 ---
    fig_power, axs_power = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    fig_power.canvas.manager.set_window_title('Joint Powers')
    fig_power.suptitle('Leg Joints: Power (W)', fontsize=18, fontweight='bold')
    axs_power = axs_power.flatten()

    # 12개의 다리 관절에 대해 데이터 그리기
    for idx in range(num_leg_joints):
        name = joint_names[idx] if idx < len(joint_names) else f"Joint {idx}"
        
        # 자유도가 추가된 플로팅 베이스(qpos/qvel 길이 > ctrl 길이) 고려한 인덱싱
        angle_idx = -len(torques[0]) + idx
        
        # 각도 그래프
        axs_angle[idx].plot(times, angles[:, angle_idx], color='tab:blue', linewidth=1.5)
        axs_angle[idx].set_title(name, fontsize=11)
        axs_angle[idx].grid(True, linestyle='--', alpha=0.6)
        
        # 각속도 그래프
        axs_vel[idx].plot(times, velocities[:, angle_idx], color='tab:orange', linewidth=1.5)
        axs_vel[idx].set_title(name, fontsize=11)
        axs_vel[idx].grid(True, linestyle='--', alpha=0.6)
        
        # 토크 그래프
        axs_torque[idx].plot(times, torques[:, idx], color='tab:red', linewidth=1.5)
        axs_torque[idx].set_title(name, fontsize=11)
        axs_torque[idx].grid(True, linestyle='--', alpha=0.6)
        
        # 파워 그래프
        axs_power[idx].plot(times, powers[:, idx], color='tab:green', linewidth=1.5)
        axs_power[idx].set_title(name, fontsize=11)
        axs_power[idx].grid(True, linestyle='--', alpha=0.6)

    # 4개의 창 모두 제일 아래쪽 열(8, 9, 10, 11번 플롯)에만 X축(시간) 라벨 표시
    for i in range(8, 12):
        axs_angle[i].set_xlabel('Time (s)', fontsize=10)
        axs_vel[i].set_xlabel('Time (s)', fontsize=10)
        axs_torque[i].set_xlabel('Time (s)', fontsize=10)
        axs_power[i].set_xlabel('Time (s)', fontsize=10)

    # 전체 레이아웃 정렬 (메인 타이틀과 겹치지 않게 여백 조정)
    fig_angle.tight_layout(rect=[0, 0, 1, 0.95])
    fig_vel.tight_layout(rect=[0, 0, 1, 0.95])
    fig_torque.tight_layout(rect=[0, 0, 1, 0.95])
    fig_power.tight_layout(rect=[0, 0, 1, 0.95])

    # 4개의 창을 한 번에 모두 띄우기 (메인 시뮬레이션과 별개 스레드이므로 block=True)
    plt.show(block=True)


# =============================================================================
# 메인 플레이 함수
# =============================================================================
def play(args: Args):
    task_cfg = default_config()
    env_cfg = task_cfg.env_config

    config_path = Path(__file__).parent / "experiments" / args.exp_name / "checkpoints" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    del config["env_config"]["reference_traj_config"]
    env_cfg.update(config["env_config"])

    # env_cfg.reference_traj_config.name = {"lafan1": ["sprint1_subject2"]}
    # env_cfg.reference_traj_config.name = {"lafan1": ["dance1_subject1"]}
    env_cfg.reference_traj_config.name = {"lafan1": ["walk2_subject1"]}
    
    # [추가] 무작위 시작을 끄고, 시작 프레임을 고정합니다.
    env_cfg.reference_traj_config.random_start = False
    
    # env_cfg.reference_traj_config.fixed_start_frame = 800 # sprint1_subject2
    # env_cfg.reference_traj_config.fixed_start_frame = 5000 # dance1_subject1 side flip
    env_cfg.reference_traj_config.fixed_start_frame = 0 # dance1_subject1
    
    
    env = PlayG1TrackingEnv(
        terrain_type=env_cfg.terrain_type,
        config=env_cfg,
        play_ref_motion=args.play_ref_motion,
        use_viewer=args.use_viewer,
        use_renderer=args.use_renderer,
        exp_name=args.exp_name,
    )
    
    ckpt_path = Path(__file__).parent / "experiments" / args.exp_name / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_path)
    if latest_ckpt is None:
        raise FileNotFoundError("No checkpoint found.")

    policy_path = latest_ckpt / "policy.pt"
    policy_jit = torch.jit.load(policy_path, map_location="cpu")
    state = env.reset()

    # ---------------------------------------------------------
    # [터미널 출력] MuJoCo 모델에서 구동 관절(Actuator) 이름 추출
    # ---------------------------------------------------------
    try:
        # 환경 구조에 따라 다를 수 있는 속성 이름들을 모두 시도합니다.
        if hasattr(env, 'mj_model'):
            mj_model = env.mj_model
        elif hasattr(env, 'model'):
            mj_model = env.model
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'model'):
            mj_model = env.unwrapped.model
        else:
            mj_model = None

        if mj_model is not None:
            # MuJoCo 바인딩 문법을 사용하여 액추에이터 이름 가져오기
            joint_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
        else:
            # 모델을 찾지 못한 경우 기본 이름 생성
            joint_names = [f"Actuator_{i}" for i in range(12)]
            
    except Exception as e:
        print(f"\n⚠️ 관절 이름을 불러오는 데 실패했습니다: {e}")
        joint_names = [f"Actuator_{i}" for i in range(12)]

    print("\n" + "="*50)
    print("🤖 G1 로봇 다리 관절(12개) 순서 확인:")
    for i, name in enumerate(joint_names[:12]):
        print(f"  [{i:02d}] {name}")
    print("="*50 + "\n")
    
    # 앞의 12개를 다리 관절(Legs) 이름으로 저장
    leg_joint_names = joint_names[:12] 
    # ---------------------------------------------------------

    # 데이터 버퍼 설정
    record_start = 0.0
    record_end = 8.0
    
    log_times = []
    log_angles = []
    log_velocities = []
    log_torques = []
    log_powers = []
    
    plot_launched = False

    len_traj = env.th.traj.data.qpos.shape[0] - len(env_cfg.reference_traj_config.name) - 1
    
    # 메인 시뮬레이션 루프
    for i in tqdm(range(len_traj)):
        with torch.no_grad():
            action = policy_jit(torch.from_numpy(state.obs["state"].reshape(1, -1).astype(np.float32))).cpu().numpy()
        state = env.step(state, action)

        # 환경 래퍼에 맞게 MuJoCo 데이터 객체 탐색
        if hasattr(env, 'data'):
            mj_data = env.data
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'data'):
            mj_data = env.unwrapped.data
        elif hasattr(env, 'mj_data'):
            mj_data = env.mj_data
        else:
            continue 

        # time.time() 대신 MuJoCo의 내부 물리 시간을 직접 가져옵니다.
        current_time = mj_data.time
        
        # 지정된 시간 구간에서만 데이터 버퍼에 저장
        if record_start <= current_time <= record_end:
            # 프로세스 간 전송을 위해 tolist()로 변환
            angles = np.copy(mj_data.qpos).tolist()
            velocities = np.copy(mj_data.qvel).tolist()
            torques = np.copy(mj_data.actuator_force).tolist()
            
            # 파워(W) 계산 (Torque * Angular Velocity)
            t_np = np.array(torques)
            v_np = np.array(velocities)
            powers = (t_np * v_np[-len(t_np):]).tolist()
            
            log_times.append(current_time)
            log_angles.append(angles)
            log_velocities.append(velocities)
            log_torques.append(torques)
            log_powers.append(powers)
            
        # 기록 종료 시간이 지나고 아직 창을 안 띄웠을 때
        elif current_time > record_end and not plot_launched:
            # =========================================================
            # [추가] 오른쪽 다리 최대 각속도/토크 추출 및 터미널 출력 로직
            # =========================================================
            t_arr = np.array(log_torques)
            v_arr = np.array(log_velocities)
            
            right_leg_stats = []
            
            for idx in range(12):
                name = leg_joint_names[idx]
                if "right" in name.lower(): # 오른쪽 다리 관절만 필터링
                    # 각속도 인덱스는 플로팅 베이스(root)를 제외한 오프셋 적용
                    vel_idx = -len(t_arr[0]) + idx
                    
                    # 0~8초 구간 내에서 절댓값 기준 최대치 찾기
                    max_v = np.max(np.abs(v_arr[:, vel_idx]))
                    max_t = np.max(np.abs(t_arr[:, idx]))
                    
                    right_leg_stats.append((name, max_v, max_t))
            
            # 가장 값이 큰 관절 찾기
            max_v_joint = max(right_leg_stats, key=lambda x: x[1])
            max_t_joint = max(right_leg_stats, key=lambda x: x[2])
            
            print("\n" + "="*60)
            print(f"⏱️  {record_start}초 ~ {record_end}초 오른쪽 다리(Right Leg) 분석 결과")
            print("="*60)
            print(f"🔥 최대 각속도 낸 관절 : {max_v_joint[0]} ({max_v_joint[1]:.2f} rad/s)")
            print(f"💪 최대 토크 낸 관절   : {max_t_joint[0]} ({max_t_joint[2]:.2f} Nm)")
            print("-" * 60)
            print("📊 세부 데이터 (오른쪽 다리 전체):")
            for stat in right_leg_stats:
                print(f"  - {stat[0]:<25} | Max Vel: {stat[1]:>6.2f} rad/s | Max Torque: {stat[2]:>6.2f} Nm")
            print("="*60 + "\n")
            # =========================================================

            print(f"✅ 데이터 수집 완료. 그래프 창(4개)을 엽니다.")
            # 멀티프로세싱으로 그래프 도시 (메인 시뮬레이션 무중단)
            p = mp.Process(target=plot_process, args=(
                log_times, log_angles, log_velocities, log_torques, log_powers, leg_joint_names
            ))
            p.start()
            plot_launched = True 

    env.close()


if __name__ == "__main__":
    # Windows, macOS 등 운영체제 환경에 따라 멀티프로세싱 시작 방식이 다를 경우 주석 해제하여 사용
    # mp.set_start_method('spawn', force=True)
    args = tyro.cli(Args)
    play(args)