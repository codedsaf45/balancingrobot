"""
Ascento Biped - LQR 밸런싱 + 점프
===================================

뷰어 창에서:
  I/K : 전진/후진
  J/L : 좌/우 회전
  Space : 정지
  P : 점프!

실행:
    python lqr_jump.py
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy import linalg
import time

# ============================================================
# 1. 상태 변수
# ============================================================
target_vel = 0.0
target_yaw_rate = 0.0
jump_trigger = False

# ============================================================
# 2. LQR 설정
# ============================================================
M_body = 2.0 + 0.2*2 + 0.15*2
M_wheel = 0.25 * 2
M_total = M_body + M_wheel
g = 9.81
L = 0.20
R = 0.055
I_body = M_body * L**2
denom = I_body + M_total * L**2

A = np.array([
    [0, 1, 0, 0],
    [M_total * g * L / denom, 0, 0, 0],
    [0, 0, 0, 1],
    [-M_body * g * L / (M_total * R), 0, 0, 0]
])
B = np.array([
    [0],
    [-1.0 / denom],
    [0],
    [1.0 / (M_total * R)]
])

Q = np.diag([200.0, 20.0, 50.0, 20.0])
R_cost = np.array([[0.5]])
P = linalg.solve_continuous_are(A, B, Q, R_cost)
K = np.linalg.inv(R_cost) @ B.T @ P
print(f"LQR Gain K: [{K[0,0]:.1f}, {K[0,1]:.1f}, {K[0,2]:.1f}, {K[0,3]:.1f}]")

# ============================================================
# 3. 유틸리티
# ============================================================
def get_pitch(data):
    w, x, y, z = data.qpos[3:7]
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    return np.arcsin(sinp)

def check_wheel_contact(model, data):
    """바퀴가 바닥에 닿아있는지 체크"""
    for i in range(data.ncon):
        con = data.contact[i]
        geom1 = model.geom(con.geom1).name
        geom2 = model.geom(con.geom2).name
        if 'lw' in (geom1, geom2) or 'rw' in (geom1, geom2):
            return True
    return False

# ============================================================
# 4. 키 콜백
# ============================================================
def key_callback(keycode):
    global target_vel, target_yaw_rate, jump_trigger
    
    if keycode == ord('i') or keycode == 265:     # 전진
        target_vel = min(target_vel + 0.1, 1.0)
    # elif keycode == ord('k') or keycode == 264:   # 후진
    #     target_vel = max(target_vel - 0.1, -1.0)
    elif keycode == ord('j') or keycode == 263:   # 좌회전
        target_yaw_rate = min(target_yaw_rate + 0.5, 3.0)
    elif keycode == ord('l') or keycode == 262:   # 우회전
        target_yaw_rate = max(target_yaw_rate - 0.5, -3.0)
    elif keycode == ord(' '):                      # 정지
        target_vel = 0.0
        target_yaw_rate = 0.0
    elif keycode == ord('p') or keycode ==264:                      # 점프!
        jump_trigger = True

# ============================================================
# 5. 점프 상태 머신
# ============================================================
class JumpController:
    """
    점프 시퀀스:
      IDLE → CROUCH → LAUNCH → FLIGHT → LAND → RECOVER → IDLE
    """
    def __init__(self):
        self.phase = "IDLE"
        self.phase_start = 0.0
        
        # 타이밍 (초)
        self.CROUCH_TIME = 0.3    # 앉는 시간
        self.LAUNCH_TIME = 0.08   # 도약 시간 (짧고 강하게)
        self.LAND_TIME = 0.5      # 착지 안정화 시간
        
        # 자세 각도 (도)
        self.CROUCH_HIP = 60      # 깊이 앉기
        self.LAUNCH_HIP = -5      # 빠르게 펴기
        self.FLIGHT_HIP = 35      # 공중 착지 준비
        self.LAND_HIP = 50        # 착지 충격 흡수
        self.NORMAL_HIP = 20      # 일반 자세
    
    def elapsed(self, t):
        return t - self.phase_start
    
    def transition(self, new_phase, t):
        print(f"  🦿 {self.phase} → {new_phase} (t={t:.2f}s)")
        self.phase = new_phase
        self.phase_start = t
    
    def update(self, model, data, t, on_ground):
        """현재 phase에 따라 hip, wheel 제어값 반환"""
        global jump_trigger
        
        dt = self.elapsed(t)
        hip_cmd = self.NORMAL_HIP
        wheel_torque = None  # None이면 LQR 사용
        
        # --- IDLE: 일반 밸런싱 ---
        if self.phase == "IDLE":
            hip_cmd = self.NORMAL_HIP
            if jump_trigger:
                jump_trigger = False
                self.transition("CROUCH", t)
        
        # --- CROUCH: 깊이 앉기 ---
        elif self.phase == "CROUCH":
            # 점진적으로 앉기
            ratio = min(dt / self.CROUCH_TIME, 1.0)
            hip_cmd = self.NORMAL_HIP + (self.CROUCH_HIP - self.NORMAL_HIP) * ratio
            wheel_torque = 0.0  # 바퀴 고정
            
            if dt >= self.CROUCH_TIME:
                self.transition("LAUNCH", t)
        
        # --- LAUNCH: 도약! ---
        elif self.phase == "LAUNCH":
            hip_cmd = self.LAUNCH_HIP  # 다리 빠르게 펴기
            wheel_torque = 0.0
            
            if dt >= self.LAUNCH_TIME:
                self.transition("FLIGHT", t)
        
        # --- FLIGHT: 공중 ---
        elif self.phase == "FLIGHT":
            hip_cmd = self.FLIGHT_HIP  # 착지 준비 자세
            wheel_torque = 0.0
            
            # 바퀴가 땅에 닿으면 착지
            if dt > 0.05 and on_ground:
                self.transition("LAND", t)
            # 너무 오래 공중이면 강제 착지
            elif dt > 1.0:
                self.transition("LAND", t)
        
        # --- LAND: 착지 + 충격 흡수 ---
        elif self.phase == "LAND":
            # 점진적으로 다리 굽히며 충격 흡수
            ratio = min(dt / 0.15, 1.0)
            hip_cmd = self.FLIGHT_HIP + (self.LAND_HIP - self.FLIGHT_HIP) * ratio
            
            # 착지 초반: 바퀴 프리, 후반: LQR 복귀
            if dt < 0.1:
                wheel_torque = 0.0
            else:
                wheel_torque = None  # LQR 복귀
            
            if dt >= self.LAND_TIME:
                self.transition("RECOVER", t)
        
        # --- RECOVER: 일반 자세 복귀 ---
        elif self.phase == "RECOVER":
            ratio = min(dt / 0.3, 1.0)
            hip_cmd = self.LAND_HIP + (self.NORMAL_HIP - self.LAND_HIP) * ratio
            
            if dt >= 0.3:
                self.transition("IDLE", t)
        
        return hip_cmd, wheel_torque


# ============================================================
# 6. 메인 루프
# ============================================================
def main():
    model = mujoco.MjModel.from_xml_path("bipedalrobot.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[4] = 0.01
    mujoco.mj_forward(model, data)
    
    jump_ctrl = JumpController()
    
    print("\n🤖 LQR 밸런싱 + 점프")
    print("=" * 40)
    print("  I/K   : 전진/후진")
    print("  J/L   : 좌/우 회전")
    print("  Space  : 정지")
    print("  P      : 점프!")
    print("=" * 40)
    
    step = 0
    x_target = 0.0
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            global target_yaw_rate
            
            # --- 상태 추출 ---
            theta = get_pitch(data)
            theta_dot = data.qvel[4]
            x_pos = data.qpos[0]
            x_vel = data.qvel[0]
            torso_z = data.qpos[2]
            on_ground = check_wheel_contact(model, data)
            
            # --- 점프 컨트롤러 ---
            hip_cmd, jump_wheel = jump_ctrl.update(model, data, data.time, on_ground)
            
            # --- 목표 위치 (IDLE일 때만 업데이트) ---
            if jump_ctrl.phase == "IDLE":
                x_target += target_vel * model.opt.timestep
                target_yaw_rate *= 0.999
            
            # --- LQR ---
            
            state = np.array([
                theta,
                theta_dot,
                x_pos - x_target,
                x_vel - target_vel
            ])
            lqr_torque = -(K @ state)[0]
            lqr_torque = np.clip(lqr_torque, -8.0, 8.0)
            
            # --- 바퀴 토크 결정 ---
            if jump_wheel is not None:
                base_torque = jump_wheel  # 점프 중: 지정된 토크
            else:
                base_torque = lqr_torque  # 일반: LQR
            
            # --- 차동 구동 ---
            yaw_torque = target_yaw_rate * 0.5 if jump_ctrl.phase == "IDLE" else 0.0
            left_torque = np.clip(base_torque + yaw_torque, -8.0, 8.0)
            right_torque = np.clip(base_torque - yaw_torque, -8.0, 8.0)
            
            # --- 적용 ---
            data.ctrl[0] = hip_cmd       # left hip
            data.ctrl[1] = hip_cmd       # right hip
            data.ctrl[2] = left_torque   # left wheel
            data.ctrl[3] = right_torque  # right wheel
            
            # --- 시뮬레이션 ---
            mujoco.mj_step(model, data)
            step += 1
            
            # --- 로그 ---
            if step % 500 == 0:
                contact = "🟢" if on_ground else "🔴"
                print(f"  t={data.time:5.1f}s | "
                      f"{jump_ctrl.phase:8s} | "
                      f"pitch={np.degrees(theta):+5.1f}° | "
                      f"z={torso_z:.3f}m | "
                      f"hip={hip_cmd:4.0f}° | "
                      f"{contact} | "
                      f"τ={base_torque:+5.2f}")
            
            # --- 뷰어 ---
            if step % 2 == 0:
                viewer.sync()
                time.sleep(model.opt.timestep * 2)
    
    print("\n시뮬레이션 종료")


if __name__ == "__main__":
    main()
