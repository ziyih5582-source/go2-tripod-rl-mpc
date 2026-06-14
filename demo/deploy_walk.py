# deploy_walk1.py — RL+MPC 三足行走部署脚本
"""
Go2 三足平衡站立 + 行走，使用 RL 控制 COM 偏移

架构：
  RL Policy → COM偏移 [dx, dy, dz] → MPCController → 关节目标 → PD控制 → 真机

使用方法：
  python deploy_walk1.py                          # 纯MPC模式
  python deploy_walk1.py --rl                     # RL+MPC模式（默认加载model_1000.pt）
  python deploy_walk1.py --rl --checkpoint=/path/to/model.pt  # 指定模型路径
  python deploy_walk1.py enp129s0                  # 指定网络接口（纯MPC模式）
  python deploy_walk1.py enp129s0 --rl            # 指定网络接口 + RL模式
  python deploy_walk1.py --interface enp129s0    # 使用 --interface 选项

按键说明：
  [Enter] → 开始初始化（机器狗站起）
  [T]     → 三足静态站立
  [G]     → 三足行走（RL控制或MPC控制）
  [R]     → 切换 RL/MPC 控制模式
  [Space] → 阻尼急停
  [Q]     → 退出
"""

# ISAACGYM MUST be imported BEFORE torch to avoid import order error
import sys
isaacgym_path = '/home/huangziyi2025/下载/IsaacGym_Preview_4_Package/isaacgym/python'
if isaacgym_path not in sys.path:
    sys.path.insert(0, isaacgym_path)
import isaacgym  # MUST be imported before torch

import math
import time
import select
import tty
import termios
import numpy as np
import torch

from unitree_sdk2py.core.channel import (ChannelPublisher, ChannelSubscriber,
                                          ChannelFactoryInitialize)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import unitree_legged_const as go2
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

from mpc_walk import MPCController, BalanceController
from RL_Environment.WeightPolicy import WeightPolicy


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════
def smooth_step(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 * (1.0 - math.cos(math.pi * t))


# ═══════════════════════════════════════════════════════════════
#  RL COM 输出范围（与训练环境和mpc_walk1.py一致 - 绝对位置）
# ═══════════════════════════════════════════════════════════════
RL_COM_X_RANGE = [0.023, 0.27]
RL_COM_Y_RANGE = [0.063, 0.067]
RL_COM_Z_RANGE = [-0.01, 0.01]


# ═══════════════════════════════════════════════════════════════
#  关节索引映射
# ═══════════════════════════════════════════════════════════════
SDK_TO_CTRL = [3, 4, 5,
               0, 1, 2,
               9, 10, 11,
               6, 7, 8]

CTRL_TO_SDK = [3, 4, 5,
               0, 1, 2,
               9, 10, 11,
               6, 7, 8]


def sdk_to_ctrl(sdk_vals: list) -> list:
    return [sdk_vals[SDK_TO_CTRL[i]] for i in range(12)]


def ctrl_to_sdk(ctrl_vals: list) -> list:
    sdk = [0.0] * 12
    for ctrl_i, val in enumerate(ctrl_vals):
        sdk[CTRL_TO_SDK[ctrl_i]] = val
    return sdk


# ═══════════════════════════════════════════════════════════════
#  关节角预设
# ═══════════════════════════════════════════════════════════════
NEUTRAL_CTRL = [0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8]


# ═══════════════════════════════════════════════════════════════
#  键盘输入
# ═══════════════════════════════════════════════════════════════
class KeyboardInput:
    def __init__(self):
        self.old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def get_char(self):
        if select.select([sys.stdin], [], [], 0.05)[0]:
            return sys.stdin.read(1)
        return None

    def close(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)


# ═══════════════════════════════════════════════════════════════
#  RL观测构建辅助
# ═══════════════════════════════════════════════════════════════
class RLObserver:
    """构建RL网络所需的观测向量（与训练环境一致）- 支持 39 维 COM 输出模式"""
    
    def __init__(self, num_obs=39):
        """
        Args:
            num_obs: 观测维度，39 用于 COM 输出模式，48 用于原始模式
        """
        self.num_obs = num_obs
        
        # 观测缩放参数（与 go2_rl_mpc.yaml 一致）
        self.lin_vel_scale = 2.0
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        
        # COM输出范围（与训练环境和mpc_walk1.py一致 - 绝对位置）
        self.com_x_range = [0.023, 0.27]
        self.com_y_range = [0.063, 0.067]
        self.com_z_range = [-0.01, 0.01]
        self.com_x_scale = (self.com_x_range[1] - self.com_x_range[0]) / 2
        self.com_x_bias = (self.com_x_range[1] + self.com_x_range[0]) / 2
        self.com_y_scale = (self.com_y_range[1] - self.com_y_range[0]) / 2
        self.com_y_bias = (self.com_y_range[1] + self.com_y_range[0]) / 2
        self.com_z_scale = (self.com_z_range[1] - self.com_z_range[0]) / 2
        self.com_z_bias = (self.com_z_range[1] + self.com_z_range[0]) / 2
        
        # 状态估计（用于计算速度）
        self.last_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_rpy = np.zeros(3)
        self.last_lin_vel = np.zeros(3)
        self.last_dof_pos = np.zeros(12)
        self.last_action = np.zeros(3)
        
        # 默认关节角度（与训练环境一致）
        self.default_dof_pos = np.array([
            0.0, 0.9, -1.8,   # FL
            0.0, 0.9, -1.8,   # FR
            0.0, 0.9, -1.8,   # RL
            0.0, 0.9, -1.8,   # RR
        ])
        
        # 用于低通滤波的变量
        self.filtered_ang_vel = np.zeros(3)
        self.filter_alpha = 0.7  # 低通滤波系数
    
    def quat_rotate_inverse(self, q, v):
        """四元数旋转逆运算"""
        q_w = q[0]
        q_vec = q[1:]
        
        v_new = 2.0 * np.cross(q_vec, v)
        v_new = v_new + (1.0 - np.dot(q_vec, q_vec)) * v
        v_new = v_new + 2.0 * q_w * np.cross(q_w * v, q_vec)
        return v_new
    
    def compute_observations(self, imu_state, dof_states, commands):
        """
        构建观测向量（与训练环境一致）
        
        39维观测构成（num_obs=39）- COM 输出模式：
          base_pos(3) + base_lin_vel(3) + base_ang_vel(3) + commands(3) = 12
          + dof_pos_scaled(12) + dof_vel_scaled(12) + actions(3) = 27
          Total = 39维
        
        48维观测构成（num_obs=48）- 原始模式：
          base_pos(3) + base_lin_vel(3) + base_ang_vel(3) + gravity(3) + commands(3) = 15
          + dof_pos_scaled(12) + dof_vel_scaled(12) + last_actions(3) + base_height(1) = 28
          + actions(3) + contact_proxy(2) = 5
          Total = 48维
        """
        dt = 0.02
        
        # 读取IMU数据
        quat = np.array([imu_state.quaternion[3],  # w
                        imu_state.quaternion[0],   # x
                        imu_state.quaternion[1],   # y
                        imu_state.quaternion[2]])  # z
        
        rpy = np.array([imu_state.rpy[0], imu_state.rpy[1], imu_state.rpy[2]])
        gyro = np.array([imu_state.gyroscope[0], imu_state.gyroscope[1], imu_state.gyroscope[2]])
        
        # 计算角速度（机体坐标系）- 使用低通滤波平滑
        delta_rpy = rpy - self.last_rpy
        raw_ang_vel = delta_rpy / dt
        self.filtered_ang_vel = self.filter_alpha * raw_ang_vel + (1 - self.filter_alpha) * self.filtered_ang_vel
        self.last_rpy = rpy.copy()
        ang_vel_body = self.filtered_ang_vel.copy()
        
        # 假设速度（基于积分的简化模型）
        # 在实际真机上，这里应该用外部定位系统或视觉里程计
        # 这里用角速度积分作为代理，并使用低通滤波
        lin_vel_proxy = self.last_lin_vel + ang_vel_body * 0.01  # 简化的速度模型
        lin_vel_proxy = np.clip(lin_vel_proxy, -1.0, 1.0)  # 限制速度范围
        self.last_lin_vel = lin_vel_proxy * 0.95  # 衰减因子防止积分漂移
        lin_vel = self.last_lin_vel.copy()
        
        # 重力方向（机体坐标系）- 用于 48 维观测
        gravity_body = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        # base_pos（假设高度固定，根据陀螺仪估算）
        base_height_est = 0.35 - (rpy[1] * 0.1)  # 简化的pitch影响估算
        base_pos = np.array([0.0, 0.0, base_height_est])
        
        # 关节位置和速度
        dof_pos = np.array([dof_states[i] for i in range(12)])
        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale
        
        dof_vel = (dof_pos - self.last_dof_pos) / dt
        self.last_dof_pos = dof_pos.copy()
        dof_vel_scaled = dof_vel * self.dof_vel_scale
        
        # 命令缩放
        commands_scaled = commands * np.array([self.lin_vel_scale, 
                                               self.lin_vel_scale, 
                                               self.ang_vel_scale])
        
        # 根据 num_obs 构建观测
        if self.num_obs == 39:
            # 39 维观测（与 go2_rl_mpc.yaml 一致）- COM 输出模式
            obs = np.concatenate([
                base_pos,                              # 3
                lin_vel * self.lin_vel_scale,          # 3
                ang_vel_body * self.ang_vel_scale,    # 3
                commands_scaled,                       # 3
                dof_pos_scaled,                        # 12
                dof_vel_scaled,                        # 12
                self.last_action,                      # 3 (actions = last COM offset)
            ])  # Total = 39
        else:
            # 48 维观测（与 go2_rl_mpc_simple.yaml 一致）
            base_height = (base_height_est - 0.3) / 0.3  # 归一化到[-1, 1]
            contact_proxy = np.array([0.5, 0.5])
            obs = np.concatenate([
                base_pos,                              # 3
                lin_vel * self.lin_vel_scale,          # 3
                ang_vel_body * self.ang_vel_scale,    # 3
                gravity_body,                          # 3
                commands_scaled,                       # 3
                dof_pos_scaled,                        # 12
                dof_vel_scaled,                        # 12
                self.last_action,                      # 3
                np.array([base_height]),               # 1
                self.last_action,                      # 3 (actions = last_actions for first step)
                contact_proxy,                         # 2
            ])  # Total = 48
        
        return obs.astype(np.float32)
    
    def decode_action(self, action):
        """将网络输出[-1,1]解码为COM偏移"""
        com_x = action[0] * self.com_x_scale + self.com_x_bias
        com_y = action[1] * self.com_y_scale + self.com_y_bias
        com_z = action[2] * self.com_z_scale + self.com_z_bias
        return np.array([com_x, com_y, com_z])


# ═══════════════════════════════════════════════════════════════
#  部署主控类
# ═══════════════════════════════════════════════════════════════
class MPCDeployer:
    # PD增益
    KP_STAND  = 50.0
    KD_STAND  = 2.5
    KP_TRIPOD = 55.0
    KD_TRIPOD = 3.0
    KP_WALK = 45.0
    KD_WALK = 2.5

    # 各阶段时长
    T_INITIAL_STAND = 2.0
    T_PRE_SHIFT     = 1.0
    T_LIFT_FR       = 1.5

    # 行走参数
    WALK_VX      = 0.3     # 较慢的前进速度
    WALK_STEP_H  = 0.025
    WALK_PERIOD  = 1.2

    def __init__(self, use_rl=False, rl_checkpoint=None):
        self.dt     = 0.02
        self.device = 'cpu'
        self.use_rl = use_rl
        
        # 运动学控制器
        self.ctrl = MPCController(num_envs=1, dt=self.dt, device=self.device, use_rl_mode=use_rl)
        # IMU 闭环平衡
        self.balance_ctrl = BalanceController(device=self.device)
        
        # RL Policy
        self.rl_policy = None
        if use_rl and rl_checkpoint:
            try:
                self.rl_policy = WeightPolicy(
                    task="go2_rl_mpc_simple",
                    checkpoint=rl_checkpoint,
                    num_envs=1,
                    device='cpu',
                    use_com_output=True,
                    num_obs=48,  # 训练使用 48 维观测 (与 go2_rl_mpc_simple.yaml 一致)
                    num_actions=3
                )
                print(f"[INFO] RL Policy 加载成功：{rl_checkpoint}")
            except Exception as e:
                print(f"[WARN] RL Policy 加载失败：{e}")
                print("[INFO] 回退到纯 MPC 模式")
                self.use_rl = False
        
        # RL观测器 - 使用 48 维观测（与训练配置一致）
        self.rl_observer = RLObserver(num_obs=48) if use_rl else None
        
        # 命令（速度目标）
        self.commands = np.array([0.0, 0.0, 0.0])  # [vx, vy, yaw_rate]
        
        self.low_cmd   = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc       = CRC()

        self.state   = "DAMP"
        self.running = True

        # 插值相关
        self.interp_from  = [0.0] * 12
        self.interp_to    = [0.0] * 12
        self.interp_steps = 0
        self.interp_cur   = 0
        self.first_run    = True
        self.last_sdk_pos = [0.0] * 12

        # 状态过渡计数器
        self.pre_shift_steps = int(self.T_PRE_SHIFT / self.dt)
        self.pre_shift_cur   = 0
        self.lift_fr_steps = int(self.T_LIFT_FR / self.dt)
        self.lift_fr_cur   = 0

        # 行走相关
        self.vx_cmd = 0.0
        self.walk_ramp_steps = int(1.0 / self.dt)
        self.walk_ramp_cur   = 0
        self.target_vx = self.WALK_VX
        
        print(f"[INFO] RL模式: {'启用' if use_rl else '禁用'}")
        print(f"[INFO] RL模型: {rl_checkpoint if rl_checkpoint else '未指定'}")

    # ─── 初始化通信 ──────────────────────────
    def Init(self):
        self.low_cmd.head[0]    = 0xFE
        self.low_cmd.head[1]    = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio       = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q    = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp   = 0
            self.low_cmd.motor_cmd[i].dq   = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd   = 0
            self.low_cmd.motor_cmd[i].tau  = 0

        self.lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_pub.Init()
        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init(self._on_low_state, 10)

        print("[INFO] 正在释放 Sport 运动服务...")
        sc  = SportClient();  sc.SetTimeout(5.0);  sc.Init()
        msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
        status, result = msc.CheckMode()
        while result['name']:
            print(f"[INFO] 活动服务: {result['name']}，趴下并释放...")
            sc.StandDown(); msc.ReleaseMode(); time.sleep(1)
            status, result = msc.CheckMode()
        print("[INFO] 底层直连模式就绪。")

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg

    def Start(self):
        self._thread = RecurrentThread(
            interval=self.dt, target=self._loop, name="mpc_loop")
        self._thread.Start()

    # ─── 插值辅助 ─────────────────────────────
    def _start_interp(self, target_ctrl: list, steps: int):
        self.interp_from  = list(self.last_sdk_pos)
        self.interp_to    = ctrl_to_sdk(target_ctrl)
        self.interp_steps = steps
        self.interp_cur   = 0

    def _send_interp(self, kp: float, kd: float) -> bool:
        t = min(self.interp_cur / max(self.interp_steps - 1, 1), 1.0)
        t = smooth_step(t)
        pos = [(1.0 - t) * self.interp_from[i] + t * self.interp_to[i]
               for i in range(12)]
        for i in range(12):
            self.low_cmd.motor_cmd[i].q   = pos[i]
            self.low_cmd.motor_cmd[i].dq  = 0.0
            self.low_cmd.motor_cmd[i].kp  = kp
            self.low_cmd.motor_cmd[i].kd  = kd
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.last_sdk_pos = pos
        self.interp_cur  += 1
        return self.interp_cur >= self.interp_steps

    # ─── 读取关节角 ────────────────────────────
    def _read_ctrl_dof(self) -> torch.Tensor:
        sdk_pos = [self.low_state.motor_state[i].q for i in range(12)]
        return torch.tensor([sdk_to_ctrl(sdk_pos)],
                            dtype=torch.float32, device=self.device)
    
    def _read_dof_np(self) -> np.ndarray:
        """读取关节角为numpy数组"""
        return np.array([self.low_state.motor_state[i].q for i in range(12)])

    # ─── 下发控制命令 ────────────────────────────
    def _send_ctrl(self, jpos_ctrl, jvel_ctrl, kp: float, kd: float):
        pos_sdk = ctrl_to_sdk(jpos_ctrl[0].tolist())
        vel_sdk = ctrl_to_sdk(jvel_ctrl[0].tolist())
        self.last_sdk_pos = list(pos_sdk)
        for i in range(12):
            self.low_cmd.motor_cmd[i].q   = pos_sdk[i]
            self.low_cmd.motor_cmd[i].dq  = vel_sdk[i]
            self.low_cmd.motor_cmd[i].kp  = kp
            self.low_cmd.motor_cmd[i].kd  = kd
            self.low_cmd.motor_cmd[i].tau = 0.0

    # ─── 获取RL输出的COM偏移 ────────────────────────────
    def _get_rl_com_offset(self) -> np.ndarray:
        """使用RL Policy获取COM偏移"""
        if self.rl_policy is None or self.rl_observer is None:
            return np.array([0.0, BalanceController.NOM_Y, 0.0])
        
        try:
            # 构建观测
            imu = self.low_state.imu_state
            dof_pos = self._read_dof_np()
            
            obs = self.rl_observer.compute_observations(imu, dof_pos, self.commands)
            
            # 更新RL Policy的观测
            self.rl_policy.obs = torch.from_numpy(obs.reshape(1, -1)).float()
            
            # 获取动作
            action = self.rl_policy.step()  # 返回numpy数组 [3]
            
            # 更新last_action
            self.rl_observer.last_action = action.copy()
            
            # 解码为COM偏移
            com_offset = self.rl_observer.decode_action(action)
            
            print(f"[RL] COM偏移: [{com_offset[0]:+.4f}, {com_offset[1]:+.4f}, {com_offset[2]:+.4f}]")
            
            return com_offset
            
        except Exception as e:
            print(f"[WARN] RL推理失败: {e}")
            return np.array([0.0, BalanceController.NOM_Y, 0.0])

    # ─── 主循环 ──────────────────────────────
    def _loop(self):
        if self.low_state is None:
            return

        # 首次自动站起
        if self.first_run:
            sdk_now = [self.low_state.motor_state[i].q for i in range(12)]
            self.last_sdk_pos = list(sdk_now)
            self._start_interp(NEUTRAL_CTRL, int(self.T_INITIAL_STAND / self.dt))
            self.state     = "INITIAL_STAND"
            self.first_run = False
            print("[STATE] INITIAL_STAND：正在缓慢站起（约2秒）...")

        # ── DAMP ──────────────────────────────────────
        if self.state == "DAMP":
            for i in range(12):
                self.low_cmd.motor_cmd[i].q   = go2.PosStopF
                self.low_cmd.motor_cmd[i].dq  = go2.VelStopF
                self.low_cmd.motor_cmd[i].kp  = 0.0
                self.low_cmd.motor_cmd[i].kd  = 2.0
                self.low_cmd.motor_cmd[i].tau = 0.0

        # ── 初始站立 / 放下 FR ────────────────────────
        elif self.state in ("INITIAL_STAND", "LOWER_FR"):
            done = self._send_interp(self.KP_STAND, self.KD_STAND)
            if done:
                if self.state == "INITIAL_STAND":
                    self.state = "STAY_STILL"
                    print("[STATE] 四足站立就绪！")
                    print("  按 [T] 三足站立  按 [G] 三足行走  按 [R] 切换RL模式")
                elif self.state == "LOWER_FR":
                    self.state = "STAY_STILL"
                    print("[STATE] FR 腿已放下，回到四足站立。")

        # ── 四足静态保持 ───────────────────────────────
        elif self.state == "STAY_STILL":
            target_sdk = ctrl_to_sdk(NEUTRAL_CTRL)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q   = target_sdk[i]
                self.low_cmd.motor_cmd[i].dq  = 0.0
                self.low_cmd.motor_cmd[i].kp  = self.KP_STAND
                self.low_cmd.motor_cmd[i].kd  = self.KD_STAND
                self.low_cmd.motor_cmd[i].tau = 0.0

        # ── PRE_SHIFT：重心平移进支撑三角 ──────────────
        elif self.state == "PRE_SHIFT":
            t = min(self.pre_shift_cur / max(self.pre_shift_steps - 1, 1), 1.0)
            alpha = smooth_step(t)
            com_y = BalanceController.NOM_Y * alpha
            com_offset = torch.tensor([[0.0, com_y, 0.0]],
                                      dtype=torch.float32, device=self.device)
            ctrl_dof = self._read_ctrl_dof()
            jpos_ctrl, jvel_ctrl = self.ctrl.compute_control(
                dof_pos      = ctrl_dof,
                com_offset   = com_offset,
                vx_cmd       = 0.0,
                stepping     = False,
                fr_lift_alpha= 0.0,
            )
            self._send_ctrl(jpos_ctrl, jvel_ctrl, self.KP_STAND, self.KD_STAND)
            self.pre_shift_cur += 1
            if self.pre_shift_cur >= self.pre_shift_steps:
                self.lift_fr_cur = 0
                self.state = "LIFT_FR"
                print("[STATE] LIFT_FR：重心已入支撑三角，正在缓慢抬起 FR 腿...")

        # ── LIFT_FR：FR 腿平滑抬起 ───────────────────────
        elif self.state == "LIFT_FR":
            t = min(self.lift_fr_cur / max(self.lift_fr_steps - 1, 1), 1.0)
            fr_alpha = smooth_step(t)
            com_offset = torch.tensor(
                [[0.0, BalanceController.NOM_Y, 0.0]],
                dtype=torch.float32, device=self.device)
            ctrl_dof = self._read_ctrl_dof()
            jpos_ctrl, jvel_ctrl = self.ctrl.compute_control(
                dof_pos      = ctrl_dof,
                com_offset   = com_offset,
                vx_cmd       = 0.0,
                stepping     = False,
                fr_lift_alpha= fr_alpha,
            )
            self._send_ctrl(jpos_ctrl, jvel_ctrl, self.KP_STAND, self.KD_STAND)
            self.lift_fr_cur += 1
            if self.lift_fr_cur >= self.lift_fr_steps:
                self.balance_ctrl.reset()
                self.state = "TRIPOD_STAND"
                print("[STATE] TRIPOD_STAND：三足闭环平衡就绪！")

        # ── TRIPOD_STAND：三足闭环平衡站立 ──────
        elif self.state == "TRIPOD_STAND":
            # 获取COM偏移（RL或MPC）
            if self.use_rl:
                rl_com = self._get_rl_com_offset()
                # 混合控制：x方向固定为经验值，y/z使用RL输出
                # 经验值：BalanceController.NOM_X = 0.025, NOM_Y = 0.065, NOM_Z = 0.00
                hybrid_com = np.array([
                    BalanceController.NOM_X,  # x方向：固定经验值
                    rl_com[1],                 # y方向：使用RL输出
                    rl_com[2]                  # z方向：使用RL输出
                ])
                com_offset = torch.tensor([hybrid_com], dtype=torch.float32, device=self.device)
                print(f"[RL-HYBRID] COM偏移: x={BalanceController.NOM_X:.4f} (固定), y={hybrid_com[1]:+.4f}, z={hybrid_com[2]:+.4f}")
            else:
                imu = self.low_state.imu_state
                roll = float(imu.rpy[0])
                pitch = float(imu.rpy[1])
                gyro_x = float(imu.gyroscope[0])
                gyro_y = float(imu.gyroscope[1])
                com_offset, _ = self.balance_ctrl.update(roll, pitch, gyro_x, gyro_y)

            ctrl_dof = self._read_ctrl_dof()
            jpos_ctrl, jvel_ctrl = self.ctrl.compute_control(
                dof_pos      = ctrl_dof,
                com_offset   = com_offset,
                vx_cmd       = 0.0,
                stepping     = False,
                fr_lift_alpha= 1.0,
            )
            self._send_ctrl(jpos_ctrl, jvel_ctrl, self.KP_TRIPOD, self.KD_TRIPOD)

        # ── WALK_START：行走软启动 ──────
        elif self.state == "WALK_START":
            # 获取COM偏移
            if self.use_rl:
                rl_com = self._get_rl_com_offset()
                # 混合控制：x方向固定为经验值，y/z使用RL输出
                hybrid_com = np.array([
                    BalanceController.NOM_X,  # x方向：固定经验值
                    rl_com[1],                 # y方向：使用RL输出
                    rl_com[2]                  # z方向：使用RL输出
                ])
                com_offset = torch.tensor([hybrid_com], dtype=torch.float32, device=self.device)
            else:
                imu = self.low_state.imu_state
                roll = float(imu.rpy[0])
                pitch = float(imu.rpy[1])
                gyro_x = float(imu.gyroscope[0])
                gyro_y = float(imu.gyroscope[1])
                com_offset, _ = self.balance_ctrl.update(roll, pitch, gyro_x, gyro_y)

            # 速度斜坡
            t = min(self.walk_ramp_cur / max(self.walk_ramp_steps - 1, 1), 1.0)
            cur_vx = self.target_vx * smooth_step(t)

            ctrl_dof = self._read_ctrl_dof()
            jpos_ctrl, jvel_ctrl = self.ctrl.compute_control(
                dof_pos      = ctrl_dof,
                com_offset   = com_offset,
                vx_cmd       = cur_vx,
                stepping     = True,
                fr_lift_alpha= 1.0,
            )
            self._send_ctrl(jpos_ctrl, jvel_ctrl, self.KP_WALK, self.KD_WALK)

            self.walk_ramp_cur += 1
            if self.walk_ramp_cur >= self.walk_ramp_steps:
                self.vx_cmd = self.target_vx
                self.state = "WALK"
                print(f"[STATE] WALK：软启动完成，RL={'启用' if self.use_rl else '禁用'}")

        # ── WALK：三足行走 ──────
        elif self.state == "WALK":
            # 获取COM偏移（RL或MPC）
            if self.use_rl:
                rl_com = self._get_rl_com_offset()
                # 混合控制：x方向固定为经验值，y/z使用RL输出
                hybrid_com = np.array([
                    BalanceController.NOM_X,  # x方向：固定经验值
                    rl_com[1],                 # y方向：使用RL输出
                    rl_com[2]                  # z方向：使用RL输出
                ])
                com_offset = torch.tensor([hybrid_com], dtype=torch.float32, device=self.device)
                ctrl_mode = "RL-HYBRID"
            else:
                imu = self.low_state.imu_state
                roll = float(imu.rpy[0])
                pitch = float(imu.rpy[1])
                gyro_x = float(imu.gyroscope[0])
                gyro_y = float(imu.gyroscope[1])
                com_offset, _ = self.balance_ctrl.update(roll, pitch, gyro_x, gyro_y)
                ctrl_mode = "MPC"

            ctrl_dof = self._read_ctrl_dof()
            jpos_ctrl, jvel_ctrl = self.ctrl.compute_control(
                dof_pos      = ctrl_dof,
                com_offset   = com_offset,
                vx_cmd       = self.vx_cmd,
                stepping     = True,
                fr_lift_alpha= 1.0,
            )
            self._send_ctrl(jpos_ctrl, jvel_ctrl, self.KP_WALK, self.KD_WALK)

        # 发布命令
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_pub.Write(self.low_cmd)

    # ─── 指令接口 ────────────────────────────────
    def cmd_tripod_stand(self):
        """三足站立"""
        if self.state not in ("STAY_STILL", "TRIPOD_STAND", "WALK"):
            print("[WARN] 请先完成初始站立")
            return

        self.ctrl.GAIT_PERIOD = 0.6
        self.ctrl.STEP_HEIGHT = 0.07
        self.ctrl.phase[0] = 0.65
        self.ctrl.reset(torch.tensor([0]))
        self.ctrl.phase[0] = 0.65

        self.pre_shift_steps = int(self.T_PRE_SHIFT / self.dt)
        self.pre_shift_cur   = 0
        self.state = "PRE_SHIFT"
        print(f"[STATE] PRE_SHIFT：正在将重心移向支撑三角形...")

    def cmd_walk(self):
        """三足行走"""
        if self.state not in ("TRIPOD_STAND", "WALK", "LIFT_FR"):
            print("[WARN] 请先按 [T] 进入三足站立状态")
            return

        self.ctrl.GAIT_PERIOD = self.WALK_PERIOD
        self.ctrl.STEP_HEIGHT = self.WALK_STEP_H
        self.vx_cmd = self.WALK_VX
        self.ctrl.phase[0] = 0.65
        self.ctrl.reset(torch.tensor([0]))
        self.ctrl.phase[0] = 0.65

        self.walk_ramp_cur = 0
        self.state = "WALK_START"
        print(f"[STATE] WALK_START：三足行走 RL={'启用' if self.use_rl else '禁用'}...")

    def cmd_toggle_rl(self):
        """切换RL模式"""
        self.use_rl = not self.use_rl
        mode = "RL" if self.use_rl else "MPC"
        print(f"[INFO] 控制模式切换为: {mode}")
        if self.state == "WALK":
            print("[INFO] 行走中切换，下次WALK生效")

    def cmd_damp(self):
        """阻尼急停"""
        self.ctrl.reset(torch.tensor([0]))
        self.balance_ctrl.reset()
        self.ctrl.GAIT_PERIOD = 0.6
        self.ctrl.STEP_HEIGHT = 0.07
        self.state = "DAMP"
        print("[STATE] 急停！进入阻尼模式。")


# ═══════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Go2 RL+MPC 部署')
    parser.add_argument('interface', nargs='?', type=str, default='', 
                        help='网络接口名称 (例如: enp129s0)')
    parser.add_argument('--interface', '--iface', '-i', dest='interface_alt', type=str, default=None,
                        help='网络接口名称 (可选的另一种写法)')
    parser.add_argument('--rl', action='store_true', help='启用RL控制模式')
    parser.add_argument('--checkpoint', type=str, default=None, help='RL模型路径')
    args = parser.parse_args()
    
    # 处理网络接口参数（positional或--interface均可）
    network_interface = args.interface if args.interface else (args.interface_alt or '')
    
    # 默认RL模型路径
    default_checkpoint = "runs/Go2RLMPCSimple/May15_18-14-35/model_2000.pt"
    
    if args.rl and args.checkpoint is None:
        args.checkpoint = default_checkpoint
    
    print("=" * 60)
    print("  Go2 三足平衡站立 & 行走 (RL+MPC)")
    print("=" * 60)
    print(f"  RL模式: {'启用' if args.rl else '禁用'}")
    if args.rl:
        print(f"  模型: {args.checkpoint}")
    print(f"  网络接口: {network_interface if network_interface else '自动检测'}")
    print("  按 [T] 三足站立  按 [G] 三足行走  按 [R] 切换RL  按 [Q] 退出")
    print("=" * 60)
    input("准备好后按 [Enter] 开始...")

    # 使用网络接口初始化通信
    if network_interface:
        print(f"[INFO] 使用网络接口: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)
    else:
        ChannelFactoryInitialize(0)

    dep = MPCDeployer(use_rl=args.rl, rl_checkpoint=args.checkpoint)
    dep.Init()
    dep.Start()

    kb = KeyboardInput()

    try:
        while dep.running:
            c = kb.get_char()
            if c:
                cu = c.upper()
                if cu == 'Q':
                    dep.cmd_damp()
                    dep.running = False
                elif c == ' ':
                    dep.cmd_damp()
                elif cu == 'T':
                    dep.cmd_tripod_stand()
                elif cu == 'G':
                    dep.cmd_walk()
                elif cu == 'R':
                    dep.cmd_toggle_rl()
            time.sleep(0.05)
    finally:
        kb.close()
        print("[INFO] 退出。")
