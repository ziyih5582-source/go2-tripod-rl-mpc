import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Import isaacgym first (before torch) to avoid import order error
# Add isaacgym path to sys.path if not already present
isaacgym_path = '/home/huangziyi2025/下载/IsaacGym_Preview_4_Package/isaacgym/python'
if isaacgym_path not in sys.path:
    sys.path.insert(0, isaacgym_path)

import isaacgym  # MUST be imported before torch

# Now import torch after isaacgym
import torch
import time
import numpy as np
from MPC_Controller.Parameters import Parameters
from MPC_Controller.utils import DTYPE
from MPC_Controller.common.StateEstimator import StateEstimate

from rsl_rl.modules import ActorCritic

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) # Under <RL_Environment>

class WeightPolicy:
    def __init__(self, 
                 task="Aliengo", 
                 checkpoint="runs/Aliengo/nn/Aliengo.pth",
                 num_envs=1,
                 device=None,
                 use_com_output=False,
                 num_obs=48,
                 num_actions=3):  # device: "cuda" or "cpu", None = auto-detect
        """
        Args:
            task: 任务名称
            checkpoint: 模型 checkpoint 路径
            num_envs: 环境数量
            device: 设备 (cuda/cpu)
            use_com_output: 是否使用 COM 输出模式 (3 维)
                False: 原始 12 维关节控制
                True: 3 维 COM 偏移输出
            num_obs: 观测维度 (默认 48)
            num_actions: 动作维度 (默认 3)
        """
        # COM 输出范围（绝对位置，与mpc_walk1.py一致）
        self.com_x_range = [0.023, 0.27]
        self.com_y_range = [0.063, 0.067]
        self.com_z_range = [-0.01, 0.01]
        
        # COM 缩放参数
        self.com_x_scale = (self.com_x_range[1] - self.com_x_range[0]) / 2
        self.com_x_bias = (self.com_x_range[1] + self.com_x_range[0]) / 2
        self.com_y_scale = (self.com_y_range[1] - self.com_y_range[0]) / 2
        self.com_y_bias = (self.com_y_range[1] + self.com_y_range[0]) / 2
        self.com_z_scale = (self.com_z_range[1] - self.com_z_range[0]) / 2
        self.com_z_bias = (self.com_z_range[1] + self.com_z_range[0]) / 2
        
        # 根据模式设置维度
        if use_com_output:
            self.num_actions = num_actions  # COM 偏移 [dx, dy, dz]
            self.num_obs = num_obs          # 观测维度
        else:
            self.num_actions = 12  # 原始 12 维关节控制
            self.num_obs = num_obs   # 观测维度
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.is_determenistic = True
        self.clip_actions = True
        
        # 观测缩放参数（与 go2_rl_mpc_simple.yaml 一致）
        self.lin_vel_scale = 2.0
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        
        # 默认关节角度（与训练环境一致）
        self.default_dof_pos = np.array([
            0.0, 0.9, -1.8,   # FL: hip, thigh, calf
            0.0, 0.9, -1.8,   # FR
            0.0, 0.9, -1.8,   # RL
            0.0, 0.9, -1.8,   # RR
        ])
        
        # 解析 checkpoint 路径
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(ROOT_DIR, checkpoint)
        
        # 直接使用 torch.load 加载模型（无需 Hydra）
        print(f"[INFO] Loading model from: {checkpoint}")
        print(f"[INFO] Model config: num_obs={self.num_obs}, num_actions={self.num_actions}")
        
        # 创建 ActorCritic 网络（使用 rsl_rl 默认配置）
        # 隐藏层配置
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # or 'relu', 'tanh'
        
        self.actor_critic = ActorCritic(
            self.num_obs,          # observation_dim
            self.num_obs,          # critic_obs_dim  
            self.num_actions,      # action_dim
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        ).to(self.device)
        
        # 加载 checkpoint
        try:
            loaded_dict = torch.load(checkpoint, map_location=self.device)
            if 'model_state_dict' in loaded_dict:
                self.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
            else:
                self.actor_critic.load_state_dict(loaded_dict)
            print(f"[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            raise

        self.actor_critic.eval()
        self.policy = self.actor_critic.act_inference

        self.num_agents = 1
        self.obs = torch.ones([self.num_agents, self.num_obs], 
                              requires_grad=False, dtype=torch.float, device=self.device)
        
        # 用于观测构建的状态
        self.last_action = np.zeros(3, dtype=np.float32)

    def step(self):
        obs = self._preproc_obs(self.obs)
        # get action

        t_start = time.time()
        with torch.no_grad():
            current_action = self.policy(obs.detach())
        if Parameters.policy_print_time:
            print("Model Inference Time: {:.5f}".format(time.time()-t_start))

        # clip actions to (-1, 1)
        if self.clip_actions:
            current_action = self._rescale_actions(
                -torch.ones_like(current_action, requires_grad=False, device=self.device), 
                torch.ones_like(current_action, requires_grad=False, device=self.device), 
                torch.clamp(current_action, -1.0, 1.0))

        # COM 输出模式：解码为实际 COM 偏移
        if self.num_actions == 3:
            # action [-1, 1] → com_offset [min, max]
            com_x = current_action[0, 0] * self.com_x_scale + self.com_x_bias
            com_y = current_action[0, 1] * self.com_y_scale + self.com_y_bias
            com_z = current_action[0, 2] * self.com_z_scale + self.com_z_bias
            return np.array([com_x.item(), com_y.item(), com_z.item()])
        else:
            # 原始 12 维关节控制
            actions_rescale = torch.mul(current_action, 
                                        torch.tensor(
                                        Parameters.MPC_param_scale,
                                        dtype=torch.float,
                                        device=self.device)).add(
                                        torch.tensor(
                                        Parameters.MPC_param_const,
                                        dtype=torch.float,
                                        device=self.device))
            return actions_rescale.detach().cpu().numpy()[0]

    def compute_observations(self, dof_states, se_result:StateEstimate, _commands, _actions):
        """
        构建观测向量（与训练环境 go2_rl_mpc.yaml 一致）
        
        39维观测构成（num_obs=39）：
          base_pos(3) + base_lin_vel(3) + base_ang_vel(3) + commands(3) = 12
          + dof_pos_scaled(12) + dof_vel_scaled(12) + actions(3) = 27
          Total = 39维
        
        48维观测构成（num_obs=48）：
          base_pos(3) + base_lin_vel(3) + base_ang_vel(3) + gravity(3) + commands(3) = 15
          + dof_pos_scaled(12) + dof_vel_scaled(12) + last_actions(3) + base_height(1) = 28
          + actions(3) + contact_proxy(2) = 5
          Total = 48维
        """
        # 获取机体状态
        base_pos = se_result.posBody.flatten() if hasattr(se_result, 'posBody') else np.array([0.0, 0.0, 0.35])
        base_quat = se_result.quat.flatten() if hasattr(se_result, 'quat') else np.array([1.0, 0.0, 0.0, 0.0])
        
        # 计算速度（机体坐标系）
        base_lin_vel = se_result.vBody.flatten() * self.lin_vel_scale
        base_ang_vel = se_result.omegaBody.flatten() * self.ang_vel_scale
        
        # 重力方向（机体坐标系）- 用于 48 维观测
        gravity_body = self._quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
        
        # 命令缩放
        commands = _commands * np.array([self.lin_vel_scale, 
                                         self.lin_vel_scale, 
                                         self.ang_vel_scale], 
                                         dtype=DTYPE)
        
        # 关节位置和速度
        dof_pos = np.array(dof_states["pos"]) if isinstance(dof_states["pos"], list) else dof_states["pos"]
        dof_vel = np.array(dof_states["vel"]) if isinstance(dof_states["vel"], list) else dof_states["vel"]
        
        # 关节位置缩放（相对于默认位置）
        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale
        dof_vel_scaled = dof_vel * self.dof_vel_scale
        
        # 根据 num_obs 构建观测
        if self.num_obs == 39:
            # 39 维观测（与 go2_rl_mpc.yaml 一致）- COM 输出模式
            observations = np.concatenate([
                base_pos,                              # 3
                base_lin_vel,                          # 3
                base_ang_vel,                          # 3
                commands,                             # 3
                dof_pos_scaled,                        # 12
                dof_vel_scaled,                        # 12
                _actions,                              # 3 (actions = COM offset)
            ])  # Total = 39
        elif self.num_obs == 48:
            # 48 维观测（与 go2_rl_mpc_simple.yaml 一致）
            base_height = (base_pos[2] - 0.3) / 0.3  # 归一化到 [-1, 1]
            contact_proxy = np.array([0.5, 0.5])
            observations = np.concatenate([
                base_pos,                              # 3
                base_lin_vel,                          # 3
                base_ang_vel,                          # 3
                gravity_body,                          # 3
                commands,                             # 3
                dof_pos_scaled,                        # 12
                dof_vel_scaled,                        # 12
                self.last_action,                      # 3
                np.array([base_height]),               # 1
                _actions,                              # 3
                contact_proxy,                         # 2
            ])  # Total = 48
        else:
            raise ValueError(f"Unsupported observation dimension: {self.num_obs}. Supported: 39, 48")
        
        obs_pad = np.expand_dims(observations.astype(np.float32), axis=0)
        self.obs = torch.from_numpy(obs_pad).to(self.device)
        
        # 更新 last_action
        self.last_action = _actions.copy() if hasattr(_actions, 'copy') else np.array(_actions)
    
    def _quat_rotate_inverse(self, q, v):
        """四元数旋转逆运算"""
        q_w = q[0]
        q_vec = q[1:]
        
        v_new = 2.0 * np.cross(q_vec, v)
        v_new = v_new + (1.0 - np.dot(q_vec, q_vec)) * v
        v_new = v_new + 2.0 * q_w * np.cross(q_w * v, q_vec)
        return v_new

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0

        return obs_batch

    def _rescale_actions(self, low, high, action):
        d = (high - low) / 2.0
        m = (high + low) / 2.0
        scaled_action =  action * d + m
        return scaled_action