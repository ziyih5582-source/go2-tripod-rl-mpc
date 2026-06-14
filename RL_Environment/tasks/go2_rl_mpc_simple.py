"""
go2_rl_mpc_simple.py — 简化版 RL+MPC 训练环境
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于 mpc_walk1.py 的简化 MPC 控制器，集成 RL 训练流程。

架构设计：
  RL Policy (高层): 输出 COM 偏移量 [dx, dy, dz] (3 维)
  MPC Controller (底层): 根据 COM 偏移计算关节控制 (12 维)

COM 输出范围：
  x: [-0.05, 0.05] 米 (前后)
  y: [-0.05, 0.05] 米 (左右)
  z: [-0.04, 0.02] 米 (上下)

使用 Go1 URDF 近似 Go2 真机
"""

import numpy as np
import os
import torch
import sys

sys.path.append("..")

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

# 导入简化版 MPC 控制器
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Go2_Controller.mpc_walk1 import MPCController, BalanceController

from .base.vec_task import VecTask


class Go2RLMPCSimple(VecTask):
    """
    简化版 RL+MPC 混合控制训练环境
    - RL 输出 COM 偏移，MPC 执行具体控制
    - 使用简化版 MPCController (mpc_walk1.py)
    """

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        
        # ==================== RL 输出 COM 范围（绝对位置，与mpc_walk1.py一致）====================
        self.com_x_range = [0.023, 0.27]   # 绝对X位置（与BalanceController.NOM_X配合）
        self.com_y_range = [0.063, 0.067]   # 绝对Y位置（与BalanceController.NOM_Y配合）
        self.com_z_range = [-0.01, 0.01]    # 绝对Z位置
        
        # COM 缩放参数：action [-1,1] → com_position [min, max]
        # formula: com = (action + 1) / 2 * (max - min) + min
        self.com_x_scale = (self.com_x_range[1] - self.com_x_range[0]) / 2
        self.com_x_bias = (self.com_x_range[1] + self.com_x_range[0]) / 2
        self.com_y_scale = (self.com_y_range[1] - self.com_y_range[0]) / 2
        self.com_y_bias = (self.com_y_range[1] + self.com_y_range[0]) / 2
        self.com_z_scale = (self.com_z_range[1] - self.com_z_range[0]) / 2
        self.com_z_bias = (self.com_z_range[1] + self.com_z_range[0]) / 2
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"] 
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"] 
        self.rew_scales["collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        self.rew_scales["tripod_stability"] = self.cfg["env"]["learn"].get("tripodStabilityRewardScale", 1.0)
        self.rew_scales["com_penalty"] = self.cfg["env"]["learn"].get("comPenaltyRewardScale", 0.1)
        self.rew_scales["action_smooth"] = self.cfg["env"]["learn"].get("actionSmoothRewardScale", 0.1)
        self.rew_scales["height_penalty"] = self.cfg["env"]["learn"].get("heightPenaltyRewardScale", 0.1)

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # 关键修改：numActions = 3 (COM 偏移), numObservations = 48
        self.cfg["env"]["numObservations"] = 48
        self.cfg["env"]["numActions"] = 3

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques_PD = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        
        # RL actions: 3 维 COM 偏移
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros_like(self.actions)
        
        # MPC 控制器 (简化版) - 使用共享控制器减少内存开销
        self.mpc_controller = MPCController(
            num_envs=self.num_envs, 
            dt=self.dt, 
            device=self.device,
            use_rl_mode=True
        )
        
        # IMU 平衡控制器 - 使用共享控制器
        self.balance_controller = BalanceController(device=self.device)

        # 步态参数
        self.GAIT_PERIOD = 0.6
        self.STEP_HEIGHT = 0.07
        self.FR_LIFT_ALPHA = 1.0  # FR 腿抬起
        
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        if self.cfg["env"].get("flat_ground", True):
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.static_friction = self.plane_static_friction
            plane_params.dynamic_friction = self.plane_dynamic_friction
            plane_params.restitution = self.plane_restitution
            self.gym.add_ground(self.sim, plane_params)
        else:
            from RL_Environment.sim_utils import add_random_uniform_terrain
            add_random_uniform_terrain(self.gym, self.sim)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = self.cfg["env"].get("urdf_file", "go1_description/urdf/go1.urdf")

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.01
        asset_options.use_mesh_materials = True

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        extremity_name = "foot"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "thigh" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        hip_names = [s for s in body_names if "hip" in s]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][i] = 0.0
            dof_props['damping'][i] = 0.0

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.actor_handles = []
        self.envs = []

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, start_pose, "robot", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)
            self.envs.append(env_ptr)
            self.actor_handles.append(robot_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hip_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "trunk")

    def pre_physics_step(self, actions):
        """
        RL 动作 → COM 偏移 → MPC 控制 → 关节力矩
        """
        self.actions = actions.clone().to(self.device)
        
        # 1. RL 动作解码：[-1, 1] → COM 偏移
        com_x = self.actions[:, 0:1] * self.com_x_scale + self.com_x_bias
        com_y = self.actions[:, 1:2] * self.com_y_scale + self.com_y_bias
        com_z = self.actions[:, 2:3] * self.com_z_scale + self.com_z_bias
        
        # 2. 批量处理 MPC 控制 - 使用共享控制器
        com_offset = torch.cat([com_x, com_y, com_z], dim=1)  # [num_envs, 3]
        
        # 使用共享控制器计算所有环境的关节目标
        jpos_target, jvel_target = self.mpc_controller.compute_control(
            dof_pos=self.dof_pos,
            com_offset=com_offset,
            vx_cmd=self.commands_x.mean().item(),  # 使用平均速度
            stepping=True,
            fr_lift_alpha=self.FR_LIFT_ALPHA,
        )
        
        # 3. PD 控制计算力矩（批量）
        pos_err = jpos_target - self.dof_pos
        vel_err = jvel_target - self.dof_vel
        self.torques = self.Kp * pos_err + self.Kd * vel_err
        self.torques = torch.clamp(self.torques, -50.0, 50.0)
        self.torques_PD = self.torques.clone()
        
        # 4. 下发力矩
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        # 将标量转换为 tensor (使用标量张量避免索引问题)
        base_index_t = torch.tensor(self.base_index, device=self.device, dtype=torch.long)
        max_episode_length_t = torch.tensor(self.max_episode_length, device=self.device, dtype=torch.long)
        
        self.rew_buf[:], self.reset_buf[:] = compute_robot_reward_v2(
            self.root_states,
            self.commands,
            self.torques_PD,
            self.contact_forces,
            self.knee_indices,
            self.hip_indices,
            self.progress_buf,
            self.rew_scales["lin_vel_xy"],
            self.rew_scales["ang_vel_z"],
            self.rew_scales["lin_vel_z"],
            self.rew_scales["ang_vel_xy"],
            self.rew_scales["collision"],
            self.rew_scales["torque"],
            self.rew_scales["com_penalty"],
            self.rew_scales["action_smooth"],
            self.rew_scales["height_penalty"],
            self.rew_scales["tripod_stability"],
            base_index_t,
            max_episode_length_t,
            self.dof_pos,
            self.default_dof_pos,
            actions,
            self.last_actions,
        )
        self.last_actions[:] = actions[:]

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 将标量转换为 tensor
        lin_vel_scale_t = torch.tensor(self.lin_vel_scale, device=self.device, dtype=torch.float32)
        ang_vel_scale_t = torch.tensor(self.ang_vel_scale, device=self.device, dtype=torch.float32)
        dof_pos_scale_t = torch.tensor(self.dof_pos_scale, device=self.device, dtype=torch.float32)
        dof_vel_scale_t = torch.tensor(self.dof_vel_scale, device=self.device, dtype=torch.float32)

        self.obs_buf[:] = compute_robot_observations_v2(
            self.root_states,
            self.commands,
            self.dof_pos,
            self.default_dof_pos,
            self.dof_vel,
            self.gravity_vec,
            self.actions,
            self.last_actions,
            lin_vel_scale_t,
            ang_vel_scale_t,
            dof_pos_scale_t,
            dof_vel_scale_t
        )

    def reset_idx(self, env_ids):
        # 刷新物理状态
        self.gym.refresh_dof_state_tensor(self.sim)
        
        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        # 重置 MPC 控制器
        self.mpc_controller.phase[env_ids] = 0.0
        self.balance_controller.reset()

        # 使用批量设置而不是 indexed 设置
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_root_states))

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

        # 速度命令：缓慢行走
        self.commands_x[env_ids] = torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # 重置 last_actions
        self.last_actions[env_ids] = 0.0


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_robot_reward_v2(
    root_states,
    commands,
    torques,
    contact_forces,
    knee_indices,
    hip_indices,
    episode_lengths,
    rew_lin_vel_xy_scale: float,
    rew_ang_vel_z_scale: float,
    rew_lin_vel_z_scale: float,
    rew_ang_vel_xy_scale: float,
    rew_collision_scale: float,
    rew_torque_scale: float,
    rew_com_penalty_scale: float,
    rew_action_smooth_scale: float,
    rew_height_penalty_scale: float,
    rew_tripod_stability_scale: float,
    base_index,
    max_episode_length,
    dof_pos,
    default_dof_pos,
    actions,
    last_actions,
):
    """
    改进版奖励函数
    """
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    base_pos = root_states[:, 0:3]

    # 速度跟踪奖励（XY平面）
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * rew_lin_vel_xy_scale
    
    # 偏航角速度奖励
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * rew_ang_vel_z_scale

    # 垂直速度惩罚
    rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * rew_lin_vel_z_scale
    
    # 横滚/俯仰角速度惩罚（稳定性）
    rew_ang_vel_xy = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1) * rew_ang_vel_xy_scale

    # 碰撞惩罚
    knee_contact = torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1.
    hip_contact = torch.norm(contact_forces[:, hip_indices, :], dim=2) > 1.
    rew_collision = torch.sum(knee_contact, dim=1) * rew_collision_scale + torch.sum(hip_contact, dim=1) * rew_collision_scale

    # 力矩惩罚
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_torque_scale
    
    # COM 平滑性惩罚（鼓励平滑的 COM 变化）
    com_penalty = torch.sum(torch.square(actions), dim=1) * rew_com_penalty_scale
    
    # 动作平滑性惩罚（鼓励平滑的动作变化）
    action_smooth_penalty = torch.sum(torch.square(actions - last_actions), dim=1) * rew_action_smooth_scale
    
    # 高度惩罚（鼓励保持合理高度）
    height_error = torch.square(base_pos[:, 2] - 0.3)  # 目标高度 0.3m
    height_penalty = height_error * rew_height_penalty_scale
    
    # 三足稳定性奖励：鼓励 roll/pitch 保持平稳
    roll_pitch_error = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
    tripod_stability = torch.exp(-roll_pitch_error / 0.5) * rew_tripod_stability_scale

    total_reward = (rew_lin_vel_xy + rew_ang_vel_z + tripod_stability - 
                    rew_lin_vel_z - rew_ang_vel_xy - rew_torque - 
                    rew_collision - com_penalty - action_smooth_penalty - height_penalty)
    total_reward = torch.clip(total_reward, 0., None)
    
    # reset conditions
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(knee_contact, dim=1)
    reset = reset | torch.any(hip_contact, dim=1)
    reset = reset | (base_pos[:, 2] < 0.1)  # 高度过低
    reset = reset | (base_pos[:, 2] > 0.6)  # 高度过高
    time_out = episode_lengths > max_episode_length
    reset = reset | time_out

    return total_reward.detach(), reset


@torch.jit.script
def compute_robot_observations_v2(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                last_actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):
    """
    改进版观测函数：48 维
    - base_lin_vel(3) + base_ang_vel(3) + gravity(3) + commands(3)
    - dof_pos(12) + dof_vel(12) + last_actions(3) + base_height(1)
    + actions(3) = 43维
    
    实际返回 48 维观测：
    base_pos(3) + base_lin_vel(3) + base_ang_vel(3) + gravity(3) + commands(3) + 
    dof_pos_scaled(12) + dof_vel_scaled(12) + last_actions(3) + base_height(1)
    """
    base_quat = root_states[:, 3:7]
    base_pos = root_states[:, 0:3]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    
    # 重力方向
    gravity = quat_rotate_inverse(base_quat, gravity_vec)
    
    # 关节位置缩放
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale
    
    # 命令缩放
    commands_scaled = commands * torch.stack([lin_vel_scale, lin_vel_scale, ang_vel_scale]).unsqueeze(0)
    
    # 高度归一化
    base_height = (root_states[:, 2:3] - 0.3) / 0.3  # 归一化到 [-1, 1] 范围
    
    # 观测拼接：48 维
    # base_pos(3) + base_lin_vel(3) + base_ang_vel(3) + gravity(3) + commands(3) = 15
    # dof_pos_scaled(12) + dof_vel_scaled(12) + last_actions(3) + base_height(1) = 28
    # actions(3) = 3
    # Total = 46 (去除重复计算)
    
    # 添加额外观测以达到 48 维
    # 观测拼接：48 维
    # base_pos(3) + base_lin_vel(3) + base_ang_vel(3) + gravity(3) + commands(3) = 15
    # dof_pos_scaled(12) + dof_vel_scaled(12) + last_actions(3) + base_height(1) = 28
    # actions(3) = 3
    # Total = 46, 需要添加 2 维
    
    # 计算 feet contact 状态（简化版：使用 base_height 的变化率作为代理）
    contact_proxy = (base_height - 0.5).abs()  # 接近 0 表示可能在地面
    contact_proxy = contact_proxy.repeat(1, 2)  # 2 维
    
    obs = torch.cat((
        base_pos,                    # 3
        base_lin_vel,                # 3
        base_ang_vel,                # 3
        gravity,                     # 3
        commands_scaled,             # 3
        dof_pos_scaled,              # 12
        dof_vel * dof_vel_scale,     # 12
        last_actions,                # 3
        base_height,                 # 1
        actions,                     # 3
        contact_proxy,               # 2 (填充到 48)
    ), dim=-1)

    return obs