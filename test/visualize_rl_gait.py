"""
visualize_rl_gait.py - 可视化训练好的 RL 模型步态
==================================================
使用方法：
    python visualize_rl_gait.py --checkpoint <模型路径> --robot <机器人类型>

参数说明：
    --checkpoint: 模型 checkpoint 路径（默认：runs/Go2RLMPC/nn/latest.pth）
    --robot: 机器人类型 (go1, go2, aliengo, a1)
    --use_rl: 是否使用 RL 模式（默认 True）
"""

import os
import inspect
import argparse
import numpy as np
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from MPC_Controller.utils import DTYPE
from MPC_Controller.Parameters import Parameters
from MPC_Controller.robot_runner.RobotRunnerFSM import RobotRunnerFSM
from MPC_Controller.robot_runner.RobotRunnerPolicy import RobotRunnerPolicy
from MPC_Controller.common.Quadruped import RobotType
from RL_Environment import gamepad_reader
from RL_Environment.WeightPolicy import WeightPolicy
from isaacgym import gymapi
from RL_Environment.sim_utils import *

# 参数解析
parser = argparse.ArgumentParser(description="可视化 RL 模型步态")
parser.add_argument("--checkpoint", type=str, default="runs/Go2RLMPC/nn/latest.pth",
                    help="模型 checkpoint 路径")
parser.add_argument("--robot", type=str, default="go1", choices=["go1", "go2", "aliengo", "a1"],
                    help="机器人类型")
parser.add_argument("--use_rl", action="store_true", default=True,
                    help="是否使用 RL 模式")
parser.add_argument("--use_gamepad", action="store_true", default=True,
                    help="是否使用手柄控制")
args = parser.parse_args()

# 机器人类型映射
robot_type_map = {
    "go1": RobotType.GO1,
    "go2": RobotType.GO1,  # 使用 Go1 URDF 近似
    "aliengo": RobotType.ALIENGO,
    "a1": RobotType.A1
}
robot = robot_type_map[args.robot]

# 仿真参数
use_gamepad = args.use_gamepad
dt = Parameters.controller_dt
num_envs = 2  # 环境数量（1 个 MPC，1 个 RL）
envs_per_row = 2
env_spacing = 3

# 初始化 Isaac Gym
gym = gymapi.acquire_gym()
sim = acquire_sim(gym, dt)
add_ground(gym, sim)

# 加载机器人资产
asset = load_asset(gym, sim, robot, False)

# 设置环境
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing*2, env_spacing, env_spacing)

envs = []
actors = []
height = 0.5

# 创建环境
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, height)
    actor_handle = gym.create_actor(env, asset, pose, "MyActor", group=i, filter=1)
    envs.append(env)
    actors.append(actor_handle)

# 创建 viewer
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
cam_pos = gymapi.Vec3(2.0, 3.0, 1.5)
cam_target = gymapi.Vec3(0.5, 0, 0.2)
gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

# 初始化控制器
controllers = []
for idx in range(num_envs):
    props = gym.get_actor_dof_properties(envs[idx], actors[idx])
    props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(envs[idx], actors[idx], props)

    if idx == 0:
        # MPC 控制器
        robotRunner = RobotRunnerFSM()
        robotRunner.init(robot)
    else:
        # RL 控制器
        try:
            robotRunner = RobotRunnerPolicy(
                checkpoint=args.checkpoint,
                use_com_output=True
            )
            robotRunner.init(robot)
            print(f"[INFO] RL Policy 加载成功：{args.checkpoint}")
        except Exception as e:
            print(f"[WARN] RL Policy 加载失败：{e}")
            print("[INFO] 使用 MPC 控制器代替")
            robotRunner = RobotRunnerFSM()
            robotRunner.init(robot)
    
    controllers.append(robotRunner)

# 手柄初始化
if use_gamepad:
    gamepad = gamepad_reader.Gamepad(vel_scale_x=1.0, vel_scale_y=0.5, vel_scale_rot=2.0)

print("\n" + "="*60)
print("  RL 模型步态可视化")
print("="*60)
print("  左侧：MPC 控制器  |  右侧：RL 控制器")
print("  使用手柄左摇杆控制速度，右摇杆控制旋转")
print("  按 B 键切换步态，按 X 键切换模式")
print("  按 Esc 关闭窗口退出")
print("="*60 + "\n")

count = 0
render_fps = 60
render_count = int(1/render_fps/dt)

# 仿真主循环
while not gym.query_viewer_has_closed(viewer):
    # 物理仿真步进
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 获取控制命令
    commands = np.zeros(3, dtype=DTYPE)
    if use_gamepad:
        lin_speed, ang_speed, e_stop = gamepad.get_command()
        Parameters.cmpc_gait = gamepad.get_gait()
        Parameters.control_mode = gamepad.get_mode()
        if not e_stop:
            commands = np.array([lin_speed[0], lin_speed[1], ang_speed], dtype=DTYPE)

    # 运行控制器
    for idx, (env, actor, controller) in enumerate(zip(envs, actors, controllers)):
        dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        body_idx = gym.find_actor_rigid_body_index(env, actor, controller._quadruped._bodyName, gymapi.DOMAIN_ACTOR)
        body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)[body_idx]
        
        legTorques = controller.run(dof_states, body_states, commands)
        gym.apply_actor_dof_efforts(env, actor, legTorques / (Parameters.controller_dt*100))

    # 安全检查
    if Parameters.locomotionUnsafe:
        if use_gamepad:
            gamepad.fake_event(ev_type='Key', code='BTN_TR', value=0)
        Parameters.locomotionUnsafe = False

    # 渲染
    if count % render_count == 0:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        count = 0
    gym.sync_frame_time(sim)

    count += 1

# 清理
if use_gamepad:
    gamepad.stop()
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("\n[INFO] 可视化结束")