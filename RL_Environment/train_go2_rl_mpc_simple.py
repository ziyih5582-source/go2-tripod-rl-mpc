import faulthandler
faulthandler.enable()
"""
train_go2_rl_mpc_simple.py — 简化版 RL+MPC 训练脚本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于 mpc_walk1.py 的简化 MPC 控制器进行 RL 训练。

架构流程：
  观测 → RL 输出 COM [dx, dy, dz] → MPC 计算关节目标 → PD 控制 → 仿真步进 → 奖励

使用方法：
  python train_go2_rl_mpc_simple.py
  或带参数：
  python train_go2_rl_mpc_simple.py task_name=Go2RLMPCSimple num_envs=256 max_iterations=2000
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制使用 CPU

import isaacgym
from datetime import datetime
import sys

# 添加父目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsl_rl.runners import OnPolicyRunner
from MPC_Controller.Parameters import Parameters

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from RL_Environment.utils.reformat import omegaconf_to_dict, print_dict
from RL_Environment.utils.utils import set_np_formatting, set_seed
from RL_Environment.utils.rsl_rl_utils import update_cfg_from_args, class_to_dict, get_load_path

from RL_Environment.tasks import isaacgym_task_map

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

## OmegaConf & Hydra Config
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


@hydra.main(config_name="config_cpu", config_path="./cfg")
def launch_hydra(cfg: DictConfig):
    """训练入口函数"""
    
    # 确保 checkpoint 路径可以指定为相对路径
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # 设置 numpy 格式化
    set_np_formatting()

    # 设置随机种子
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    print(f"[INFO] 创建训练环境: {cfg.task_name}")
    
    # 创建环境
    env = isaacgym_task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless
    )

    # 创建 PPO 训练器
    train_cfg = isaacgym_task_map["ConfigPPO"]
    train_cfg = update_cfg_from_args(train_cfg, cfg)

    # 创建日志目录
    log_root = os.path.join(ROOT_DIR, 'runs', cfg.task_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S'))

    train_cfg_dict = class_to_dict(train_cfg)
    ppo_runner = OnPolicyRunner(env, train_cfg_dict, log_dir, cfg.rl_device)

    # 加载预训练模型（如果指定）
    if cfg.test or cfg.checkpoint:
        try:
            print(f"加载模型: {cfg.checkpoint}")
            ppo_runner.load(cfg.checkpoint)
        except:
            print("加载失败，尝试加载最新模型...")
            resume_path = get_load_path(log_root)
            print(f"从最新运行加载: {resume_path}")
            ppo_runner.load(resume_path)

    # 保存配置
    experiment_dir = log_dir
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # 开始训练
    if not cfg.test:
        print(f"\n{'='*60}")
        print(f"  开始训练 RL+MPC 控制器")
        print(f"  任务: {cfg.task_name}")
        print(f"  环境数: {cfg.task.env.numEnvs}")
        print(f"  最大迭代: {train_cfg.runner.max_iterations}")
        print(f"{'='*60}\n")
        
        ppo_runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations, 
            init_at_random_ep_len=False
        )
    else:
        # 测试模式
        print("\n进入测试模式...")
        policy = ppo_runner.get_inference_policy(device=env.device)
        obs = env.get_observations()

        for i in range(10 * int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    # 关键设置：启用 RL 模式
    Parameters.bridge_MPC_to_RL = True
    
    launch_hydra()