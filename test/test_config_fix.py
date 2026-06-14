"""
验证 config.yaml 修复后 WeightPolicy 的配置加载
"""

import os
import sys

# 设置环境变量避免导入 isaacgym
os.environ["SKIP_ISAACGYM"] = "1"

ROOT_DIR = "/home/huangziyi2025/rl-mpc-locomotion"
sys.path.insert(0, ROOT_DIR)

# 临时模拟 isaacgym 模块
class MockIsaacGym:
    pass
sys.modules['isaacgym'] = MockIsaacGym()

from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize

print("="*60)
print("验证 config.yaml 修复")
print("="*60)

# 注册 resolver
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

# 切换到配置目录
os.chdir(os.path.join(ROOT_DIR, "RL_Environment"))

print("\n[Step 1] 加载 Hydra 配置...")
initialize(config_path="./cfg")

task = "go2_rl_mpc_simple"
checkpoint = "RL_Environment/runs/Go2RLMPCSimple/May14_23-55-02/model_1000.pt"
num_envs = 1

cfg = compose(config_name="config", 
              overrides=[f"checkpoint={checkpoint}", 
                         f"task={task}",
                         f"num_envs={str(num_envs)}"])

print(f"cfg.task_name = {cfg.task_name}")
print(f"type(cfg.task_name) = {type(cfg.task_name)}")

# 测试 isinstance 检查 (DictConfig vs dict)
print(f"\nisinstance(cfg.task_name, dict) = {isinstance(cfg.task_name, dict)}")
print(f"isinstance(cfg.task_name, DictConfig) = {isinstance(cfg.task_name, DictConfig)}")

# 正确的检查方式
if isinstance(cfg.task_name, DictConfig):
    task_cfg = cfg.task_name
    task_name_str = task_cfg.get('task_name', task)
elif isinstance(cfg.task_name, dict):
    task_cfg = cfg.task_name
    task_name_str = task_cfg.get('task_name', task)
else:
    task_cfg = cfg.task_name
    task_name_str = str(task_cfg)

print(f"\ntask_cfg type = {type(task_cfg)}")
print(f"task_name_str = {task_name_str}")

# 验证访问配置
print(f"\n验证配置访问:")
print(f"  lin_vel_scale = {task_cfg['env']['learn']['linearVelocityScale']}")
print(f"  ang_vel_scale = {task_cfg['env']['learn']['angularVelocityScale']}")

# 验证日志路径
log_root = os.path.join(ROOT_DIR, 'runs', task_name_str)
print(f"\nlog_root = {log_root}")

print("\n✓ 所有检查通过！修复有效。")