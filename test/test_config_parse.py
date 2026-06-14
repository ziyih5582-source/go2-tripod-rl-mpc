#!/usr/bin/env python3
"""
验证配置文件结构（不依赖 Hydra 运行时）
"""
import os
import sys

print("=" * 60)
print("测试配置文件解析")
print("=" * 60)

# 直接读取 YAML 文件
import yaml

config_path = "/home/huangziyi2025/rl-mpc-locomotion/RL_Environment/cfg/config_cpu.yaml"

print(f"\n[1] 读取配置文件: {config_path}")
try:
    with open(config_path, 'r') as f:
        config_cpu = yaml.safe_load(f)
    print("    - config_cpu.yaml 读取成功")
except Exception as e:
    print(f"    - ERROR: {e}")
    sys.exit(1)

print(f"\n[2] 检查 config_cpu.yaml 结构:")
print(f"    - task_name: {config_cpu.get('task_name', 'NOT FOUND')}")
print(f"    - defaults: {config_cpu.get('defaults', 'NOT FOUND')}")

# 检查 defaults 是否包含 task
defaults = config_cpu.get('defaults', [])
has_task_in_defaults = any('task' in str(d) for d in defaults)
print(f"    - defaults 中包含 task: {has_task_in_defaults}")

# 读取 task 配置文件
task_file = "/home/huangziyi2025/rl-mpc-locomotion/RL_Environment/cfg/task/go2_rl_mpc_simple.yaml"
print(f"\n[3] 读取 task 配置: {task_file}")
try:
    with open(task_file, 'r') as f:
        task_config = yaml.safe_load(f)
    print("    - task 配置读取成功")
    
    # 检查 learn 参数
    print(f"\n[4] 检查 learn 参数:")
    env = task_config.get('env', {})
    learn = env.get('learn', {})
    
    lin_vel_scale = learn.get('linearVelocityScale', None)
    ang_vel_scale = learn.get('angularVelocityScale', None)
    dof_pos_scale = learn.get('dofPositionScale', None)
    dof_vel_scale = learn.get('dofVelocityScale', None)
    
    print(f"    - linearVelocityScale: {lin_vel_scale}")
    print(f"    - angularVelocityScale: {ang_vel_scale}")
    print(f"    - dofPositionScale: {dof_pos_scale}")
    print(f"    - dofVelocityScale: {dof_vel_scale}")
    
    if all([lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale]):
        print("\n[SUCCESS] 配置文件解析测试通过！")
        print("\n修复说明:")
        print("  WeightPolicy.py 已修改为:")
        print("  1. 使用 config_cpu.yaml 而不是 config.yaml")
        print("  2. 从 cfg.task 获取 task 配置")
        print("  3. 使用 OmegaConf.select 安全获取嵌套值")
    else:
        print("\n[ERROR] 部分配置参数缺失")
        sys.exit(1)
        
except Exception as e:
    print(f"    - ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("注意: 需要在 RL_Environment 目录下运行才能正确加载 Hydra 配置")
print("=" * 60)
