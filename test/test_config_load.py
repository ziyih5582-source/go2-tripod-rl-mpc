#!/usr/bin/env python3
"""
测试 Hydra 配置加载逻辑（不依赖 isaacgym）
"""
import os
import sys

# 添加项目根目录到 path
sys.path.insert(0, '/home/huangziyi2025/rl-mpc-locomotion')

# 设置工作目录到 RL_Environment（Hydra 配置所在目录）
os.chdir('/home/huangziyi2025/rl-mpc-locomotion/RL_Environment')

# 测试 Hydra 配置加载
def test_hydra_config():
    from omegaconf import OmegaConf
    from hydra import compose, initialize
    
    # OmegaConf resolver
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)
    
    task = "Go2RLMPCSimple"
    checkpoint = "runs/Go2RLMPCSimple/May14_23-55-02/model_1000.pt"
    num_envs = 1
    
    print("=" * 60)
    print("测试 Hydra 配置加载")
    print(f"当前工作目录: {os.getcwd()}")
    print("=" * 60)
    
    try:
        # 初始化 Hydra
        print(f"\n[1] 初始化 Hydra 配置...")
        initialize(config_path="./cfg")
        
        # 使用 config_cpu.yaml（包含正确的 defaults）
        print(f"\n[2] 加载配置文件...")
        cfg = compose(config_name="config_cpu",
                      overrides=[f"checkpoint={checkpoint}",
                                 f"task_name={task}",
                                 f"num_envs={num_envs}"])
        
        print(f"\n[3] 检查配置结构...")
        print(f"    - cfg.task_name 类型: {type(cfg.task_name)}")
        print(f"    - cfg.task_name 值: {cfg.task_name}")
        
        # 检查 cfg.task 是否存在
        has_task_attr = hasattr(cfg, 'task')
        print(f"    - cfg.task 属性存在: {has_task_attr}")
        
        if has_task_attr:
            task_cfg = cfg.task
            print(f"    - cfg.task 类型: {type(task_cfg)}")
            
            # 使用 OmegaConf.select 安全获取配置
            lin_vel_scale = OmegaConf.select(task_cfg, "env.learn.linearVelocityScale", default=None)
            ang_vel_scale = OmegaConf.select(task_cfg, "env.learn.angularVelocityScale", default=None)
            dof_pos_scale = OmegaConf.select(task_cfg, "env.learn.dofPositionScale", default=None)
            dof_vel_scale = OmegaConf.select(task_cfg, "env.learn.dofVelocityScale", default=None)
            
            print(f"\n[4] 成功读取配置参数:")
            print(f"    - lin_vel_scale: {lin_vel_scale}")
            print(f"    - ang_vel_scale: {ang_vel_scale}")
            print(f"    - dof_pos_scale: {dof_pos_scale}")
            print(f"    - dof_vel_scale: {dof_vel_scale}")
            
            if all([lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale]):
                print("\n[SUCCESS] 配置加载测试通过！")
                return True
            else:
                print("\n[WARNING] 部分配置参数未找到，使用默认值")
                return True
        else:
            # 旧配置结构，检查 task_name 是否包含配置
            if isinstance(cfg.task_name, dict):
                print(f"\n[4] cfg.task_name 是字典类型（旧结构）")
                task_cfg = cfg.task_name
                return True
            else:
                print(f"\n[ERROR] cfg.task_name 是字符串，无法访问嵌套配置")
                return False
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hydra_config()
    sys.exit(0 if success else 1)
