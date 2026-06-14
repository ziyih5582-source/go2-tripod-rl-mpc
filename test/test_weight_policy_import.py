"""
测试 WeightPolicy 能否正确加载 RL 模型
绕过 IsaacGym 依赖，直接测试 hydra 配置解析
"""

import os
import sys

# 设置环境变量避免导入 isaacgym
os.environ["SKIP_ISAACGYM"] = "1"

# 添加路径
ROOT_DIR = "/home/huangziyi2025/rl-mpc-locomotion"
sys.path.insert(0, ROOT_DIR)

# 临时模拟 isaacgym 模块，避免导入错误
class MockIsaacGym:
    pass
sys.modules['isaacgym'] = MockIsaacGym()

# 测试 Hydra 配置加载
from omegaconf import OmegaConf

print("="*60)
print("测试 Hydra 配置加载")
print("="*60)

# 注册 resolver
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

# 切换到配置目录
os.chdir(os.path.join(ROOT_DIR, "RL_Environment"))

# 测试 compose
from hydra import compose, initialize

print("\n[测试] 初始化 Hydra 并加载配置...")

try:
    initialize(config_path="./cfg")
    cfg = compose(config_name="config", 
                  overrides=["checkpoint=RL_Environment/runs/Go2RLMPCSimple/May14_23-55-02/model_1000.pt",
                             "task=go2_rl_mpc_simple",
                             "num_envs=1"])
    
    print(f"[成功] task_name: {cfg['task_name']}")
    print(f"[成功] checkpoint: {cfg.checkpoint}")
    print(f"[成功] linearVelocityScale: {cfg['task']['env']['learn']['linearVelocityScale']}")
    
except Exception as e:
    print(f"[失败] {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成！")