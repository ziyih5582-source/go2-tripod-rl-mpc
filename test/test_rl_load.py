#!/usr/bin/env python3
"""
测试 RL Policy 加载是否正常
"""
import os
import sys

# 添加项目根目录到 path
sys.path.insert(0, '/home/huangziyi2025/rl-mpc-locomotion')

# 设置工作目录
os.chdir('/home/huangziyi2025/rl-mpc-locomotion/RL_Environment')

# 测试 WeightPolicy 加载
def test_weight_policy():
    from RL_Environment.WeightPolicy import WeightPolicy
    
    checkpoint = "runs/Go2RLMPCSimple/May14_23-55-02/model_1000.pt"
    
    print("=" * 60)
    print("测试 RL Policy 加载")
    print("=" * 60)
    
    try:
        print(f"\n[1] 尝试加载模型: {checkpoint}")
        policy = WeightPolicy(
            task="go2_rl_mpc_simple",
            checkpoint=checkpoint,
            num_envs=1,
            device='cpu',
            use_com_output=True
        )
        
        print(f"\n[2] 模型加载成功！")
        print(f"    - 任务名称: {policy.task_name_str}")
        print(f"    - 观测维度: {policy.num_obs}")
        print(f"    - 动作维度: {policy.num_actions}")
        print(f"    - 设备: {policy.device}")
        print(f"    - lin_vel_scale: {policy.lin_vel_scale}")
        print(f"    - ang_vel_scale: {policy.ang_vel_scale}")
        print(f"    - dof_pos_scale: {policy.dof_pos_scale}")
        print(f"    - dof_vel_scale: {policy.dof_vel_scale}")
        
        # 测试推理
        print(f"\n[3] 测试推理...")
        action = policy.step()
        print(f"    - 动作输出: {action}")
        
        print("\n[SUCCESS] RL Policy 加载和推理测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_weight_policy()
    sys.exit(0 if success else 1)
