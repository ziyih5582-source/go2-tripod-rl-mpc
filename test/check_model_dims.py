import torch
import os

# 模型路径
model_path = "RL_Environment/runs/Go2RLMPCSimple/May14_23-55-02/model_1000.pt"

# 检查文件是否存在
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

print(f"Loading model from {model_path}...")

# 加载模型权重
checkpoint = torch.load(model_path, map_location='cpu')

print("\n=== Checkpoint Keys ===")
print(f"Keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")

if isinstance(checkpoint, dict):
    # 检查是否有actor和critic网络
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print("\n=== State Dict Keys (first 20) ===")
    keys = list(state_dict.keys())
    for key in keys[:20]:
        print(f"  {key}: {state_dict[key].shape}")
    
    if len(keys) > 20:
        print(f"  ... and {len(keys) - 20} more keys")
    
    # 寻找actor和critic网络的结构
    actor_keys = [k for k in keys if 'actor' in k.lower() or 'mlp' in k.lower()]
    print(f"\n=== Potential actor keys ===")
    for key in actor_keys[:10]:
        print(f"  {key}: {state_dict[key].shape}")

# 尝试找到输入输出维度
# 通常actor网络的最后一层输出是动作维度
# 最后一层的权重形状可以帮助推断
print("\n=== Inferring dimensions from weight shapes ===")

# 查找policy网络
policy_keys = [k for k in keys if 'policy' in k.lower()]
print(f"Policy keys: {policy_keys}")

# 查找fc层（通常是全连接层）
fc_keys = [k for k in keys if 'fc' in k.lower() or 'linear' in k.lower()]
print(f"FC/Linear keys: {fc_keys}")

# 尝试找到mlp.0.weight（第一层）和最后一层
first_layer = [k for k in keys if 'mlp.0.weight' in k or 'fc.0.weight' in k or 'layers.0.weight' in k]
last_layer = [k for k in keys if 'output.weight' in k or 'mlp_out.weight' in k]

print(f"\nFirst layer candidates: {first_layer}")
print(f"Last layer candidates: {last_layer}")

# 分析actor网络
actor_weight_keys = [k for k in keys if 'actor' in k.lower() and 'weight' in k.lower()]
critic_weight_keys = [k for k in keys if 'critic' in k.lower() and 'weight' in k.lower()]

if actor_weight_keys:
    print(f"\n=== Actor Network Weights ===")
    for key in sorted(actor_weight_keys):
        print(f"  {key}: {state_dict[key].shape}")

if critic_weight_keys:
    print(f"\n=== Critic Network Weights ===")
    for key in sorted(critic_weight_keys):
        print(f"  {key}: {state_dict[key].shape}")

# 尝试从weight形状推断维度
# 对于actor: 输入层 -> 隐藏层 -> 输出层(output)
# weight shape: (out_features, in_features)
print("\n=== Dimension Inference ===")

# 找最后一层权重（动作输出）
output_weights = [k for k in keys if ('actor' in k.lower() or 'mlp' in k.lower()) and 
                   ('weight' in k.lower()) and
                   (not any(x in k for x in ['0', '1', '2', '3', '4']))]

# 找第一层权重（观察输入）
input_weights = [k for k in keys if ('actor' in k.lower() or 'mlp' in k.lower()) and 
                  '0.weight' in k.lower() and 'weight' in k.lower()]

if output_weights:
    for key in output_weights:
        shape = state_dict[key].shape
        print(f"Output layer ({key}): shape = {shape}")
        print(f"  -> Output dim = {shape[0]}")
        print(f"  -> Input to this layer = {shape[1]}")

if input_weights:
    for key in input_weights:
        shape = state_dict[key].shape
        print(f"Input layer ({key}): shape = {shape}")
        print(f"  -> Input dim = {shape[1]}")