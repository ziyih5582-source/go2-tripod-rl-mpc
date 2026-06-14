# CPU 训练使用说明

## 概述

本项目已添加 CPU 训练支持，适用于以下场景：
- 无 GPU 环境的开发调试
- 算法验证
- 代码测试

## 使用方法

### 1. CPU 训练命令

```bash
cd RL_Environment
python train_cpu.py task=Aliengo env.numEnvs=4
```

### 2. GPU 训练命令（默认）

```bash
cd RL_Environment
python train.py task=Aliengo
```

### 2. 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `config=config_cpu` | 使用 CPU 配置文件 | 必需 |
| `task=Aliengo` | 机器人类型 | Aliengo/Go1/A1 |
| `env.numEnvs` | 并行环境数 | 4-16 (CPU) |

### 3. 性能建议

| CPU 核心数 | 推荐环境数 | 预期训练速度 |
|-----------|-----------|--------------|
| 4 核 | 4 | 非常慢 |
| 8 核 | 8 | 慢 |
| 16 核 | 16 | 中等 |

**注意**: CPU 训练速度远低于 GPU，建议仅用于开发调试。

## 修改的文件

1. **`RL_Environment/cfg/config_cpu.yaml`** - 新建 CPU 配置文件
   - `pipeline: 'cpu'` - CPU 流水线
   - `sim_device: 'cpu'` - CPU 物理仿真
   - `rl_device: 'cpu'` - CPU 运行 RL
   - `num_threads: 8` - CPU 线程数

2. **`RL_Environment/WeightPolicy.py`** - 添加 device 参数支持
   ```python
   def __init__(self, ..., device=None):
       # 自动检测或指定设备
       if device is None:
           self.device = "cuda" if torch.cuda.is_available() else "cpu"
       else:
           self.device = device
   ```

3. **`RL_Environment/tasks/aliengo.py`** - CPU/GPU 自适应数据传输
   ```python
   if self.device.type == 'cpu':
       # CPU 模式：直接使用 numpy()
       actions_np = actions_rescale.detach().numpy()
   else:
       # GPU 模式：先转 CPU
       actions_np = actions_rescale.detach().cpu().numpy()
   ```

## 故障排除

### 问题 1: Isaac Gym 初始化失败
```
Error: Failed to create sim
```
**解决方案**: 确保 Isaac Gym 已正确安装并支持 CPU 模式

### 问题 2: 训练速度极慢
**解决方案**: 减少 `env.numEnvs` 或增加 `num_threads`

### 问题 3: 内存不足
**解决方案**: 减少并行环境数 `env.numEnvs=2`

## GPU 训练（默认）

如需使用 GPU 训练，使用原配置：
```bash
cd RL_Environment
python train.py task=Aliengo