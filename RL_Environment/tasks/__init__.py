from .aliengo import Aliengo
from .go1 import Go1
from .a1 import A1Task
from .legged_config_ppo import LeggedCfgPPO
from .go2_rl_mpc import Go2RLMPC
from .go2_rl_mpc_simple import Go2RLMPCSimple

# Mappings from strings to environments
isaacgym_task_map = {
    "Aliengo": Aliengo,
    "Go1": Go1,
    "A1": A1Task,
    "ConfigPPO": LeggedCfgPPO,
    "Go2RLMPC": Go2RLMPC,  # RL+MPC 混合控制（复杂版）
    "Go2RLMPCSimple": Go2RLMPCSimple,  # 简化版 RL+MPC（基于 mpc_walk1）
}
