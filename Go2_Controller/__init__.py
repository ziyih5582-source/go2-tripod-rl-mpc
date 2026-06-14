# Go2_Controller — Go2 三足行走 RL+MPC 协同控制包
#
# 对外暴露核心类，方便外部 import：
#   from Go2_Controller import MPCController, BalanceController, Go2IK
#   等价于：
#   from Go2_Controller.mpc_walk1 import MPCController, BalanceController, Go2IK

from .mpc_walk1 import MPCController, BalanceController, Go2IK

__all__ = ["MPCController", "BalanceController", "Go2IK"]
