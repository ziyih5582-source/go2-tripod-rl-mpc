⚠️ DEPRECATED — 早期迭代版本，已废弃
=========================================

此目录包含最早期的三足行走 demo，已被根目录的正式版本取代：

  早期版本                    正式版本
  ─────────                   ────────
  demo/mpc_walk.py      →     Go2_Controller/mpc_walk1.py
  demo/deploy_walk.py   →     deploy_walk1.py (根目录)

正式版本经过真机验证，功能更完善、可配置性更强。
请不要再使用 demo/ 下的文件，仅在需要回溯早期实现时参考。

原始内容：
  这是两个完美的walk demo,使用的时候把这两个移动到父目录就行.注意路径问题
  conda activate rlmpc
  export PYTHONPATH=~/Isaacgympackage/go2_deploy/unitree_sdk2_python:$PYTHONPATH
  export PYTHONPATH=~/Isaacgympackage/go2_deploy/unitree_sdk2_python/example/go2/low_level:$PYTHONPATH
  export LD_LIBRARY_PATH=~/miniconda3/envs/rlmpc/lib:$LD_LIBRARY_PATH
