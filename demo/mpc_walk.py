#一个完美的行走demo,和deploy_stand.py配合食用
"""
mpc_v5.py — Go2 三足运动学控制器 + IMU 闭环平衡控制器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v5 新增：BalanceController
  - 读取 IMU roll / pitch / 角速度
  - PD 控制器实时调整 com_offset，让身体主动"仰身"抵消倾斜
  - 内置低通滤波（避免颤抖）和安全阈值急停触发

v5.1 新增：fr_lift_alpha 参数
  - compute_control() / _plan_feet() 增加 fr_lift_alpha ∈ [0, 1]
  - fr_lift_alpha = 0 → FR 腿落地（四足站立模式）
  - fr_lift_alpha = 1 → FR 腿完全收起（三足站立模式，原始行为）
  - 过渡过程中 FR 足端平滑插值，com_offset 对 FR 的作用线性淡出
  - 配合 deploy_v5.py 的 PRE_SHIFT / LIFT_FR 状态实现无突变过渡

坐标系约定（Unitree Go2 Body Frame，右手系）
  X: 前  Y: 左  Z: 上
  roll > 0  → 右侧向下（机身向右倾斜）
  pitch > 0 → 机头向上

com_offset 含义
  foot_target_body = nominal_foot - com_offset
  com_offset.y > 0 → 足端相对向右移 → 等效质心向左移
  com_offset.x > 0 → 足端相对向后移 → 等效质心向前移

关节顺序（算法侧，与 Isaac Lab 一致）
  [FL_hip, FL_thigh, FL_calf,
   FR_hip, FR_thigh, FR_calf,
   RL_hip, RL_thigh, RL_calf,
   RR_hip, RR_thigh, RR_calf]
"""

import torch
from typing import Tuple


# ═══════════════════════════════════════════════════════════════
#  解析逆运动学（与 v4/v5 完全一致）
# ═══════════════════════════════════════════════════════════════
class Go2IK:
    _HIP_POS = [
        [ 0.1934,  0.0465, 0.0],   # 0: FL
        [ 0.1934, -0.0465, 0.0],   # 1: FR
        [-0.1934,  0.0465, 0.0],   # 2: RL
        [-0.1934, -0.0465, 0.0],   # 3: RR
    ]

    def __init__(self, device='cpu'):
        self.device  = device
        self.l_hip   = torch.tensor(0.0955, device=device, dtype=torch.float32)
        self.l_thigh = torch.tensor(0.213,  device=device, dtype=torch.float32)
        self.l_calf  = torch.tensor(0.213,  device=device, dtype=torch.float32)
        self.hip_pos = torch.tensor(
            self._HIP_POS, dtype=torch.float32, device=device)

    def solve(self, leg: int, foot_body: torch.Tensor) -> torch.Tensor:
        h  = self.hip_pos[leg]
        px = foot_body[:, 0] - h[0]
        py = foot_body[:, 1] - h[1]
        pz = foot_body[:, 2] - h[2]
        sign = 1.0 if leg in (0, 2) else -1.0
        r_yz  = (py.pow(2) + pz.pow(2) + 1e-8).sqrt()
        r_sag = (r_yz.pow(2) - self.l_hip ** 2).clamp(min=1e-6).sqrt()
        θ_hip = torch.atan2(pz, py) + torch.atan2(r_sag, self.l_hip * sign)
        r_leg   = (px.pow(2) + r_sag.pow(2) + 1e-8).sqrt()
        cos_c   = ((r_leg.pow(2) - self.l_thigh ** 2 - self.l_calf ** 2)
                   / (2 * self.l_thigh * self.l_calf)).clamp(-0.99, 0.99)
        θ_calf  = -torch.acos(cos_c)
        sin_φ   = (self.l_calf * (-θ_calf).sin() / r_leg).clamp(-0.99, 0.99)
        θ_thigh = torch.atan2(px, r_sag) + torch.asin(sin_φ)
        return torch.stack([θ_hip, θ_thigh, θ_calf], dim=-1)


# ═══════════════════════════════════════════════════════════════
#  三足步态控制器（v5.1：新增 fr_lift_alpha 参数）
# ═══════════════════════════════════════════════════════════════
class MPCController:
    GAIT_PERIOD = 0.6
    SWING_RATIO = 0.15
    STEP_HEIGHT = 0.07
    BODY_HEIGHT = -0.30

    FR_RETRACT_DX = -0.04
    FR_RETRACT_Z  = -0.11

    _PHASE_OFF = {0: 0.000, 2: 0.5, 3: 0.75}

    # RL COM 输出范围
    RL_COM_X_RANGE = [0.023, 0.27]
    RL_COM_Y_RANGE = [0.063, 0.067]
    RL_COM_Z_RANGE = [-0.01, 0.01]

    def __init__(self,
                 num_envs : int,
                 dt       : float,
                 jpos_lim : Tuple[float, float] = (-3.14, 3.14),
                 jvel_lim : float = 20.0,
                 device   : str = 'cpu',
                 use_rl_mode: bool = False):  # 新增：是否使用 RL 模式
        """
        Args:
            use_rl_mode: 是否启用 RL 模式
                False: 原始 MPC 模式，com_offset 由 BalanceController 计算
                True: RL 模式，com_offset 由 RL 网络输出
        """
        self.use_rl_mode = use_rl_mode
        self.num_envs = num_envs
        self.dt       = dt
        self.jpos_lim = jpos_lim
        self.jvel_lim = jvel_lim
        self.device   = device
        self.ik    = Go2IK(device=device)
        self.phase = torch.zeros(num_envs, device=device)
        hip  = self.ik.hip_pos
        side = torch.tensor([1, -1, 1, -1], dtype=torch.float32, device=device)
        self._nom_y = hip[:, 1] + side * self.ik.l_hip
        self._nom_x = hip[:, 0]

    def compute_control(self,
                        dof_pos      : torch.Tensor,
                        com_offset   : torch.Tensor,
                        vx_cmd       : float = 0.0,
                        stepping     : bool  = True,
                        fr_lift_alpha: float = 1.0,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        fr_lift_alpha : float, [0, 1]
            0.0 → FR 腿落地（四足站立），1.0 → FR 腿完全收起（三足站立）
            过渡阶段用 0→1 线性/smooth-step 插值，实现平滑抬腿。
            com_offset 对 FR 的作用随 fr_lift_alpha 增大而线性淡出：
              FR 接地时整体随身体偏移，FR 离地后独立定位。
        """
        if stepping:
            self.phase = (self.phase + self.dt / self.GAIT_PERIOD) % 1.0
        foot_targets = self._plan_feet(com_offset, vx_cmd, fr_lift_alpha)
        N    = self.num_envs
        jpos = torch.zeros(N, 12, device=self.device)
        for leg in range(4):
            s = leg * 3
            jpos[:, s:s+3] = self.ik.solve(leg, foot_targets[:, leg])
        jpos = jpos.clamp(*self.jpos_lim)
        jvel = ((jpos - dof_pos) / self.dt).clamp(-self.jvel_lim, self.jvel_lim)
        return jpos, jvel

    def _plan_feet(self,
                   com_offset   : torch.Tensor,
                   vx_cmd       : float,
                   fr_lift_alpha: float = 1.0,
                   ) -> torch.Tensor:
        """
        FR 腿足端插值逻辑
        ─────────────────────────────────────────────────────────────
        fr_lift_alpha = 0（接地）:
            足端 = 名义位置（与支撑腿同等对待），com_offset 完整作用。
        fr_lift_alpha = 1（收起）:
            足端 = FR_RETRACT 位置，com_offset 不作用（与 v5 原始行为一致）。
        0 < fr_lift_alpha < 1:
            足端 XYZ 线性插值；com_offset 对 FR 的作用以 (1-alpha) 权重淡出。
            物理意义：FR 抬离地面的同时，该腿从"参与重心管理"逐渐切换到"自由收起"。
        """
        N   = self.num_envs
        out = torch.zeros(N, 4, 3, device=self.device)
        t_st   = self.GAIT_PERIOD * (1.0 - self.SWING_RATIO)
        dx_rbt = min(vx_cmd * t_st * 0.2 , 0.015)

        for leg in range(4):
            nx = float(self._nom_x[leg])
            ny = float(self._nom_y[leg])

            if leg == 1:  # ── FR 腿：平滑插值 ──────────────────────
                # 接地位（fr_lift_alpha=0）
                fr_x_gnd = nx                       # 名义 X
                fr_y_gnd = ny                       # 名义 Y（含 com_offset 淡出）
                fr_z_gnd = self.BODY_HEIGHT         # 落地高度

                # 收起位（fr_lift_alpha=1）
                fr_x_up  = nx + self.FR_RETRACT_DX  # 向后收 4 cm
                fr_y_up  = ny                        # Y 方向不收
                fr_z_up  = self.FR_RETRACT_Z         # 抬起高度

                # 线性插值足端位置
                a = fr_lift_alpha
                fr_x = fr_x_gnd * (1.0 - a) + fr_x_up * a
                fr_y = fr_y_gnd                      # Y 同
                fr_z = fr_z_gnd * (1.0 - a) + fr_z_up * a

                # com_offset 随 alpha 淡出：接地时 FR 跟随全身偏移，收起后独立
                com_w = 1.0 - a   # 接地时 =1，收起时 =0
                out[:, 1, 0] = fr_x - com_offset[:, 0] * com_w
                out[:, 1, 1] = fr_y - com_offset[:, 1] * com_w
                out[:, 1, 2] = fr_z - com_offset[:, 2] * com_w
                continue

            # ── 三条支撑腿（FL / RL / RR）：与 v5 原始逻辑完全一致 ──
            lp = (self.phase + self._PHASE_OFF[leg]) % 1.0
            in_swing = lp < self.SWING_RATIO
            s = (lp / self.SWING_RATIO).clamp(0.0, 1.0)
            t = ((lp - self.SWING_RATIO) / (1.0 - self.SWING_RATIO)).clamp(0.0, 1.0)
            swing_x  = (nx - dx_rbt) + s * (2.0 * dx_rbt)
            stance_x = (nx + dx_rbt) - t * (2.0 * dx_rbt)
            fx = torch.where(in_swing, swing_x, stance_x)
            swing_z  = self.BODY_HEIGHT + 4.0 * self.STEP_HEIGHT * s * (1.0 - s)
            stance_z = torch.full((N,), self.BODY_HEIGHT, device=self.device)
            fz = torch.where(in_swing, swing_z, stance_z)
            fy = torch.full((N,), ny, device=self.device)
            out[:, leg, 0] = fx - com_offset[:, 0]
            out[:, leg, 1] = fy - com_offset[:, 1]
            out[:, leg, 2] = fz - com_offset[:, 2]

        return out

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.phase[env_ids] = 0.0

    def apply_rl_com_offset(self, rl_com_offset: torch.Tensor):
        """
        应用 RL 输出的 COM 偏移
        
        Parameters:
            rl_com_offset: [1, 3] RL 输出的 COM 偏移 [dx, dy, dz]
        
        Returns:
            com_offset: 处理后的 COM 偏移，可直接用于 compute_control
        """
        # 限幅保护
        com_x = rl_com_offset[:, 0].clamp(self.RL_COM_X_RANGE[0], self.RL_COM_X_RANGE[1])
        com_y = rl_com_offset[:, 1].clamp(self.RL_COM_Y_RANGE[0], self.RL_COM_Y_RANGE[1])
        com_z = rl_com_offset[:, 2].clamp(self.RL_COM_Z_RANGE[0], self.RL_COM_Z_RANGE[1])
        
        com_offset = torch.stack([com_x, com_y, com_z], dim=1)
        return com_offset


# ═══════════════════════════════════════════════════════════════
#  v5 不变：IMU 闭环平衡控制器
# ═══════════════════════════════════════════════════════════════
class BalanceController:
    """
    IMU 闭环平衡控制器（专为三足静态站立设计）

    ┌──────────────────────────────────────────────────────────┐
    │  开环问题：com_offset 固定 = [0, 0.065, 0]              │
    │            → 受到扰动或地面不平时无法自动恢复             │
    │                                                          │
    │  闭环方案：PD(roll, gyro_x) → Δcom_y                    │
    │            PD(pitch, gyro_y) → Δcom_x                   │
    │            tilt_magnitude   → Δcom_z (降低重心)          │
    │            低通滤波后输出 com_offset                     │
    └──────────────────────────────────────────────────────────┘

    物理直觉
    ─────────────────────────────────────────────────────────────
    roll > 0（右倾）→ com_y 增大 → 足端相对右移 → 质心相对左移
                                   → 抵消右倾 ✓
    pitch > 0（仰头）→ com_x 增大 → 足端相对后移 → 质心相对前移
                                    → 抵消仰头 ✓
    |tilt| 大 → com_z 减小 → 足端相对上移 → 身体降低
                               → 降低重心，提升稳定裕度 ✓

    参数调校指南
    ─────────────────────────────────────────────────────────────
    KP_ROLL/PITCH 过小 → 响应慢，难以抵抗扰动
    KP_ROLL/PITCH 过大 → 过冲振荡
    KD_ROLL/PITCH      → 阻尼，减小振荡
    ALPHA 接近 1       → 响应快但可能颤抖jie guo
    ALPHA 接近 0       → 过于平滑，响应迟钝
    推荐先在悬吊状态下测试，确认方向正确后再放到地面。
    """

    # ── PD 增益 ──────────────────────────────────────────────
    KP_ROLL  = 0.15    # [m/rad]    侧向 com 偏移量 / 横滚角
    KD_ROLL  = 0.08   # [m·s/rad]  微分阻尼
    KP_PITCH = 0.10    # [m/rad]    纵向 com 偏移量 / 俯仰角
    KD_PITCH = 0.06

    # ── 三足站立基准 com_offset ───────────────────────────────
    # com_y = 0.085 把质心推入 FL-RL-RR 三角形内部
    NOM_X =  0.025
    NOM_Y =  0.065
    NOM_Z =  0.00

    # ── 修正量安全限幅 ────────────────────────────────────────
    MAX_DX = 0.04    # m，纵向
    MAX_DY = 0.055   # m，侧向（不能超过三角形半径）
    MAX_DZ = 0.04    # m，仅允许向下（降重心）

    # ── 低通滤波系数（50 Hz 控制频率）────────────────────────
    # 等效截止频率 ≈ ALPHA * 50 / (2π) ≈ 1.4 Hz
    ALPHA = 0.1

    # ── 急停触发阈值 ─────────────────────────────────────────
    SAFE_ROLL  = 0.30   # rad ≈ 17°
    SAFE_PITCH = 0.25   # rad ≈ 14°
    SAFE_ROLL_VEL  = 1.5   # rad/s，角速度过快也急停
    SAFE_PITCH_VEL = 1.5

    def __init__(self, device: str = 'cpu'):
        self.device = device
        # 从名义值开始，防止刚切入时跳变
        self.com = torch.tensor(
            [[self.NOM_X, self.NOM_Y, self.NOM_Z]],
            dtype=torch.float32, device=device)
        self.unsafe = False
        # 调试计数器
        self._dbg_counter = 0

        self.i_roll  = torch.zeros(1, device=device)
        self.i_pitch = torch.zeros(1, device=device)
        self.KI = 0.02          # 很小的积分系数
        self.MAX_I = 0.015       # 积分饱和上限，防止震荡

    # ──────────────────────────────────────────────────────────
    def update(self,
               roll   : float, pitch  : float,
               gyro_x : float, gyro_y : float,
               ) -> Tuple[torch.Tensor, bool]:
        """
        每控制周期（dt=0.02s）调用一次。

        Parameters
        ----------
        roll   : imu_state.rpy[0]        横滚角 (rad)
        pitch  : imu_state.rpy[1]        俯仰角 (rad)
        gyro_x : imu_state.gyroscope[0]  绕 X 轴角速度 (rad/s)
        gyro_y : imu_state.gyroscope[1]  绕 Y 轴角速度 (rad/s)

        Returns
        -------
        com_offset : Tensor [1, 3]  下一帧的质心偏移目标
        unsafe     : bool           True → 倾斜过大，应立即急停
        """

        # ① 安全检查
        self.unsafe = (
            abs(roll)   > self.SAFE_ROLL  or
            abs(pitch)  > self.SAFE_PITCH or
            abs(gyro_x) > self.SAFE_ROLL_VEL or
            abs(gyro_y) > self.SAFE_PITCH_VEL
        )

        # ② PD 修正计算
        #   侧向（Y）：roll > 0（右倾）→ Δy > 0（质心左移）
        dy_raw = self.KP_ROLL  * roll  + self.KD_ROLL  * gyro_x
        #   纵向（X）：pitch > 0（仰头）→ Δx > 0（质心前移）
        dx_raw = self.KP_PITCH * pitch + self.KD_PITCH * gyro_y
        #   高度（Z）：倾斜越大重心越低（仅允许负修正）
        tilt_mag = (roll ** 2 + pitch ** 2) ** 0.5
        dz_raw   = -0.04 * tilt_mag

        # ③ 限幅（保证不超出支撑三角形）
        dx = max(-self.MAX_DX, min(self.MAX_DX, dx_raw))
        dy = max(-self.MAX_DY, min(self.MAX_DY, dy_raw))
        dz = max(-self.MAX_DZ, min(0.0, dz_raw))   # 只降不升

        #积分项
        dt = 0.02
        self.i_roll  += torch.tensor([roll  * dt], device=self.device)
        self.i_pitch += torch.tensor([pitch * dt], device=self.device)
        self.i_roll  = self.i_roll.clamp(-self.MAX_I, self.MAX_I)
        self.i_pitch = self.i_pitch.clamp(-self.MAX_I, self.MAX_I)

        # 在 PD 原始修正量上叠加积分
        dy_raw += self.KI * self.i_roll.item()
        dx_raw += self.KI * self.i_pitch.item()


        # ④ 目标 com_offset
        target = torch.tensor(
            [[self.NOM_X + dx,
              self.NOM_Y + dy,
              self.NOM_Z + dz]],
            dtype=torch.float32, device=self.device)

        # ⑤ 低通滤波（平滑输出，避免关节颤抖）
        self.com = self.com + self.ALPHA * (target - self.com)

        # ⑥ 调试打印（每 ~1s 一次）
        self._dbg_counter += 1
        if self._dbg_counter >= 50:
            self._dbg_counter = 0
            cy = float(self.com[0, 1])
            cx = float(self.com[0, 0])
            cz = float(self.com[0, 2])
            print(f"[BAL] roll={roll:+.3f} pitch={pitch:+.3f} "
                  f"gyro_x={gyro_x:+.2f} gyro_y={gyro_y:+.2f} | "
                  f"com=[{cx:+.3f},{cy:+.3f},{cz:+.3f}]")

        return self.com.clone(), self.unsafe

    # ──────────────────────────────────────────────────────────
    def reset(self):
        """切换状态时重置，从名义值重新开始收敛"""
        self.com = torch.tensor(
            [[self.NOM_X, self.NOM_Y, self.NOM_Z]],
            dtype=torch.float32, device=self.device)
        self.unsafe = False
        self._dbg_counter = 0
        
        self.i_roll.zero_()
        self.i_pitch.zero_()


    # ──────────────────────────────────────────────────────────
    @property
    def com_numpy(self):
        return self.com[0].numpy()


# ═══════════════════════════════════════════════════════════════
#  快速数值验证
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("  IK 验证（中性站立）")
    print("=" * 60)
    ik = Go2IK()
    for leg, name in enumerate(['FL', 'FR', 'RL', 'RR']):
        ny   = float(ik.hip_pos[leg, 1]) + (1 if leg in (0,2) else -1) * ik.l_hip
        foot = torch.tensor([[float(ik.hip_pos[leg, 0]), ny, -0.27]])
        ang  = ik.solve(leg, foot)[0]
        print(f"  {name}: hip={ang[0]:.3f}  thigh={ang[1]:.3f}  calf={ang[2]:.3f}")

    print()
    print("=" * 60)
    print("  fr_lift_alpha 验证：FR 足端插值")
    print("=" * 60)
    ctrl = MPCController(num_envs=1, dt=0.02)
    dof = torch.zeros(1, 12)
    com = torch.tensor([[0.0, 0.065, 0.0]])
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        jpos, _ = ctrl.compute_control(dof, com, vx_cmd=0.0,
                                        stepping=False, fr_lift_alpha=alpha)
        fr_hip   = jpos[0, 3].item()
        fr_thigh = jpos[0, 4].item()
        fr_calf  = jpos[0, 5].item()
        print(f"  alpha={alpha:.2f}: FR hip={fr_hip:+.3f} "
              f"thigh={fr_thigh:+.3f} calf={fr_calf:+.3f}")
    print("  ✓ alpha=0 时 FR 接地；alpha=1 时 FR 完全收起")

    print()
    print("=" * 60)
    print("  BalanceController 验证")
    print("=" * 60)
    bc = BalanceController()

    # 模拟：机器狗向右倾斜 0.1 rad
    print("  模拟右倾 0.1 rad，期望 com_y 增大（大于 0.065）:")
    for _ in range(30):
        com_out, unsafe = bc.update(roll=0.10, pitch=0.0, gyro_x=0.0, gyro_y=0.0)
    print(f"  com_y = {com_out[0,1]:.4f}（期望 > 0.065）  unsafe={unsafe}")
    assert com_out[0, 1] > 0.065, "ERROR: com_y 应增大！"

    # 模拟：回到水平
    bc.reset()
    print("  模拟机器狗水平（无扰动），期望 com_y 稳定在 0.065:")
    for _ in range(100):
        com_out, unsafe = bc.update(roll=0.0, pitch=0.0, gyro_x=0.0, gyro_y=0.0)
    print(f"  com_y = {com_out[0,1]:.4f}（期望 ≈ 0.065）  unsafe={unsafe}")
    assert abs(com_out[0,1] - 0.065) < 0.001, "ERROR: 应收敛到名义值！"

    # 模拟：倾斜过大触发急停
    bc.reset()
    com_out, unsafe = bc.update(roll=0.35, pitch=0.0, gyro_x=0.0, gyro_y=0.0)
    print(f"  大角度 roll=0.35 → unsafe={unsafe}（期望 True）")
    assert unsafe, "ERROR: 应触发安全标志！"

    print()
    print("  ✓ 所有验证通过")
