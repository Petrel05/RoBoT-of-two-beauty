# 轮腿机器人姿态与运动控制对比项目

本项目实现了一个轻量级二维轮腿机器人仿真平台，用于比较四类控制器：

- `lqr`：基于局部线性化模型的连续时间 LQR；
- `wbc_qp`：显式建模降阶动力学、接触力、摩擦锥、执行器限幅和预测安全约束的 WBC/QP；
- `rl_ppo`：在稳定基线控制器上叠加动作修正量的残差 PPO；
- `rl_ppo_direct`：直接输出三路关节力矩的 PPO。

仓库中同时包含固定测试场景、随机训练场景、PPO 训练脚本、批量评估脚本、曲线绘制工具、最终实验输出和完整 LaTeX 报告。

## 1. 当前最终结果

当前用于报告的统一评估结果保存在：

```text
outputs/compare_four_full_wbc/
```

该目录由同一次四控制器批量评估生成，包含：

```text
outputs/compare_four_full_wbc/
├── metrics.csv              # 4 个控制器 x 8 个场景，共 32 行指标
├── figures/                 # 32 张单控制器曲线图 + 8 张四控制器对比图 + 8 张 WBC/QP 诊断图
└── logs/                    # 32 个原始 .npz 仿真日志
```

八个固定测试场景下的平均结果如下。正式评估默认不把真实外力直接提供给任何控制器，控制器只能根据状态误差响应扰动：

| 控制器 | 成功场景 | 平均高度 RMSE / m | 平均姿态 RMSE / deg | 平均速度 RMSE / (m/s) | 平均饱和比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lqr` | `8/8` | `0.027323` | `0.390307` | `0.552078` | `0.000988` |
| `wbc_qp` | `8/8` | `0.000380` | `0.009513` | `0.257876` | `0.019039` |
| `rl_ppo` | `8/8` | `0.009810` | `2.497230` | `0.370462` | `0.015658` |
| `rl_ppo_direct` | `8/8` | `0.007023` | `1.588770` | `0.081625` | `0.016958` |

结果表明：降阶 WBC/QP 在高度和姿态控制方面显著优于基础 LQR；直接力矩 PPO 的平均速度跟踪误差最低，但在 `5 m/s` 复合边界场景中的姿态误差明显大于 WBC/QP。四个控制器在题目边界场景 `G_requirement_boundary` 和越界压力场景 `E_combined_stress` 中都完成了完整的 `8 s` 仿真。

详细分析见：

```text
report.tex
report.pdf
```

## 2. 快速复现

### 2.1 安装依赖

项目依赖定义在 `requirements.txt`：

```text
numpy
scipy
matplotlib
gymnasium
stable-baselines3
torch
tqdm
rich
```

在任意 Python 虚拟环境中执行：

```bash
python -m pip install -r requirements.txt
```

当前 Windows 工作站已经配置好 GPU 环境：

```text
D:\Anaconda\envs\gpupytorch\python.exe
```

PowerShell 中可以直接使用：

```powershell
$PYTHON = "D:\Anaconda\envs\gpupytorch\python.exe"
& $PYTHON -B -c "import numpy, scipy, gymnasium, torch, stable_baselines3; print('environment ok')"
```

当前已验证的关键版本为：

```text
numpy              1.26.4
scipy              1.13.1
gymnasium          1.1.1
torch              2.5.1
stable-baselines3  2.7.1
```

### 2.2 运行四控制器统一评估

PowerShell：

```powershell
$PYTHON = "D:\Anaconda\envs\gpupytorch\python.exe"
& $PYTHON -B scripts\run_compare.py `
  --controllers lqr wbc_qp rl_ppo rl_ppo_direct `
  --residual-model-path outputs\models\ppo_wheel_leg_residual_random `
  --direct-model-path outputs\models\ppo_wheel_leg_direct `
  --out-dir outputs\compare_four_full_wbc
```

Linux、macOS 或已经激活虚拟环境的终端：

```bash
python -B scripts/run_compare.py \
  --controllers lqr wbc_qp rl_ppo rl_ppo_direct \
  --residual-model-path outputs/models/ppo_wheel_leg_residual_random \
  --direct-model-path outputs/models/ppo_wheel_leg_direct \
  --out-dir outputs/compare_four_full_wbc
```

### 2.3 只运行模型控制器

不加载 PPO 模型时，只需要 NumPy、SciPy 和 Matplotlib：

```bash
python -B scripts/run_compare.py \
  --controllers lqr wbc_qp \
  --out-dir outputs/compare_model_based
```

### 2.4 失败后继续记录曲线

默认情况下，控制器触发失败判据后仿真会提前结束。若需要观察失稳后的完整曲线：

```bash
python -B scripts/run_compare.py \
  --controllers lqr wbc_qp rl_random \
  --continue-after-failure \
  --out-dir outputs/compare_continue_after_failure
```

## 3. 仿真模型

### 3.1 状态、输入和积分

机器人状态为：

```text
[x, y, theta, vx, vy, omega]
```

其中：

- `x`、`y`：机器人髋部在二维平面中的位置；
- `theta`：躯干俯仰角；
- `vx`、`vy`：髋部速度；
- `omega`：躯干角速度。

控制输入为三路力矩：

```text
[tau_wheel, tau_knee, tau_hip]
```

仿真步长为 `0.01 s`，每个固定场景持续 `8 s`。状态更新使用四阶 Runge-Kutta 积分。

### 3.2 主要参数

参数集中定义于 `src/config/params.py`：

| 参数 | 数值 | 含义 |
| --- | ---: | --- |
| `mass` | `50.0 kg` | 机器人质量 |
| `body_inertia` | `5.0 kg*m^2` | 躯干转动惯量 |
| `thigh_length` | `0.5 m` | 大腿长度 |
| `shank_length` | `0.5 m` | 小腿长度 |
| `wheel_radius` | `0.1 m` | 轮半径 |
| `gravity` | `9.81 m/s^2` | 重力加速度 |
| `friction_coeff` | `0.8` | 轮地摩擦系数 |
| `y_ref` | `0.82 m` | 默认髋部目标高度 |
| `nominal_leg_length` | `0.72 m` | 被动腿部弹性元件的平衡长度 |
| `leg_stiffness` | `900 N/m` | 等效腿部刚度 |
| `leg_damping` | `55 N*s/m` | 等效腿部阻尼 |
| `contact_pitch_coupling` | `0.25` | 切向接触力传递到躯干俯仰的等效比例 |
| `tau_limits` | `[30, 160, 120] N*m` | 轮、膝、髋力矩上限 |

### 3.3 地形模型

`src/model/terrain.py` 提供四种地形：

| 类名 | 用途 |
| --- | --- |
| `FlatTerrain` | 平坦地面 |
| `SineTerrain` | 单一正弦地形 |
| `MultiSineTerrain` | 多频正弦叠加，并限制最大高度 |
| `NoiseTerrain` | 平滑伪随机粗糙地形 |

所有地形均提供：

```text
height(x)
slope(x)
curvature(x)
```

### 3.4 失败判据

仿真在以下情况之一出现时标记失败：

| 失败原因 | 判据 |
| --- | --- |
| `theta_fail` | 躯干倾角绝对值超过 `30 deg` |
| `height_fail` | 髋部高度偏离目标高度超过 `0.15 m` |
| `leg_infeasible` | 腿长超出两连杆逆运动学可达范围 |
| `leg_collapsed` | 腿长小于 `0.18 m` |

### 3.5 外力测量与公平评估

场景中的水平外力始终进入前向动力学，但正式评估默认不把真实外力值直接暴露给控制器。这使 LQR、WBC/QP 和两种 PPO 都在相同的盲扰动条件下比较。若要模拟已安装外力传感器或外力估计器，可显式启用：

```bash
python -B scripts/run_compare.py \
  --controllers lqr wbc_qp \
  --provide-force-measurement \
  --out-dir outputs/compare_with_force_sensor
```

## 4. 固定评估场景

`scripts/run_compare.py` 使用 `src/config/scenarios.py` 中的八个固定测试场景：

| 场景 | 目标速度 | 地形 | 外力 |
| --- | ---: | --- | --- |
| `A_impulse_push` | `2 m/s` | 平地 | `5.0 s` 时施加 `100 N`、持续 `0.1 s` 的水平脉冲 |
| `B_constant_push` | `3 m/s` | 平地 | 持续 `80 N` 水平推力 |
| `C_rough_terrain` | `2 m/s` | 振幅 `0.05 m`、波长 `1.0 m` 的正弦地形 | 无 |
| `C_noise_terrain` | `2 m/s` | 平滑随机地形 | 无 |
| `D_large_irregular_terrain` | `2 m/s` | 最大高度约 `0.085 m` 的多频地形 | 无 |
| `E_combined_stress` | `3 m/s` | 最大高度约 `0.090 m` 的多频地形 | 持续 `60 N` 推力，并叠加 `100 N` 脉冲 |
| `F_high_speed_flat` | `5 m/s` | 平地 | 无 |
| `G_requirement_boundary` | `5 m/s` | 振幅 `0.05 m`、波长 `1.0 m` 的正弦地形 | 持续 `100 N` 水平推力 |

场景 A 至 D 用于验证单一扰动下的跟踪能力；场景 E 是峰值达到 `160 N` 的越界压力测试；场景 F 和 G 用于补齐题目规定的 `0-5 m/s` 速度范围，以及 `5 m/s`、`100 N`、不平地形同时存在时的边界工况。

## 5. 控制器实现

### 5.1 LQR

实现文件：

```text
src/controllers/lqr.py
```

LQR 在平坦地面平衡点附近工作。程序根据当前目标速度构造参考状态，通过中心差分数值计算线性化矩阵 `A` 和 `B`，再使用连续代数 Riccati 方程求解反馈增益：

```text
P = solve_continuous_are(A, B, Q, R)
K = inv(R) * B.T * P
tau = tau_eq - K * (state - state_ref)
```

最终力矩会经过统一限幅。该控制器结构简单、计算开销小，但面对持续外力或复杂地形时会出现较明显的稳态偏差。

### 5.2 显式约束降阶 WBC/QP

实现文件：

```text
src/controllers/wbc_qp.py
src/model/dynamics.py
```

WBC/QP 每个控制周期显式优化任务加速度、接触力、执行器力矩和安全松弛量：

```text
z = [
  ax, ay, alpha,
  Ft, Fn,
  tau_wheel, tau_knee, tau_hip,
  leg_low_slack, leg_high_slack,
  height_low_slack, height_high_slack
]
```

其中：

- `ax`、`ay`、`alpha`：期望质心平动和躯干角加速度；
- `Ft`、`Fn`：轮地切向力和法向力；
- `tau_wheel`、`tau_knee`、`tau_hip`：三路执行器力矩；
- 四个 `slack`：预测腿长和预测高度约束的软化量。

#### 硬等式约束

`src/model/dynamics.py` 构造五个仿射等式：

```text
m * ax - Ft = F_ext - c_x * vx - m * g * terrain_slope
m * ay - Fn = -m * g - c_y * vy
I * alpha - contact_moment_arm * Ft - 0.15 * tau_knee - tau_hip
  = -k_theta * theta - c_theta * omega
tau_wheel - wheel_radius * Ft = 0
tau_knee + 0.6 * tau_hip - leg_moment_arm * Fn
  = -leg_moment_arm * passive_leg_force
```

其中：

```text
passive_leg_force =
  leg_stiffness * (nominal_leg_length - leg_length)
  - leg_damping * leg_rate
contact_moment_arm =
  contact_pitch_coupling * max(y - terrain_height, 0.1)
```

外力按题意作用于躯干质心，因此只进入水平平动方程，不会凭空产生俯仰力矩。地形通过轮心高度、腿长、腿长变化率、坡度阻力和预瞄约束影响运动，不再使用曲率乘速度平方的人工竖直加速度项。

#### 硬不等式和边界约束

```text
|Ft| <= friction_coeff * Fn
Fn >= 0.35 * mass * gravity
|tau_i| <= tau_limit_i
slack_i >= 0
```

#### 预测安全约束

控制器使用 `0.30 s` 的地形预瞄窗口，根据未来地形高度和坡度线性预测腿长及髋部高度。预测腿长和高度必须保持在安全范围内；必要时允许使用高惩罚权重的松弛量，以避免优化问题在边界场景中直接不可行。

#### 目标函数

目标函数综合考虑：

- 任务加速度跟踪误差；
- 力矩大小；
- 相邻周期力矩变化；
- 接触力正则项；
- 腿长松弛量；
- 高度松弛量。

当前问题在数学上是带线性等式和不等式约束的凸二次规划，代码将其装配为标准 QP 形式 `0.5 z.T @ P @ z + q.T @ z`、`l <= A @ z <= u`，并使用 OSQP 专用 QP 求解器求解。若候选解出现非有限值、等式残差过大或不等式约束违反量过大，控制器会回退到稳定基线初值。

#### WBC/QP 诊断量

WBC/QP 会额外记录：

```text
qp_success
qp_solver_success
qp_fallback
qp_iterations
qp_solver_status_val
qp_objective
qp_primal_residual
qp_dual_residual
qp_solve_time_ms
qp_eq_residual
qp_ineq_violation
qp_friction_ratio
qp_leg_slack
qp_height_slack
qp_predicted_leg_length
qp_predicted_height
qp_planned_contact_normal
qp_planned_contact_tangent
```

每个 WBC/QP 场景都会生成一张 `*__diagnostics.png` 图，展示摩擦边界、预测腿长、残差、松弛量、求解状态和回退状态。

### 5.3 残差 PPO

实现文件：

```text
src/controllers/rl_policy.py
src/controllers/base.py
src/rl/env.py
```

残差 PPO 输出范围为 `[-1, 1]` 的三维动作。动作经过缩放后叠加到稳定基线控制器力矩上：

```text
tau = stabilizing_baseline_tau + action * rl_residual_scale
```

这种方式降低了训练难度，使策略重点学习模型误差、地形扰动和外力扰动下的修正量。

最终评估使用的模型为：

```text
outputs/models/ppo_wheel_leg_residual_random.zip
```

### 5.4 直接力矩 PPO

直接 PPO 同样输出三维归一化动作，但动作会直接映射为力矩：

```text
tau_wheel = action[0] * 30
tau_knee  = 0.5 * (action[1] + 1) * 160
tau_hip   = action[2] * 120
```

膝关节是主要伸腿执行器，因此映射到 `[0, 160] N*m`；轮和髋关节仍采用对称范围。由于策略需要同时学会支撑身体、保持姿态和跟踪速度，训练难度高于残差 PPO，适合使用分阶段课程训练。

最终评估使用的模型为：

```text
outputs/models/ppo_wheel_leg_direct.zip
```

### 5.5 PPO 模型加载兼容

`src/controllers/rl_policy.py` 在加载已有模型时会显式传入固定的观测空间和动作空间，并兼容 NumPy 1.x 与 NumPy 2.x 的私有模块路径差异。这样可以在当前 `numpy 1.26.4` 环境中读取较新环境保存的 PPO 模型，无需为了反序列化升级整个科学计算环境。

## 6. RL 环境与训练

### 6.1 观测空间

PPO 的观测向量为：

```text
[
  y - y_ref,
  theta,
  vx - v_cmd,
  vy,
  omega,
  leg_length,
  terrain_height,
  terrain_slope,
  v_cmd
]
```

观测维度为 `9`，动作维度为 `3`。

### 6.2 奖励函数

`src/rl/env.py` 中的奖励函数综合考虑：

- 高度误差；
- 姿态误差；
- 速度误差；
- 动作幅值；
- 加速度幅值；
- 直接 PPO 的力矩幅值；
- 动作接近饱和时的惩罚；
- 触发失败判据后的额外惩罚。

### 6.3 随机训练场景

`src/config/random_scenarios.py` 提供三类随机场景：

| 场景集合 | 速度范围 | 地形 | 外力 |
| --- | --- | --- | --- |
| `random_easy` | `0-2 m/s` | 平地 | 无 |
| `random_force` | `0-3 m/s` | 平地 | 平滑随机脉冲或持续外力 |
| `random_full` | `0-5 m/s` | 平地、正弦、多频或噪声地形 | 平滑随机脉冲或持续外力 |

固定评估场景与随机训练场景相互分离，避免直接在测试集上训练。

### 6.4 训练残差 PPO

```bash
python scripts/train_rl.py \
  --action-mode residual \
  --scenario-set random_full \
  --timesteps 300000 \
  --n-envs 4 \
  --save-path outputs/models/ppo_wheel_leg_residual_random
```

单独评估残差 PPO：

```bash
python scripts/eval_rl.py \
  --model-path outputs/models/ppo_wheel_leg_residual_random \
  --out-dir outputs/eval_rl
```

### 6.5 分阶段训练直接 PPO

阶段 1：在简单平地任务上学习支撑和基础移动。

```bash
python scripts/train_rl.py \
  --action-mode direct \
  --scenario-set random_easy \
  --timesteps 1000000 \
  --n-envs 8 \
  --save-path outputs/models/ppo_wheel_leg_direct_stage1
```

阶段 2：加入随机外力。

```bash
python scripts/train_rl.py \
  --action-mode direct \
  --scenario-set random_force \
  --load-path outputs/models/ppo_wheel_leg_direct_stage1 \
  --timesteps 1000000 \
  --n-envs 8 \
  --save-path outputs/models/ppo_wheel_leg_direct_stage2
```

阶段 3：加入随机地形和随机外力。

```bash
python scripts/train_rl.py \
  --action-mode direct \
  --scenario-set random_full \
  --load-path outputs/models/ppo_wheel_leg_direct_stage2 \
  --timesteps 1500000 \
  --n-envs 8 \
  --save-path outputs/models/ppo_wheel_leg_direct
```

## 7. 输出文件说明

### 7.1 指标文件

批量评估会生成：

```text
outputs/<run_name>/metrics.csv
```

通用指标包括：

```text
success
failure_reason
rmse_h
max_abs_h
rmse_theta_deg
max_abs_theta_deg
rmse_v
sat_ratio
```

内部计算还包含运行时长、jerk、能量和平均力矩范数。WBC/QP 额外输出 QP 可行解比例、OSQP 求解成功比例、回退比例、平均迭代次数、平均/最大求解耗时、最大原始/对偶残差、最大约束残差、最大摩擦利用率和最大安全松弛量。

### 7.2 日志文件

每个控制器和场景对应一个 `.npz` 日志：

```text
outputs/<run_name>/logs/<scenario>__<controller>.npz
```

日志保存状态、参考速度、外力、地形、腿长、力矩、接触力、加速度、失败状态和 QP 诊断量。

### 7.3 图像文件

常规图像：

```text
outputs/<run_name>/figures/<scenario>__<controller>.png
```

WBC/QP 专用图像：

```text
outputs/<run_name>/figures/<scenario>__wbc_qp__diagnostics.png
```

多控制器叠加对比图：

```text
outputs/<run_name>/figures/<scenario>__controller_comparison.png
```

## 8. 生成报告

报告源文件：

```text
report.tex
```

最终 PDF：

```text
report.pdf
```

项目根目录中的 `.latexmkrc` 已配置 XeLaTeX。生成报告：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
```

报告当前引用：

```text
outputs/compare_four_full_wbc/figures/
```

因此不要删除 `outputs/compare_four_full_wbc/`，除非随后重新运行四控制器统一评估。

## 9. 回归测试

运行核心逻辑测试：

```bash
python -B -m unittest discover -s tests -v
```

测试覆盖质心外力语义、WBC 仿射等式与前向动力学一致性、随机地形曲率平滑性、准确仿真时域、失败状态日志、外力测量开关和题目边界场景。

## 10. 项目结构

```text
RoBoT-of-two-beauty/
├── .gitignore
├── README_CODE.md
├── requirements.txt
├── report.tex
├── report.pdf
├── 题目.pdf
├── scripts/
│   ├── run_compare.py          # 固定场景批量评估入口
│   ├── train_rl.py             # PPO 训练入口
│   └── eval_rl.py              # 残差 PPO 单独评估入口
├── tests/
│   └── test_core_logic.py       # 动力学、仿真循环和场景回归测试
├── src/
│   ├── config/
│   │   ├── params.py           # 机器人、仿真和安全参数
│   │   ├── scenarios.py        # 固定场景与基础训练场景
│   │   └── random_scenarios.py # 随机训练场景采样器
│   ├── controllers/
│   │   ├── base.py             # 控制器接口、基线力矩和动作映射
│   │   ├── lqr.py              # LQR 控制器
│   │   ├── wbc_qp.py           # 显式约束降阶 WBC/QP 控制器
│   │   └── rl_policy.py        # PPO 控制器与模型加载兼容逻辑
│   ├── model/
│   │   ├── dynamics.py         # 动力学、RK4、失败判据和 WBC 仿射模型
│   │   ├── kinematics.py       # 轮心高度、腿部逆运动学和 RL 观测
│   │   └── terrain.py          # 地形模型
│   ├── rl/
│   │   └── env.py              # Gymnasium 环境与奖励函数
│   ├── simulation/
│   │   ├── runner.py           # 仿真循环
│   │   └── logger.py           # 日志收集与 .npz 保存
│   └── evaluation/
│       ├── metrics.py          # 指标计算和 CSV 输出
│       └── plots.py            # 常规曲线、四控制器对比图与 WBC/QP 诊断图
└── outputs/
    ├── models/                 # PPO 模型
    └── compare_four_full_wbc/  # 报告使用的最终统一评估结果
```

各级 `__init__.py` 用于将目录声明为 Python 包。

## 11. 可删除的临时文件

以下文件均可重新生成，不影响源码：

```text
.DS_Store
report.aux
report.fdb_latexmk
report.fls
report.log
report.out
report.synctex.gz
report.synctex(busy)
report.toc
report.xdv
pdflatex*.fls
xelatex*.fls
outputs/.mplconfig/
outputs/_review*/
outputs/models/_review*.zip
src/**/__pycache__/
scripts/**/__pycache__/
```

删除报告临时文件前，建议先关闭编辑器中的 PDF 预览，以免 `report.synctex.gz` 仍被占用。

以下目录和文件应当保留：

```text
outputs/models/
outputs/compare_four_full_wbc/
report.tex
report.pdf
src/
scripts/
requirements.txt
题目.pdf
```

## 12. 实现边界

本项目面向课程报告和控制方法对比，采用二维降阶模型，而不是完整多刚体动力学引擎。当前 WBC/QP 已显式描述任务加速度、接触力、摩擦锥、执行器映射、力矩限制和预测安全约束，并使用 OSQP 专用 QP 求解器求解标准凸 QP；但它仍属于与现有降阶仿真匹配的 WBC/QP。若进一步面向高频实时控制，可以引入显式关节坐标、刚体质量矩阵、接触雅可比，并复用 QP 工作区以降低在线求解开销。
