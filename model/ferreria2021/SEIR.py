import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ設定
N = 100
β = 1.5
γ = 0.1
σ = 1/3  # 例として潜伏期間を3日とする
t_span = [0, 160]  # 時間の範囲
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # 評価する時間の点
init_values = [N - 1, 0, 1, 0]  # 初期状態: [S0, E0, I0, R0]

# SEIRモデルの微分方程式
def seir_model(t, y, N, β, γ, σ):
    S, E, I, R = y
    dSdt = -β * S * I / N
    dEdt = β * S * I / N - σ * E
    dIdt = σ * E - γ * I
    dRdt = γ * I
    return [dSdt, dEdt, dIdt, dRdt]

# ODEを解く
solution = solve_ivp(seir_model, t_span, init_values, args=(N, β, γ, σ), t_eval=t_eval, method='RK45')

# 解を取得
S, E, I, R = solution.y

# プロット
plt.figure(figsize=(10, 6))
plt.plot(t_eval, S, label='S: Susceptible')
plt.plot(t_eval, E, label='E: Exposed')
plt.plot(t_eval, I, label='I: Infectious')
plt.plot(t_eval, R, label='R: Recovered')
plt.xlabel('Time / days')
plt.ylabel('Number of people')
plt.legend()
plt.title('SEIR Model')
plt.grid(True)
plt.show()
