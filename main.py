# As equações foram tiradas do documento original:
# HODGKIN, A. L.; HUXLEY, A. F. A quantitative description of ion currents
# and its applications to conduction and excitation in nerve membranes.
# Journal of Physiology, v. 117, n. 1, p. 500–544, 1952.
# Os cálculos dos alphas e betas foram retirados do livro:
# Fonte: IZHIKEVICH, E. M. Dynamical Systems in Neuroscience : The Geometry of
# Excitability and Bursting. London: The MIT Press, 2007.
import numpy as np
import matplotlib.pyplot as plt

# Calcula Alpha N
def alpha_n(v):
    return 0.01 * ((10 - v) / (np.exp((10 - v) / 10) - 1))

# Calcula Alpha M
def alpha_m(v):
    return 0.1 * ((25 - v) / (np.exp((25 - v) / 10) - 1))

# Calcula Alpha H
def alpha_h(v):
    return 0.07 * np.exp(-v / 20)

# Calcula Beta N
def beta_n(v):
    return 0.125 * np.exp(-v / 80)

# Calcula Beta M
def beta_m(v):
    return 4 * np.exp(-v / 18)

# Calcula Beta H
def beta_h(v):
    return 1 / (np.exp((30 - v) / 10) + 1)

# Cria um sinal de estímulo elétrico externo
def estimulo(t, impulso):
    est = np.zeros(len(t))
    for i in range(0, len(impulso)):
        est[t >= impulso[i][0]] = impulso[i][2]
        est[t >= impulso[i][1]] = 0
    return est

# Constantes
# Potencial de equilíbrio K+, Na+ e potencial em que ocorre a corrente
# de fuga
Ek = -12
ENa = 120
El = 10.6
# Valor absoluto do potencial de repouso
Er = 0
# Condutância máxima por cm²
gk = 36
gna = 120
gl = 0.3
# Capacidade da Membrana
C = 1

# Frequência de 1000 Hz. f=1/t
dt = 0.001
# Tempo de 0 a 100 milisegundos
t = np.arange(0, 100, dt)
est = estimulo(t, [[20, 100, 10]])

# Calcula os valores de n, m e h para t=0
n = alpha_n(0) / (alpha_n(0) + beta_n(0))
m = alpha_m(0) / (alpha_m(0) + beta_m(0))
h = alpha_h(0) / (alpha_h(0) + beta_h(0))

# Deslocamento do potencial da membrana
v = np.zeros(len(t))

for i in range(len(t) - 1):
    # Calcula as correntes iônicas INa, IK e Il
    # Ik = gk(V-Vk) - Eq. A.4
    # Vk = Ek - Er
    # gk = gk' n**4 - Eq. A.6
    Ik = gk * np.power(n, 4) * (v[i] - (Ek - Er))
    # INa = gna(V-VNa) - Eq. A.3
    # VNa = ENa - Er
    # gna = m**3 h gna'  - Eq. A.8
    INa = np.power(m, 3) * h * gna * (v[i] - (ENa - Er))
    # Il = gl'(V-Vl) - Eq. A.5
    # Vl = El - Er
    Il = gl * (v[i] - (El - Er))
    # Corrente iônica: I = INa + Ik + Il - Eq. A.2
    I = INa + Ik + Il

    # Realiza o cálculo das equações: A.1, A.9, A.7, A.10
    # dv/dt = f(x,y)
    dvdt = (est[i] - I) / C
    dmdt = alpha_m(v[i]) * (1 - m) - beta_m(v[i]) * m
    dndt = alpha_n(v[i]) * (1 - n) - beta_n(v[i]) * n
    dhdt = alpha_h(v[i]) * (1 - h) - beta_h(v[i]) * h

    # Aplica o Método de Euler para determinar o próximo ponto da
    # equação diferencial utilizando um único passo.
    # y(n+1) = y(n) + h * f(x, y)
    v[i + 1] = v[i] + dt * dvdt
    m = m + dt * dmdt
    h = h + dt * dhdt
    n = n + dt * dndt

fig, axs = plt.subplots(2, 1)
axs[0].plot(t, est)
axs[0].set_xlabel("Tempo (ms)")
axs[0].set_ylabel("Estímulo Elétrico")
axs[0].grid(True)

axs[1].plot(t, v)
axs[1].set_xlabel("Tempo (ms)")
axs[1].set_ylabel("Tensão da membrana (mV)")
axs[1].grid(True)

fig.suptitle("Modelo de Hodgkin-Huxley")
fig.tight_layout()
plt.show()
