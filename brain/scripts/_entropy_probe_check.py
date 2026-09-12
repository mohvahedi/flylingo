"""End-to-end check of the SHIPPED entropy term in PolicyAdapter.observe.

Isolates the entropy term by making the REINFORCE factor exactly zero
(reward == baseline -> advantage == 0) and the dopamine factor off, so the only
thing that can move the logits is the entropy line. Then it compares the logit
change the adapter actually produces with a central finite difference of
H(z) = -sum p log p.

Run: .venv/Scripts/python.exe scripts/_entropy_probe_check.py
"""
import numpy as np

from brain.learning.adapter import PolicyAdapter


def H(z):
    z = np.asarray(z, np.float64)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    return float(-(p * np.log(p + 1e-300)).sum())


def softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def fd_grad(z, eps=1e-6):
    return np.array([(H(z + eps * e) - H(z - eps * e)) / (2 * eps)
                     for e in np.eye(len(z))], np.float64)


def zerosum(v):
    v = np.asarray(v, np.float64)
    return v - v.mean()


def cos(a, b):
    a, b = zerosum(a), zerosum(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


print('=== A0. the expressions themselves vs central finite difference of H ===')
worst_new, worst_old = 0.0, 0.0
for trial in range(12):
    rng = np.random.default_rng(500 + trial)
    n_act = [4, 8][trial % 2]
    z = rng.uniform(-4, 4, n_act)
    p = softmax(z)
    Hv = H(z)
    fd = fd_grad(z)
    new = -p * (Hv + np.log(p + 1e-300))
    old = p * (Hv - np.log(p + 1e-300))
    worst_new = max(worst_new, float(np.max(np.abs(new - fd))))
    worst_old = max(worst_old, float(np.max(np.abs(old - fd))))
print(f'  max |  -p(H+log p)  - FD(dH/dz) | = {worst_new:.3e}   <- SHIPPED line')
print(f'  max |   p(H-log p)  - FD(dH/dz) | = {worst_old:.3e}   <- old line')

print()
print('=== A. live adapter logit change vs central finite difference ===')
worst_cos, worst_abs = 1.0, 0.0
for trial in range(6):
    rng = np.random.default_rng(100 + trial)
    f = rng.standard_normal(128)
    f /= np.sqrt(np.mean(f * f))
    n_act = [4, 8][trial % 2]
    # entropy_coef=1 arm and a reward that exactly cancels the baseline, so the
    # REINFORCE factor contributes nothing and the dopamine path is off.
    def build(coef, n_act=n_act, f=f):
        a = PolicyAdapter(in_dim=128, n_actions=n_act, hidden=16, seed=0,
                          entropy_coef=coef, dopamine=False)
        a.W1 = (np.random.default_rng(7).standard_normal(a.W1.shape) / 4).astype(np.float32)
        a.W2 = (np.random.default_rng(8).standard_normal(a.W2.shape) / 4).astype(np.float32)
        return a
    a1 = build(1.0)
    a0 = build(0.0)
    z0 = a1.logits(f).astype(np.float64)
    assert np.allclose(z0, a0.logits(f).astype(np.float64))
    _, logp_act = a1.sample(f, np.random.default_rng(trial))
    a0.sample(f, np.random.default_rng(trial))
    # reward == baseline == 0 -> advantage 0 -> only the entropy line can move anything
    a1.observe(reward=0.0, logprob=logp_act)
    a0.observe(reward=0.0, logprob=logp_act)
    z1 = a1.logits(f).astype(np.float64)
    z0_ref = a0.logits(f).astype(np.float64)
    delta_z = z1 - z0
    zpre = z0 if np.allclose(z0_ref, z0) else (z1 - delta_z)
    fd = fd_grad(zpre)
    old = softmax(zpre) * (H(zpre) - np.log(softmax(zpre) + 1e-300))
    c = cos(delta_z, fd)
    worst_cos = min(worst_cos, c)
    worst_abs = max(worst_abs, float(np.max(np.abs(zerosum(delta_z) -
                                                     zerosum(fd) * (np.linalg.norm(zerosum(delta_z)) /
                                                                    (np.linalg.norm(zerosum(fd)) + 1e-300))))))
    print(f'  trial {trial} n_actions={n_act} lr={a1.lr}: '
          f'cos(delta_logits, FD grad of H) = {c:+.6f} | '
          f'cos(delta_logits, OLD p*(H-log p)) = {cos(delta_z, old):+.6f} | '
          f'H before/after = {H(zpre):.6f} -> {H(z1):.6f} (rose: {H(z1) > H(zpre)})')
print('  worst cosine to the true FD gradient over trials: %+.6f  (below 1 because the'
      ' adapter steps in parameter space, so the logit move is J J^T dz, a smoothed dz)'
      % worst_cos)
print(f'  worst |delta_logits - FD| after matching magnitudes: {worst_abs:.3e}')

print()
print('=== B. gradient ascent on z raises H (entropy bonus points up) ===')
z = np.array([0.9, -1.2, 0.3, -0.05], np.float64)
print(f'  H(start) = {H(z):.9f}')
for k in range(60):
    z = z + 0.25 * fd_grad(z)          # ascend H
    if k in (0, 4, 19, 59):
        print(f'  step {k + 1:>2}: H = {H(z):.9f}  z = {z.round(4)}')
print(f'  log(4) = {np.log(4):.9f} (uniform). Ascent converges to it and does not overshoot:'
      f' H <= log 4 = {H(z) <= np.log(4) + 1e-12}')

print()
print('=== C. the adapter line at the uniform point ===')
a = PolicyAdapter(in_dim=128, n_actions=4, hidden=16, seed=0, entropy_coef=1.0, dopamine=False)
a.W1[...] = 0.0
a.W2[...] = 0.0                     # every logit is 0 -> uniform policy
f = np.ones(128, np.float32)
z0 = a.logits(f).copy()
print(f'  z at uniform: {z0}')
a.observe(reward=0.0, logprob=float(np.log(0.25)))
z1 = a.logits(f).copy()
print(f'  FD gradient of H at uniform:      {fd_grad(z0.astype(np.float64)).round(12)}')
print(f'  adapter logit move at uniform:    {(z1 - z0).round(12)}')
print(f'  old formula p*(H - log p) would move logits the other way: '
      f'{(softmax(z0) * (H(z0) - np.log(softmax(z0)))).round(6)}')
best = -1.0
rng = np.random.default_rng(3)
for _ in range(200):
    zz = rng.normal(0, 2.0, 4)
    for _ in range(400):
        zz = zz + 0.5 * fd_grad(zz)
    best = max(best, H(zz))
print(f'  uniform is the global top: best H found by 200x400-step ascents = {best:.12f},'
      f' log 4 = {np.log(4):.12f}, never exceeds it: {best <= np.log(4) + 1e-9}')
