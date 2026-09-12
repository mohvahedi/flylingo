
import numpy as np
def sm64(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
def H64(z):
    p = sm64(z)
    return float(-(p*np.log(p)).sum())
def true64(z):
    p = sm64(z); h = -(p*np.log(p)).sum()
    return -p*(h + np.log(p))
def coded64(z):
    p = sm64(z); h = -(p*np.log(p)).sum()
    return p*(h - np.log(p))
def fd64(z, eps=1e-6):
    g = np.zeros_like(z)
    for j in range(len(z)):
        zp = z.copy(); zp[j]+=eps; zm=z.copy(); zm[j]-=eps
        g[j] = (H64(zp)-H64(zm))/(2*eps)
    return g
rng = np.random.default_rng(0)
worst_t = worst_c = 0.0
for _ in range(200):
    z = rng.standard_normal(4)*1.5
    f = fd64(z)
    worst_t = max(worst_t, np.abs(f-true64(z)).max())
    worst_c = max(worst_c, np.abs(f-coded64(z)).max())
print("float64 over 200 random z (4 actions):")
print("  max |FD - TRUE formula|  = %.3e   <- formula is correct" % worst_t)
print("  max |FD - AS-CODED term| = %.3e   <- as-coded term is not a gradient" % worst_c)
z = rng.standard_normal(4)*1.5
c = coded64(z) - coded64(z).mean(); t = true64(z) - true64(z).mean()
print("  zero-sum part: cos(as-coded, true) = %.4f  (negative => as-coded sharpens)" % float(c@t/(np.linalg.norm(c)*np.linalg.norm(t))))
# Effect on entropy magnitude over a trajectory: ascent with each formula
z0 = rng.standard_normal(4)*0.1
for nm, fn in (("true", true64), ("as-coded", coded64)):
    z = z0.copy()
    for _ in range(200):
        z += 0.05*fn(z)
    print("  200 steps * 0.05 %-9s => H %.6f  (start %.6f)" % (nm, H64(z), H64(z0)))
