
import sys, time, json
import numpy as np
sys.path.insert(0, r"D:/Projects/flylingo/brain")
from brain.reservoir import FlyReservoir, load_connectome
from brain.learning.train import load_task, features_for, encode_challenge

t0=time.time(); conn = load_connectome(); t1=time.time()
r = FlyReservoir(conn, embedding_dim=256, dims=128, seed=7301)
print("load_connectome s=%.2f" % (t1-t0))
print("modes ok:", r.mode)
t0=time.time()
tasks = load_task(eval_fraction=0.25, seed=0)
print("task", tasks.summary())
E = np.stack([encode_challenge(c) for c in tasks.challenges])
print("embeddings unique rows:", len({e.tobytes() for e in E}))
F = features_for(r, "intact", tasks.challenges)
print("feats", F.shape, "per-step time s=%.3f" % ((time.time()-t0)/len(tasks.challenges)))
print("feat RMS per row: mean %.4f min %.4f max %.4f" % (np.sqrt((F**2).mean(1)).mean(), np.sqrt((F**2).mean(1)).min(), np.sqrt((F**2).mean(1)).max()))
print("between-challenge variance of mean-feature: %.6f vs within %.6f" % (F.mean(0).var(), F.var(0).mean()))
D = np.linalg.norm(F[:,None,:]-F[None,:,:], axis=2)
print("pairwise feature dist: min %.4f mean %.4f max %.4f" % (D[D>0].min(), D[D>0].mean(), D.max()))
ci = np.array([int(c["correctIndex"]) for c in tasks.challenges])
print("correctIndex distribution all:", np.bincount(ci, minlength=4))
print("train labels:", np.bincount(np.array([int(c['correctIndex']) for c in tasks.train]), minlength=4))
print("eval labels:", np.bincount(np.array([int(c['correctIndex']) for c in tasks.eval]), minlength=4))
