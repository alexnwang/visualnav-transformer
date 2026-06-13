import re, csv, statistics as st
from collections import defaultdict
OUT="/home/anw2067/slurm_logs/policy_mje/mje-9867760.out"
SHORT="/home/anw2067/visualnav-transformer/train/data_splits/nymeria/test/viz_shortlists/"
rows=[]
pat=re.compile(r'\[(\S+) (\d+)/(\d+)\]\s+(\S+)\s+best_mje=([\d.]+)\s+leaf=([\d.]+)\s+init_mje=([\d.]+)')
for line in open(OUT):
    m=pat.search(line)
    if m:
        rows.append((m.group(1), m.group(4), float(m.group(5)), float(m.group(6)), float(m.group(7))))
# CSV
with open(SHORT+"mje_4sets_draw_mask_N64.csv","w") as f:
    w=csv.writer(f); w.writerow(["set","task","mje","leaf","init_mje"])
    for s,t,mje,leaf,im in rows: w.writerow([s,t,f"{mje:.6f}",f"{leaf:.6f}",f"{im:.6f}"])
print("wrote CSV with", len(rows), "rows")
# figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
pred=defaultdict(list); init=defaultdict(list)
for s,t,mje,leaf,im in rows: pred[s].append(mje); init[s].append(im)
allv=[r[2] for r in rows]; hi=float(np.percentile(allv,99)); edges=np.linspace(0,hi,31)
colors={"hand":"#4c9be8","balanced-hand":"#1f4e8c","lateral":"#e8794c","balanced-lateral":"#a83232"}
fig,ax=plt.subplots(figsize=(8,5))
for s in ["hand","balanced-hand","lateral","balanced-lateral"]:
    v=pred[s]; iv=init[s]; c=colors[s]; med=np.median(v); im=np.median(iv)
    ax.hist(v,bins=edges,histtype="step",density=True,linewidth=2,color=c,
            label=f"{s} (n={len(v)}, med={med:.3f}, init={im:.3f})")
    ax.hist(iv,bins=edges,histtype="step",density=True,linewidth=1.2,color=c,ls="--",alpha=0.7)
ax.set_xlabel("final-pose MJE (m)  —  solid: best-of-64 policy, dashed: stay-put baseline")
ax.set_ylabel("density"); ax.set_title("Policy MJE vs stay-put over 4 motion task sets (N=64)")
ax.legend(); fig.tight_layout(); fig.savefig(SHORT+"mje_4sets_draw_mask_N64.png",dpi=130)
print("wrote figure")
print("\n=== final per-set (set | n | mje med | mean | init med | gain) ===")
for s in ["hand","balanced-hand","lateral","balanced-lateral"]:
    v=pred[s]; iv=init[s]; mm=st.median(v); imm=st.median(iv)
    print(f"  {s:16s} | {len(v):4d} | {mm:.3f} | {sum(v)/len(v):.3f} | {imm:.3f} | {imm/mm:.2f}x")
