import pickle
from collections import defaultdict
BASE="/home/anw2067/visualnav-transformer/train/data_splits/nymeria/test/"
OUT=BASE+"viz_shortlists/"
hp=pickle.load(open(BASE+"viz_candidates_dist8.pkl","rb"))
lp=pickle.load(open(BASE+"viz_candidates_dist8_lat.pkl","rb"))
def hvis(d): return bool(d.get("rh_vis")) or bool(d.get("lh_vis"))
def dedup(q,stride=8):
    bt=defaultdict(list)
    for d in q: bt[d["track"]].append(d)
    k=[]
    for t,r in bt.items():
        r.sort(key=lambda d:d["curr_time"]); last=-10**9
        for d in r:
            if d["curr_time"]-last>=stride: k.append(d); last=d["curr_time"]
    return k
def latfeat(d):
    if d["totR"]>=d["totL"]: lat,tot,vis=d["latR"],d["totR"],d.get("rh_vis")
    else: lat,tot,vis=d["latL"],d["totL"],d.get("lh_vis")
    return lat,(lat/tot if tot>1e-9 else 0.0),bool(vis)
def save(name,dd):
    tasks=[{"track":d["track"],"curr_time":int(d["curr_time"]),"goal_time":int(d["goal_time"])} for d in dd]
    pickle.dump(tasks,open(OUT+name,"wb"))
    print(f"{name}: n={len(tasks)} tracks={len(set(t['track'] for t in tasks))}")
LO,HI=0.25,0.75
bh=[d for d in hp if d["curr_time"]>=3 and hvis(d) and LO<=d["loco"]<=HI and d["arm_max"]>=0.5]
save("tasks_balanced_hand_dedup.pkl",dedup(bh))
bl=[]
for d in lp:
    if d["curr_time"]<3 or not (LO<=d["loco"]<=HI): continue
    lat,lf,vis=latfeat(d)
    if vis and lf>=0.74 and lat>=0.3: bl.append(d)
save("tasks_balanced_lateral_dedup.pkl",dedup(bl))
# drop the stale single 'balanced' pickle to avoid confusion
import os
old=OUT+"tasks_balanced_dedup.pkl"
if os.path.exists(old): os.remove(old); print("removed stale tasks_balanced_dedup.pkl")
