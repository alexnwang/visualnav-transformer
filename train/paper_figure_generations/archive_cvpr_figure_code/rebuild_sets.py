import pickle
from collections import defaultdict

BASE="/home/anw2067/visualnav-transformer/train/data_splits/nymeria/test/"
OUT=BASE+"viz_shortlists/"
hand_pool=pickle.load(open(BASE+"viz_candidates_dist8.pkl","rb"))
lat_pool =pickle.load(open(BASE+"viz_candidates_dist8_lat.pkl","rb"))
CTX=3  # context_size; guard curr_time-CTX>=0

def hvis(d): return bool(d.get("rh_vis")) or bool(d.get("lh_vis"))
def dedup(qual, stride=8):
    bytrack=defaultdict(list)
    for d in qual: bytrack[d["track"]].append(d)
    kept=[]
    for t,rows in bytrack.items():
        rows.sort(key=lambda d:d["curr_time"]); last=-10**9
        for d in rows:
            if d["curr_time"]-last>=stride: kept.append(d); last=d["curr_time"]
    return kept
def lat_feat(d):
    if d["totR"]>=d["totL"]: lat,tot,vis=d["latR"],d["totR"],d.get("rh_vis")
    else:                    lat,tot,vis=d["latL"],d["totL"],d.get("lh_vis")
    return lat,(lat/tot if tot>1e-9 else 0.0),bool(vis)

hand_q=[d for d in hand_pool if d["curr_time"]>=CTX and hvis(d) and d["arm_max"]>=0.844]
bal_q =[d for d in hand_pool if d["curr_time"]>=CTX and hvis(d) and d["loco"]>=1.774 and d["arm_max"]>=0.593]
lat_q=[]
for d in lat_pool:
    if d["curr_time"]<CTX: continue
    lat,lf,vis=lat_feat(d)
    if vis and lat>=0.686 and lf>=0.74: lat_q.append(d)

for name,q in [("hand",hand_q),("lateral",lat_q),("balanced",bal_q)]:
    dd=dedup(q)
    tasks=[{"track":d["track"],"curr_time":int(d["curr_time"]),"goal_time":int(d["goal_time"])} for d in dd]
    p=OUT+f"tasks_{name}_dedup.pkl"
    pickle.dump(tasks,open(p,"wb"))
    print(f"{name:9s} deduped={len(tasks):4d} tracks={len(set(t['track'] for t in tasks))}  -> {p}")
