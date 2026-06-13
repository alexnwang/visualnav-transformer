import pickle
from collections import defaultdict
BASE="/home/anw2067/visualnav-transformer/train/data_splits/nymeria/test/"
OUT=BASE+"viz_shortlists/"
pool=pickle.load(open(BASE+"viz_candidates_dist8.pkl","rb"))
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
q=[d for d in pool if d["curr_time"]>=3 and hvis(d) and 0.5<=d["loco"]<=2.5 and d["arm_max"]>=0.5]
dd=dedup(q)
tasks=[{"track":d["track"],"curr_time":int(d["curr_time"]),"goal_time":int(d["goal_time"])} for d in dd]
pickle.dump(tasks,open(OUT+"tasks_balanced_dedup.pkl","wb"))
print(f"balanced (loco[0.5,2.5], arm>=0.5): qualifying={len(q)} deduped={len(tasks)} tracks={len(set(t['track'] for t in tasks))}")
