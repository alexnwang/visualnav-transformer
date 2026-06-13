SIF=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
OVERLAY=/scratch/anw2067/nymeria.sqf
S=/home/anw2067/visualnav-transformer/train/paper_figure_generations/extract_highres_frame.py
declare -A G=(
 [20231106_s1_amanda_rodgers_act0_uu42ld]=3149
 [20230814_s0_leah_gaines_act1_4u9p3x]=1013
 [20231215_s1_dylan_lambert_act0_ud1ljw]=82
 [20231110_s0_thomas_brown_act4_x3t73z]=2967
 [20231027_s1_stacie_cross_act2_kijh3i]=1046
 [20230731_s0_tammy_campos_act4_m94oql]=2263
 [20230905_s1_elizabeth_morgan_act1_i1tdvq]=2166
 [20231128_s1_michael_vargas_act2_16jyco]=880
)
for trk in "${!G[@]}"; do
  echo "=== $trk goal=${G[$trk]} ==="
  singularity exec --overlay ${OVERLAY}:ro ${SIF} bash -lc \
    "conda activate nomad_train2 && python $S --track $trk --curr_times ${G[$trk]}" 2>&1 | tail -2
done
echo "ALL_GOAL_FRAMES_DONE"
