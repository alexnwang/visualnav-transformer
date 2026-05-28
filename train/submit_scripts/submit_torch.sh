#!/bin/bash

# Submit a config to one or more GPU profiles. When >1 profile is given, they
# race under a shared RACE_TAG and the first one to start cancels the rest.
#
# Usage: ./submit_torch.sh <config.yaml> [profile1 profile2 ...]
#
# If no profiles are given, the default trio races: l40s4 a100 h100.
# Unknown / non-yaml positional args (e.g. a stale `4` from older entries in
# submit_experiments.sh) are skipped with a warning.

# Profile definitions: gpu_type num_gpus cpus mem_gb
declare -A PROFILES=(
    [l40s2]="l40s 2 32 200"
    [l40s4]="l40s 4 64 400"
    [l40s8]="l40s 8 128 800"
    [a100]="a100 1 16 120"
    [a100x2]="a100 2 32 200"
    [a100x4]="a100 4 64 400"
    [h100]="h100 1 16 120"
    [h100x2]="h100 2 32 200"
)

CONFIG_FILE=${1}

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE argument is required"
    echo "Usage: $0 <config.yaml> [profile1 profile2 ...]"
    echo "Profiles: ${!PROFILES[@]}"
    exit 1
fi
shift

# Collect profiles from remaining args; skip non-profile tokens.
profiles=()
for arg in "$@"; do
    if [[ -n "${PROFILES[$arg]:-}" ]]; then
        profiles+=("$arg")
    else
        echo "[skip] unknown/non-profile arg: $arg"
    fi
done

# Default trio if no profiles provided.
if [ ${#profiles[@]} -eq 0 ]; then
    profiles=(l40s4 a100 h100)
fi

source activate nomad_train
cd /home/anw2067/visualnav-transformer/train

# Wall-clock limit; override with SBATCH_TIME=24:00:00 ./submit_torch.sh ...
SBATCH_TIME="${SBATCH_TIME:-36:00:00}"

# Race tag shared across siblings so first-to-start can scancel the rest.
RACE_TAG="nomad-$(date +%s)-$$"

submit_one() {
    local gpu_type=$1
    local num_gpus=$2
    local cpus=$3
    local mem_gb=$4

    sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --constraint=${gpu_type}
#SBATCH --time=${SBATCH_TIME}
#SBATCH --mem=${mem_gb}GB
#SBATCH --job-name=${RACE_TAG}
#SBATCH --output=/home/anw2067/slurm_logs/nomad-%j.out
#SBATCH --error=/home/anw2067/slurm_logs/nomad-%j.err
#SBATCH --account=torch_pr_230_tandon_advanced

scancel --state=PENDING --jobname=${RACE_TAG} -u \$USER

cd /home/anw2067/visualnav-transformer/train
singularity exec --nv \
  --overlay /scratch/anw2067/nymeria.sqf:ro \
  --overlay /scratch/anw2067/egodex_processed_v2_clean.sqf:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  bash -l -c "conda activate nomad_train && ./torch_run.sh ${CONFIG_FILE} ${num_gpus}"
EOF
}

echo "[submit] $CONFIG_FILE  profiles=${profiles[*]}  race_tag=$RACE_TAG"
for p in "${profiles[@]}"; do
    # shellcheck disable=SC2086
    submit_one ${PROFILES[$p]}
done
