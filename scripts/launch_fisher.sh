#!/usr/bin/bash
#SBATCH --job-name=fisher
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hahn2@cs.cmu.edu
#SBATCH --output=/projects/tir6/strubell/hahn2/mask/tir_logs/%x-%J-%a.out
#SBATCH --err=/projects/tir6/strubell/hahn2/mask/tir_logs/%x-%J-%a.err
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --exclude=tir-0-[7,9,13,15,17,19],tir-0-32

source ~/anaconda3/etc/profile.d/conda.sh

conda activate mask

export HF_HOME=/scratch/hahn2/.cache

cd /projects/tir6/strubell/hahn2/mask

./scripts/run_fisher.sh $1 $2
