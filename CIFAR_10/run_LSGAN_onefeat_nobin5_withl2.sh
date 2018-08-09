#!/bin/bash
#SBATCH -A rohit.gajawada
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=32
module add cuda/8.0
module add cudnn/7-cuda-8.0

python2 main_onefeat.py --losstype="lsgan" --advweight=1.0 | tee lsgan_onefeat.txt
