#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 50000
#SBATCH --gres=gpu:1
#SBATCH -t 4-23:00
#SBATCH --mail-user=tejask@mit.edu
cd /om/user/tejask/DeepSR
# ./pagekite.py $1 $(( $1 - 1804 )).mrkulk.pagekite.me &
# th -ldisplay.start $1 0.0.0.0 &
./run_gpu MovingGoalsMedium $1 dsr 1 5000 500 1 256
