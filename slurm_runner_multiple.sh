jobname=$1 #'space_invaders' #also game_name
seed=$2

# for seed in {1805..1805}; do # {1805..1808}; do
  temp="${jobname}_seed_${seed}"
  stdOut=log.${temp}.stdout
  stdErr=log.${temp}.stderr
  resFile=result.${temp}
  stdOut=log.${jobname}.${temp}.stdout
  stdErr=log.${jobname}.${temp}.stderr
  logRoot=/om/user/tejask/DeepSR/logs

  module add openmind/miniconda/3.18.3-python2
  module add cuda65/toolkit/6.5.14
  module add openmind/libjpeg-turbo/1.3.1
  export QUE='squeue -o "%.18i %.9P %.60j %.10b %.5C %.8u %.2t %.10M %.6D %R" | grep tejask'

  export PATH=$PATH:/om/user/tejask/mytorch/bin:/om/user/tejask/lmdb/libraries/liblmdb
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/om/user/tejask/mytorch/lib:/om/user/tejask/zmq/lib:/om/user/tejask/lmdb/libraries/liblmdb
  export SQ='squeue -o "%.18i %.9P %.50j %.10b %.5C %.8u %.2t %.10M %.6D %R"'
  export TORCH_DIR='/om/user/tejask/torch'
  # User specific aliases and functions
  module load slurm

  sbatch -o ${logRoot}/${stdOut}  -e ${logRoot}/${stdErr} --job-name=${temp}  slurm_runner.sh ${seed} ${jobname}
  sleep 2
# done
