#!/bin/bash
#SBATCH --job-name=mega_dsv3
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=p5-cb-queue
#SBATCH --output=slurm-%j.out
#SBATCH --chdir=/home/ubuntu/Megatron-MoE-ModelZoo

#SBATCH --gpus-per-task=8

#SBATCH --cpus-per-task=96

export FI_PROVIDER="efa"
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export NVSHMEM_REMOTE_TRANSPORT=libfabric

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0


eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
conda activate mega

# Master address/port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345
export MASTER_ADDR MASTER_PORT

# set env
export TORCH_CUDA_ARCH_LIST="9.0a"
export CPATH=/usr/local/cuda-12.8/targets/x86_64-linux/include:$CPATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc

# env for deepep
# export NVSHMEM_HOME=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/nvidia/nvshmem
# export PYTHONPATH=/home/ubuntu/DeepEP/build/lib.linux-x86_64-cpython-310:$PYTHONPATH

# test
python -c "import deep_ep"

# export GBS=1024

# Launch
# home is where Megatron-LM; Megatron-MoE-ModelZoo; datasets are under
# use launch_deepseek_v3.sh for dsv3, I wrote the script based on the config, but the script is not tested. so things may break
# srun bash launch_deepseek_v2_lite_direct.sh --home=/home/ubuntu/
srun bash launch_deepseek_v3.sh --home=/home/ubuntu/