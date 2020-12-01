#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=<haakosbo>
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --job-name="tflite_classification"
#SBATCH --output=tflite_test_run.out
#SBATCH --mail-user=<haakosbo@ntnu.no>
#SBATCH --mail-type=ALL
# WORKDIR=${TFlite-model-maker example}
# cd ${WORKDIR}
# echo "we are running from this directory: $SLURM_SUBMIT_DIR"
# echo " the name of the job is: $SLURM_JOB_NAME"
# echo "Th job ID is $SLURM_JOB_ID"
# echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
# echo "Number of nodes: $SLURM_JOB_NUM_NODES"
# echo "We are using $SLURM_CPUS_ON_NODE cores"
# echo "We are using $SLURM_CPUS_ON_NODE cores per node"
# echo "Total of $SLURM_NTASKS cores"
# module purge
# module load module_set_1
# mpirun python main.py