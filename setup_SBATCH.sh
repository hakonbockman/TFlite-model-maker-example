#!/bin/bash

#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:V100:1

# Memory can be left to default
#SBATCH --job-name="tflite_classification"
#SBATCH --output=tflite_classification_SBATCH.out
#SBATCH --mail-user=haakosbo@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge 
module load GCC/8.3.0
module load CUDA/10.1.243
module load OpenMPI/3.1.4
module load cuDNN/7.6.4.38
module load Python/3.7.4
module list

python --version #(should be 3.7.4)   // could be that you need to load python module again..

pip --version #(should be the system pip)
source /lustre1/work/haakosbo/env/python374/bin/activate
pip --version #(should be virtual environment pip and not system pip)

pip install tflite-model-maker==0.2.3 --no-cache-dir

pip install tensorflow==2.3.0 --no-cache-dir       #//Tensorflow 2.3.0 is compatible with the cuDNN 7.6 and CUDA 10.1

echo "Now the Modules, Virtual Environment and Packages are loaded"
echo "Starting to setup for SBATCH"



python  /lustre1/work/haakosbo/TFlite-model-maker-example/main.py

uname -a