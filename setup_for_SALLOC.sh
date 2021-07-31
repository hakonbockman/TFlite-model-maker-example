#!/bin/bash


module purge 

module load GCC/10.2.0 #GCC/8.3.0 
module load CUDA/11.1.1-GCC-10.2.0 #CUDA/10.1.243 
module load OpenMPI/4.0.5-GCC-10.2.0 #OpenMPI/3.1.4 
module load cuDNN/8.0.4.30-CUDA-11.1.1 #cuDNN/7.6.4.38 
module load Python/3.8.6-GCCcore-10.2.0 #Python/3.7.4 

#module load GCC/10.2.0
#module load CUDA/11.1.1-GCC-10.2.0
#module load OpenMPI/4.0.5-GCC-10.2.0 
#module load cuDNN/8.0.4.30-CUDA-11.1.1
#module load Python/3.8.6-GCCcore-10.2.0

module list


python --version #(should be 3.8.6)   #// could be that you need to load python module again..


pip --version #(should be the system pip)
source /cluster/work/haakosbo/env/python386/bin/activate
pip --version #(should be virtual environment pip and not system pip)


pip install -q tflite-model-maker #==0.2.3 --no-cache-dir

pip install -q tensorflow-gpu #==2.3.0 --no-cache-dir       #//Tensorflow 2.3.0 is compatible with the cuDNN 7.6 and CUDA 10.1

echo "Now the Modules, Virtual Environment and Packages are loaded in"

mpiexec python /cluster/work/haakosbo/TFlite-model-maker-example/main.py
