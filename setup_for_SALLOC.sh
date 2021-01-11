#!/bin/bash

module purge 

module load GCC/8.3.0
module load CUDA/10.1.243
module load cuDNN/7.6.4.38
module load Python/3.7.4

module list


python --version #(should be 3.7.4)   // could be that you need to load python module again..


pip --version #(should be the system pip)
source /lustre1/work/haakosbo/env/python374/bin/activate
pip --version #(should be virtual environment pip and not system pip)


pip install tflite-model-maker==0.2.3 --no-cache-dir

pip install tensorflow==2.3.0 --no-cache-dir       #//Tensorflow 2.3.0 is compatible with the cuDNN 7.6 and CUDA 10.1

echo "Now the Modules, Virtual Environment and Packages are loaded in"

python main.py
