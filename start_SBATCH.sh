#!/bin/bash

chmod u+x setup_SBATCH.sh
sbatch setup_SBATCH.sh

tail -f tflite_classification_SBATCH.out