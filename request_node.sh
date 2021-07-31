#!/bin/bash

salloc --partition=GPUQ --account=share-ie-idi --time=08:00:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:V100:1 

#compute-