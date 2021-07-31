#!/bin/bash
echo $0
full_path=$(realpath $0)
echo "fullpath: " $full_path
dir_path=$(dirname $full_path)
echo "dir_path: " $dir_path
examples=$(dirname $dir_path)
echo "examples: " $examples
data_dir="$examples/../saue bilder/Combined/IR/not_sau/with_blurry/removed_duplicate_STRICT/"
echo "DATA: $data_dir"
mkdir -p "${data_dir}/Train_set"
shuf