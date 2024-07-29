#!/bin/bash
root_dir=$1
model_path=$2
#python tools.py "$root_dir" test
python eval.py --output_dir "$root_dir" --model_path "$model_path" --input_imgs "$root_dir"
#dir="$root_dir\Images_divide"
#rm -rf "$dir"
#python tools.py "$root_dir" compose
#dir="$root_dir\vis"
#rm -rf "$dir"
python postprocess.py "$root_dir"