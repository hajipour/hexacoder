#!/bin/bash

# Define the Python file path
DIR_NAME="codegen-350m-8cwes-dataset_all_temp0.6"

python human_eval_gen.py --model_type lm --model_dir 350m --output_name $DIR_NAME --temp 0.6
python human_eval_exec.py --output_name $DIR_NAME
python print_results.py --eval_type human_eval --eval_dir ../experiments/human_eval/$DIR_NAME

DIR_NAME="codegen-350m-8cwes-dataset_all_temp0.8"

python human_eval_gen.py --model_type lm --model_dir 350m --output_name $DIR_NAME --temp 0.8
python human_eval_exec.py --output_name $DIR_NAME
python print_results.py --eval_type human_eval --eval_dir ../experiments/human_eval/$DIR_NAME
