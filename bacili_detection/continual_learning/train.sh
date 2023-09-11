#!/bin/bash

dataset_file=$1
image_dir=$2
output_dir=$3
resume=$4
logging_file=$5
epochs=$6
num_classes=$7
batch_size=$8
device=$9
tags=${10}

cd /Users/simon/Documents/Projects/TFM/bacili_detection/detr &&  python main.py \
    --dataset_file $dataset_file \
    --image_dir $image_dir \
    --output_dir $output_dir \
    --resume $resume \
    --logging_file $logging_file \
    --epochs $epochs \
    --num_classes $num_classes \
    --batch_size $batch_size \
    --device $device \
    --tags $tags
