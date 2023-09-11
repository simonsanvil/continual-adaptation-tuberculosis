#!/bin/bash

image_dir=$1
output_dir=$2
resume=$3
epochs=$4
batch_size=$5
device=$6
tags=$7

cd /Users/simon/Documents/Projects/TFM/bacili_detection/detr &&  python main.py \
    --dataset_file bacilli_detection \
    --image_dir $image_dir \
    --output_dir $output_dir \
    --resume $resume \
    --logging_file "$output_dir/train.log" \
    --epochs $epochs \
    --num_classes 2 \
    --batch_size $batch_size \
    --device $device \
    --tags $tags
