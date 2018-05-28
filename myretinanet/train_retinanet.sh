#!/usr/bin/env zsh

python retinanet_main.py \
--resnet_checkpoint=/home/ubuntu/modelzoo/resnet_v2_50/resnet_v2_50.ckpt \
--training_file_pattern=/home/ubuntu/datadisk/coco/train-*.tfrecord \
--model_dir=/home/ubuntu/models_home
--mode=train


