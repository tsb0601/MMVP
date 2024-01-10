#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /home/simonzhai/LLaVA1.5/checkpoints_afterfinetune/llava-dino_and_clip-mergeprojector-pretrain-vicuna-llava-v1.5\
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/simonzhai/LLaVA1.5/LLaVA/playground/data/eval/pope/images/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/ \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
