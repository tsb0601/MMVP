#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path /home/simonzhai/LLaVA1.5/checkpoints_afterfinetune/llava-dino_and_clip-mergeprojector-pretrain-vicuna-llava-v1.5 \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

OPENAI_API_KEY="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"  python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b-eval-2.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl
