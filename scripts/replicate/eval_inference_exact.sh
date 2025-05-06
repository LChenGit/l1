# MODEL_PATH=$1

# ./scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 512 --datasets aime2025 math amc olympiad_bench
# ./scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 1024 --datasets aime2025 math amc olympiad_bench
# ./scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 2048 --datasets aime2025 math amc olympiad_bench
# ./scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 3600 --datasets aime2025 math amc olympiad_bench

# ./scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 512 --datasets aime gpqa mmlu_1000 lsat
# ./scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 1024 --datasets aime gpqa mmlu_1000 lsat
# ./scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 2048 --datasets aime gpqa mmlu_1000 lsat
# ./scripts/eval/eval_model_token.sh --model $MODEL_PATH --num-tokens 3600 --datasets aime gpqa mmlu_1000 lsat



# ./scripts/eval/eval_model_token.sh --num-tokens 512 --datasets aime2025 math amc olympiad_bench
# ./scripts/eval/eval_model_token.sh --num-tokens 1024 --datasets aime2025 math amc olympiad_bench
# ./scripts/eval/eval_model_token.sh --num-tokens 2048 --datasets aime2025 math amc olympiad_bench
# ./scripts/eval/eval_model_token.sh --num-tokens 3600 --datasets aime2025 math amc olympiad_bench

./scripts/eval/eval_model_token.sh --num-tokens 512 --datasets aime2025 gpqa lsat mmlu_1000 2>&1 | tee 512_log.txt
./scripts/eval/eval_model_token.sh --num-tokens 1024 --datasets aime2025 gpqa lsat mmlu_1000 2>&1 | tee 1024_log.txt
./scripts/eval/eval_model_token.sh --num-tokens 2048 --datasets aime2025 gpqa lsat mmlu_1000 2>&1 | tee 2048_log.txt
./scripts/eval/eval_model_token.sh --num-tokens 3600 --datasets aime2025 gpqa lsat mmlu_1000 2>&1 | tee 3600_log.txt