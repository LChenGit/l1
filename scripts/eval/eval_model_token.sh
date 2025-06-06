set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
# MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
MODEL_PATH="l3lab/L1-Qwen-1.5B-Exact"
NUM_TOKENS=1024  # Add default NUM_TOKENS
MAX_TOKENS=$((NUM_TOKENS * 2))  # Set MAX_TOKENS to twice NUM_TOKENS
DATATYPES=("gpqa" "mmlu_1000" "lsat" "aime2025" "math" "amc" "aime" "olympiad_bench")

n_gpus_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "n_gpus_per_node: ${n_gpus_per_node}"

# OUTPUT_DIR=$OUTPUT_DIR
# export OUTPUT_DIR="/lus/eagle/projects/argonne_tpc/chen/repo/myfork/l1/cache/output"
# Set default if not provided externally
: "${OUTPUT_DIR:=/lus/eagle/projects/argonne_tpc/chen/repo/myfork/l1/cache/output}"
: "${DATA_DIR:=/lus/eagle/projects/argonne_tpc/chen/repo/myfork/l1/cache/data}"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-tokens)  # Add new argument
            NUM_TOKENS="$2"
            MAX_TOKENS=$((NUM_TOKENS * 2))
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --num-tokens <num_tokens> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Number of Tokens: ${NUM_TOKENS}"
echo "Max Tokens: ${MAX_TOKENS}"

# MAX_TOKENS=8192

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    echo "Starting generation for ${DATA_TYPE} at $(date)"
    start_time=$(date +%s)
    mkdir -p "${OUTPUT_DIR}/output_${NUM_TOKENS}"
    
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${n_gpus_per_node} \
        data.path=$DATA_DIR/data_${NUM_TOKENS}/${DATA_TYPE}.parquet \
        data.output_path=$OUTPUT_DIR/output_${NUM_TOKENS}/${DATA_TYPE}.parquet \
        data.n_samples=16 \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=${MAX_TOKENS} \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Finished generation for ${DATA_TYPE} at $(date)"
    echo "Total generation time for ${DATA_TYPE}: ${duration} seconds"
    echo "----------------------------------------"
done

