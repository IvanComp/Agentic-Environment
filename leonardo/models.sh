model_gpus() {
    case "$1" in
        glm-4.7|qwen3:235b) echo 4 ;;
        gpt-oss:120b) echo 2 ;;
        *) echo 1 ;;
    esac
}

model_resources() {
    local gpus
    gpus="$(model_gpus "$1")"
    if (( gpus > 1 )); then
        echo "--gres=gpu:${gpus} --cpus-per-task=$((gpus * 8)) --mem=$((gpus * 120))G"
    fi
}
