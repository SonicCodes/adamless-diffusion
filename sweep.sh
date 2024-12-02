#!/bin/bash

# Adam learning rates
ADAM_LRS=("8e-4" "5e-4" "1e-4" "5e-5")
PSGD_LRS=("1e-3" "5e-4" "1e-4" "5e-5")
PSGD_MOMENTUMS=("0.9" "0.95" "0.99")
# Muon learning rates
MUON_LRS=("0.01" "0.02" "0.007" "0.005")
# Beta options
BETAS=("0.9/0.95" "0.95/0.99" "0.95/0.95")
# Batch sizes
BATCH_SIZES=("512" "1024")

# Trap Ctrl+C and kill all background processes
trap 'echo "Ctrl+C pressed. Killing all experiments..."; kill $(jobs -p) 2>/dev/null; exit 1' SIGINT

# Create array to store all commands
declare -a commands

for lr in "${PSGD_LRS[@]}"; do
    for beta in "${PSGD_MOMENTUMS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            commands+=("--optim psgd --psgd_lr ${lr} --momentum ${beta} --batch_size ${bs}")
        done
    done
done

# Add Adam experiments
for lr in "${ADAM_LRS[@]}"; do
    for beta in "${BETAS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            commands+=("--optim adam --adam_lr ${lr} --beta_option ${beta} --batch_size ${bs}")
        done
    done
done

# Add soap
for lr in "${ADAM_LRS[@]}"; do
    for beta in "${BETAS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            commands+=("--optim soap --adam_lr ${lr} --beta_option ${beta} --batch_size ${bs}")
        done
    done
done

# # Add Muon experiments
for mlr in "${MUON_LRS[@]}"; do
    for alr in "${ADAM_LRS[@]}"; do
        for beta in "${BETAS[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
                commands+=("--optim muon --muon_lr ${mlr} --adam_lr ${alr} --beta_option ${beta} --batch_size ${bs}")
            done
        done
    done
done

total=${#commands[@]}
completed=0

# Function to update progress
update_progress() {
    completed=$((completed + 1))
    printf "\rProgress: %d/%d experiments completed" $completed $total
}

# Function to run experiments on a GPU
run_gpu_queue() {
    local gpu=$1
    local start=$2
    local step=$3
    
    for i in $(seq $start $step $((total-1))); do
        if [ $i -lt $total ]; then
            CUDA_VISIBLE_DEVICES=$gpu python rf.py --cifar ${commands[$i]} > /dev/null 2>&1
            update_progress
        fi
    done
}

number_of_gpus=8
jobs_per_gpu=2

echo "Starting $total experiments across $number_of_gpus GPUs ($jobs_per_gpu jobs per GPU)"
echo "Progress: 0/$total experiments completed"

 
total_jobs=$((number_of_gpus * jobs_per_gpu))

# Launch jobs across GPUs
for gpu in $(seq 0 $((number_of_gpus-1))); do
    for job in $(seq 0 $((jobs_per_gpu-1))); do
        start=$((gpu * jobs_per_gpu + job))
        run_gpu_queue $gpu $start $total_jobs &
    done
done

# Wait for all queues to complete
wait

echo -e "\nAll experiments completed!"