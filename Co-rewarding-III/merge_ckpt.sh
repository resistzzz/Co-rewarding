#!/bin/bash



steps=(30 60 90 120 150 174)

model_path=DAPO_Dataset/XXXX
for step in "${steps[@]}"; do
  echo ">>> Merging checkpoint at step $step ..."
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/$model_path/global_step_${step}/actor \
    --target_dir ../../merged_hf_model/$model_path/global_step_${step}/actor
done

echo ">>> All merges completed!"



