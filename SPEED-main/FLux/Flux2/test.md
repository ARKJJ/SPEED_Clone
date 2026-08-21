CUDA_VISIBLE_DEVICES=0 python attn_memit.py \
  --sd_ckpt "black-forest-labs/FLUX.2-klein-4B" \
  --device "cuda:0" \
  --target_concepts "Mickey" \
  --anchor_concepts "cat" \
  --update_lambda 0.1 \
  --retain_path "../data/instance.csv" \
  --heads "concept" \
  --save_path "logs/checkpoints" \
  --file_name "erase_Mickey" \
  --params QKV \
  --trace_num_steps 4 


 CUDA_VISIBLE_DEVICES=0 python mlp.py \
  --sd_ckpt "black-forest-labs/FLUX.2-klein-4B" \
  --target_concepts "Snoopy" \
  --anchor_concepts "cat" \
  --heads concept \
  --trace_num_steps 4 \
  --retain_path "../data/instance.csv" \
  --update_lambda 1 \
  --threshold 3e-2 \
  --save_path logs/checkpoints \
  --file_name Snoopy_cat_


  CUDA_VISIBLE_DEVICES=0 python attn.py \
  --sd_ckpt "black-forest-labs/FLUX.2-klein-4B" \
  --device "cuda:0" \
  --trace_num_steps 4 \
  --retain_path "../data/instance.csv" \
  --heads "concept" \
  --params QKV \
  --update_lambda 1 \
  --threshold 1e-4 \
  --target_concepts "Mickey" \
  --anchor_concepts "animal" \
  --save_path "logs/checkpoints" \
  --file_name "animal_nul" 


  CUDA_VISIBLE_DEVICES=0 python sample.py \
  --sd_ckpt "black-forest-labs/FLUX.2-klein-4B" \
  --mode "original,edit" \
  --erase_type "instance" \
  --total_timesteps 4 \
  --num_samples 2 \
  --batch_size 2 \
  --target_concept "Snoopy" \
  --contents "Snoopy" \
  --edit_ckpt "logs/checkpoints/Snoopy_cat_.safetensors" \
  --save_root "logs/mlp" 