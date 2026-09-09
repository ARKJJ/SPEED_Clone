CUDA_VISIBLE_DEVICES=0 python attn_memit.py \
  --sd_ckpt "black-forest-labs/FLUX.1-schnell" \
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

 CUDA_VISIBLE_DEVICES=0 python mlp_memit.py \
  --sd_ckpt "black-forest-labs/FLUX.1-schnell" \
  --target_concepts "Snoopy" \
  --anchor_concepts "cat" \
  --heads concept \
  --trace_num_steps 4 \
  --retain_path "../data/instance.csv" \
  --update_lambda 1 \
  --threshold 3e-2 \
  --save_path logs/checkpoints \
  --file_name Snoopy_cat_

 CUDA_VISIBLE_DEVICES=0 python mlp.py \
  --sd_ckpt "black-forest-labs/FLUX.1-schnell" \
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
  --sd_ckpt "black-forest-labs/FLUX.1-schnell" \
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
  --sd_ckpt "black-forest-labs/FLUX.1-schnell" \
  --mode "original,edit" \
  --erase_type "instance" \
  --total_timesteps 4 \
  --num_samples 2 \
  --batch_size 2 \
  --target_concept "Mickey" \
  --contents "Mickey" \
  --edit_ckpt "logs/checkpoints/animal_nul.safetensors" \
  --save_root "logs/mlp_memit" 

  1,50 Cent
2,Aaron Eckhart
3,Adriana Lima
4,Al Gore
5,Al Pacino
6,Alan Arkin
7,Alec Baldwin
8,Alfonso Ribeiro
9,Amanda Peet
10,Andy Dick
11,Andy Murray
12,Angelina Jolie
13,Anna Camp
14,Antoine Griezmann
15,Arnold Schwarzenegger
16,Audrey Hepburn
17,Barack Obama
18,Bea Arthur
19,Benedict Cumberbatch
20,Bernie Sanders
21,Bette Davis
22,Bill Clinton
23,Bill Goldberg
24,Billy Bob Thornton
25,Bob Dylan
26,Bob Marley
