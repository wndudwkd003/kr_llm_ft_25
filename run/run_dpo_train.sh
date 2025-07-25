# run script for SFT training

# export HF_HOME=~/.cache/huggingface
# export TRANSFORMERS_CACHE=~/.cache/huggingface

CUDA_VISIBLE_DEVICES=0,1 python -m scripts.train_dpo --path "output/2025-07-25_16-27-29_kakaocorp_kanana-1.5-8b-instruct-2505_r_128_ra_256_rd_0.0_sft"
