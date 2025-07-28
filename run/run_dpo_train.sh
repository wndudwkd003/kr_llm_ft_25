# run script for SFT training

# export HF_HOME=~/.cache/huggingface
# export TRANSFORMERS_CACHE=~/.cache/huggingface

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_dpo --path "output/2025-07-24_23-05-37_kakaocorp_kanana-1.5-8b-instruct-2505_r_128_ra_256_rd_0_sft"
