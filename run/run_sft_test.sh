# run script for SFT training

CUDA_VISIBLE_DEVICES=3 python -m scripts.test_sft --save_dir "output/2025-07-28_03-16-11_kakaocorp_kanana-1.5-8b-instruct-2505_r_64_ra_64_rd_0.2_float16_sft"
CUDA_VISIBLE_DEVICES=3 python -m scripts.test_sft --save_dir "output/2025-07-28_04-54-17_kakaocorp_kanana-1.5-8b-instruct-2505_r_64_ra_64_rd_0.2_float16_sft"
