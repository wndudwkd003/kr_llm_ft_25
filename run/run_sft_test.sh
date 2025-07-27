# run script for SFT training

CUDA_VISIBLE_DEVICES=0 python -m scripts.test_sft --save_dir "output/2025-07-26_16-19-42_kakaocorp_kanana-1.5-8b-base_r_64_ra_64_rd_0.2_float16_sft"
CUDA_VISIBLE_DEVICES=0 python -m scripts.test_sft --save_dir "output/2025-07-27_02-40-47_kakaocorp_kanana-1.5-8b-base_r_64_ra_64_rd_0.2_float16_sft"
