### 폴더 구조

```
.
|-- configs
|   |-- dpo_config.yaml
|   |-- lora_config.yaml
|   |-- model_config.yaml
|   |-- sft_config.yaml
|   `-- system_config.yaml
|-- data
|   |-- rag
|   |   |-- corpus.pdf
|   |   `-- index
|   `-- raw
|       |-- dev.json
|       |-- test.json
|       `-- train.json
|-- output
|   `-- 2025-07-24_07-40-50_kakaocorp_kanana-1.5-8b-base_r_128_ra_256_rd_0_sft
|       |-- README.md
|       |-- checkpoint-1200
|       |   |-- README.md
|       |   |-- adapter_config.json
|       |   |-- adapter_model.safetensors
|       |   |-- chat_template.jinja
|       |   |-- optimizer.pt
|       |   |-- rng_state.pth
|       |   |-- scaler.pt
|       |   |-- scheduler.pt
|       |   |-- special_tokens_map.json
|       |   |-- tokenizer.json
|       |   |-- tokenizer_config.json
|       |   |-- trainer_state.json
|       |   `-- training_args.bin
|       |-- configs
|       |   |-- lora_config.yaml
|       |   |-- model_config.yaml
|       |   |-- sft_config.yaml
|       |   `-- system_config.yaml
|       |-- logs
|       |   `-- events.out.tfevents.1753342868.c0e82bb3dc9c.2233602.0
|       |-- lora_adapter
|       |   |-- README.md
|       |   |-- adapter_config.json
|       |   `-- adapter_model.safetensors
|       `-- metrics.json
|-- readme.md
|-- run
|   `-- run_sft_train.sh
|-- scripts
|   |-- __pycache__
|   |   |-- train_sft.cpython-311.pyc
|   |   `-- train_sft.cpython-312.pyc
|   |-- train_dpo.py
|   `-- train_sft.py
|-- src
|   |-- configs
|   |   |-- __pycache__
|   |   |   |-- config_manager.cpython-312.pyc
|   |   |   |-- lora_config.cpython-312.pyc
|   |   |   |-- model_config.cpython-312.pyc
|   |   |   |-- sft_config.cpython-312.pyc
|   |   |   `-- system_config.cpython-312.pyc
|   |   |-- config_manager.py
|   |   |-- lora_config.py
|   |   |-- model_config.py
|   |   |-- sft_config.py
|   |   |-- system_config.py
|   |   `-- tokens
|   |       `-- token.yaml
|   |-- data
|   |   |-- __pycache__
|   |   |   |-- dataset.cpython-312.pyc
|   |   |   `-- sft_dataset.cpython-312.pyc
|   |   |-- dataset.py
|   |   |-- dpo_dataset.py
|   |   `-- sft_dataset.py
|   |-- models
|   |-- rag
|   |   |-- indexer.py
|   |   |-- rag_dataset.py
|   |   `-- retriever.py
|   |-- test
|   |-- train
|   |   |-- __pycache__
|   |   |   `-- sft_trainer.cpython-312.pyc
|   |   |-- dpo_trainer.py
|   |   `-- sft_trainer.py
|   `-- utils
|       |-- __pycache__
|       |   |-- data_utils.cpython-312.pyc
|       |   `-- path_utils.cpython-312.pyc
|       |-- adapter_utils.py
|       |-- data_utils.py
|       `-- path_utils.py
```
