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
|-- scripts
`-- src
    |-- configs
    |   |-- config_manager.py
    |   |-- lora_config.py
    |   |-- model_config.py
    |   |-- sft_config.py
    |   |-- system_config.py
    |   `-- tokens
    |       `-- token.yaml
    |-- data
    |   |-- dataset.py
    |   |-- dpo_dataset.py
    |   `-- sft_dataset.py
    |-- models
    |-- rag
    |   |-- indexer.py
    |   |-- rag_dataset.py
    |   `-- retriever.py
    |-- test
    |-- train
    |   |-- dpo_trainer.py
    |   `-- sft_trainer.py
    `-- utils
        |-- adapter_utils.py
        `-- data_utils.py
```
