from src.data.dataset import CustomDataset

class SFTDataset(CustomDataset):
    def __init__(self, fname, tokenizer):
        super().__init__(fname, tokenizer)

