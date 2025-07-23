from src.data.dataset import CustomDataset

class DpoDataset(CustomDataset):
    def __init__(self, fname, tokenizer):
        super().__init__(fname, tokenizer)
        self.IGNORE_INDEX = -100

    def make_chat(self, inp):
        # Call the parent method to create the chat structure
        chat = super().make_chat(inp)
        
        # Additional processing for DPO-specific requirements can be added here
        return chat