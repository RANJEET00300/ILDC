import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class LongContextDataset(Dataset):
    def __init__(self, tokenized_data, chunk_size):
        """
        tokenized_data: 1D Tensor of token IDs
        chunk_size: total length of context (e.g., past_len + future_len)
        """
        self.chunk_size = chunk_size
        
        # Drop the remainder to perfectly chunk
        total_length = len(tokenized_data)
        num_chunks = total_length // chunk_size
        
        self.data = tokenized_data[:num_chunks * chunk_size].view(num_chunks, chunk_size)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(model_id="google/gemma-3-1b-it", dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", chunk_size=4096, batch_size=1):
    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading dataset {dataset_name} ({dataset_config})...")
    raw_dataset = load_dataset(dataset_name, dataset_config, split="train")
    
    # Concatenate all text
    print("Tokenizing data...")
    full_text = "\n\n".join(raw_dataset["text"])
    
    # We tokenize everything into a single massive 1D tensor
    # Note: For massive datasets, this should be chunked/streamed. 
    # For warm-up on wikitext-2, this fits comfortably in RAM.
    tokens = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
    
    print(f"Total tokens: {tokens.size(0)}")
    
    dataset = LongContextDataset(tokens, chunk_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"Created Dataloader with {len(dataloader)} batches of size {batch_size}x{chunk_size}")
    return dataloader
