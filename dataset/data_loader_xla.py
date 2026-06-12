import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
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


def get_dataloader_xla(model_id="google/gemma-3-1b-it", dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", chunk_size=4096, batch_size=1, world_size=1, rank=0):
    """
    Creates a DataLoader with DistributedSampler for XLA multi-core TPU training.
    
    Args:
        model_id: HuggingFace model ID for the tokenizer
        dataset_name: HuggingFace dataset name
        dataset_config: HuggingFace dataset config
        chunk_size: Total sequence length (past_len + future_len)
        batch_size: Per-core batch size
        world_size: Number of TPU cores (typically 8 for v5e-8)
        rank: Current core index (0-7)
    
    Returns:
        dataloader: A standard DataLoader (to be wrapped with MpDeviceLoader externally)
        sampler: The DistributedSampler (needed for .set_epoch() between epochs)
    """
    print(f"[Rank {rank}] Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"[Rank {rank}] Loading dataset {dataset_name} ({dataset_config})...")
    raw_dataset = load_dataset(dataset_name, dataset_config, split="train")
    
    # Concatenate all text
    print(f"[Rank {rank}] Tokenizing data...")
    full_text = "\n\n".join(raw_dataset["text"])
    
    # Tokenize everything into a single massive 1D tensor
    # Note: For massive datasets, this should be chunked/streamed. 
    # For warm-up on wikitext-2, this fits comfortably in RAM.
    tokens = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
    
    print(f"[Rank {rank}] Total tokens: {tokens.size(0)}")
    
    dataset = LongContextDataset(tokens, chunk_size)
    
    # DistributedSampler partitions data across TPU cores
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True  # Critical: ensures static batch shapes for XLA
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        drop_last=True,  # Critical: ensures static batch shapes for XLA
        num_workers=0     # XLA prefers 0 workers to avoid host contention
    )
    
    print(f"[Rank {rank}] Created Dataloader with {len(dataloader)} batches (per-core) of size {batch_size}x{chunk_size}")
    return dataloader, sampler
