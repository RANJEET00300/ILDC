import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def get_wikipedia_batches(tokenizer_path, batch_size=4, required_length=2561, max_batches=10):
    """
    Streams Wikipedia, filters for long articles, and yields uniform batches.
    """
    print("⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print("🌐 Streaming Wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    current_batch_input_ids = []
    current_batch_attention = []
    batches_yielded = 0
    articles_scanned = 0

    for example in ds:
        text = example["text"]
        articles_scanned += 1

        # ⚡ FAST FILTERING: 2561 tokens is roughly 10,000+ characters.
        if len(text) < 10000:
            continue

        # Tokenize and truncate exactly to our requirement (2561)
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=required_length,
            return_tensors="pt",  # returns PyTorch tensors
        )

        input_ids = inputs["input_ids"][0]

        # If the article actually met the requirement (has exactly 2561 tokens)
        if input_ids.size(0) == required_length:
            current_batch_input_ids.append(input_ids)
            current_batch_attention.append(inputs["attention_mask"][0])

            # When we have enough articles to form a batch
            if len(current_batch_input_ids) == batch_size:
                # Stack them into a unified tensor of shape [batch_size, 2561]
                batch = {
                    "input_ids": torch.stack(current_batch_input_ids),
                    "attention_mask": torch.stack(current_batch_attention),
                }

                yield batch

                # Reset for the next batch
                current_batch_input_ids = []
                current_batch_attention = []
                batches_yielded += 1

                if batches_yielded >= max_batches:
                    break
