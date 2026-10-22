"""WikiText-103 as GPT-2 tokens, the two partitions of the validation split, and perplexity."""
import math
from pathlib import Path

import numpy as np
import torch

BLOCK = 1024           # context length, and the window length of every perplexity
VAL_A_WINDOWS = 120    # partition A of the validation split: its first 120 windows; partition B: the rest
TOKENS = {"train": 119_721_489, "validation": 251_048, "test": 287_644}


def load(cache="data"):
    """The three splits as uint16 token arrays. Built on first use from wikitext-103-raw-v1: the lines of each split
    joined with "\\n\\n" and encoded with GPT-2's byte-pair tokenizer, without special tokens."""
    paths = {split: Path(cache) / f"wikitext103_{split}.npy" for split in TOKENS}
    if not all(p.exists() for p in paths.values()):
        import tiktoken
        from datasets import load_dataset
        encoder = tiktoken.get_encoding("gpt2")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        Path(cache).mkdir(parents=True, exist_ok=True)
        for split, path in paths.items():
            tokens = encoder.encode_ordinary("\n\n".join(dataset[split]["text"]))
            np.save(path, np.array(tokens, dtype=np.uint16))
    splits = {split: np.load(path, mmap_mode="r") for split, path in paths.items()}
    for split, n in TOKENS.items():
        if len(splits[split]) != n:
            raise RuntimeError(f"{split} has {len(splits[split])} tokens, expected {n}")
    return splits


def validation_partitions(validation):
    """(A, B): A holds the first VAL_A_WINDOWS windows, B the remaining ones."""
    return validation[:VAL_A_WINDOWS * BLOCK + 1], validation[VAL_A_WINDOWS * BLOCK:]


def batch(tokens, size, generator, device):
    """`size` windows at uniformly random positions (with replacement): inputs and next-token targets."""
    starts = torch.randint(0, len(tokens) - BLOCK - 1, (size,), generator=generator).tolist()
    windows = torch.from_numpy(np.stack([tokens[i:i + BLOCK + 1] for i in starts]).astype(np.int64)).to(device)
    return windows[:, :-1], windows[:, 1:]


@torch.no_grad()
def perplexity(model, tokens, device):
    """exp of the mean loss over consecutive non-overlapping windows of BLOCK tokens, each scored without earlier context;
    tokens after the last whole window are not scored."""
    model.eval()
    losses = []
    for i in range(0, len(tokens) - BLOCK, BLOCK):
        window = torch.from_numpy(tokens[i:i + BLOCK + 1].astype(np.int64)).to(device)
        losses.append(model(window[None, :-1], window[None, 1:]).item())
    mean = sum(losses) / len(losses)
    return math.exp(mean) if mean < 709 else math.inf
