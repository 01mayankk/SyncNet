# Data Directory — SyncNet

This directory stores raw datasets, processed feature tables, and serialized graph data.

## Subdirectories
- `raw/`: Raw downloaded social media datasets (e.g. TwiBot-22, Cresci-2017). **Git ignored.**
- `processed/`: Cleaned text data, tokenized inputs, and normalized tabular features. **Git ignored.**
- `features/`: Extracted Transformer embeddings, behavioral feature matrices, and PyTorch Geometric graph datasets. **Git ignored.**

## Dataset Storage & Memory Policy
- Raw datasets must not be committed to Git.
- Large data processing must operate within the **~16 GB system RAM project budget**. Use chunking or reproducible sampling when loading large datasets.
