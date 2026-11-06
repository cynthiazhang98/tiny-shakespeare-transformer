# Tiny Shakespeare Transformer: A Decoder-Only Language Model Built from Scratch in PyTorch

A small GPT-style, decoder-only Transformer language model trained from scratch on the Tiny Shakespeare dataset using a custom-trained Byte-Pair Encoding (BPE) tokenizer. This project (COMS W4995, Assignment 3) walks through the full pipeline: data loading, tokenization, model implementation, training, evaluation, and attention visualization.

## Overview

- **Dataset:** [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1.1M characters, all ASCII)
- **Split:** 80% train / 20% validation (892,315 / 223,079 characters)
- **Tokenizer:** BPE via Hugging Face `tokenizers`, whitespace pre-tokenizer, vocab size 500
  - 356,261 training tokens, 92,599 validation tokens
- **Task:** Next-token prediction with a context window of 50 tokens
- **Hardware:** Google Colab, T4 GPU

## Model Architecture

`TinyTransformer` (~524K parameters), implemented entirely from scratch:

| Component | Details |
|---|---|
| Token embedding | `nn.Embedding(500, 128)` |
| Positional encoding | Fixed sinusoidal |
| Transformer blocks | 2 layers, pre-norm residual design |
| Attention | Custom multi-head causal self-attention (4 heads, head dim 32) with upper-triangular mask |
| Normalization | Custom **RMSNorm** (learnable scale, no mean-centering) |
| Feed-forward | Linear(128→512) → GELU → Linear(512→128) |
| Output | Final RMSNorm + linear LM head to vocab |

The model returns attention weights from every layer so they can be visualized.

## Training Setup

- Optimizer: AdamW, learning rate 3e-4
- Loss: Cross-entropy
- Batch size: 64, shuffled training set
- Epochs: 10
- Gradient clipping: max norm 1.0
- Metrics: train/validation loss and validation perplexity

## Results

| Epoch | Train Loss | Val Loss | Val PPL |
|---|---|---|---|
| 1 | 3.4930 | 4.0104 | 55.17 |
| 5 | 2.2166 | 5.1311 | 169.21 |
| 10 | 1.8886 | 6.1318 | 460.27 |

Training loss falls steadily, but validation loss rises from the very first epoch, so validation perplexity is best at epoch 1 (55.17) and worsens to 460.27 by epoch 10. This is a clear sign of **overfitting** (or a distribution mismatch). Because the split is a contiguous 80/20 cut of the text, the validation set contains different plays and characters than the training set, which makes generalization harder. Likely improvements include dropout, weight decay, early stopping, a random/shuffled-by-chunk split, a larger/more varied vocabulary, and a learning-rate schedule.

## Visualizations

The notebook produces and saves:

- `token_freq.png`: top 30 most frequent BPE tokens
- `input_target_table.png`: example input/target sequence pair
- `loss_curves.png`: train vs. validation loss
- `perplexity_curve.png`: validation perplexity over epochs
- `attn_layer1_head1.png`, `attn_layer1_head2.png`, `attn_layer2_head1.png`: individual attention heatmaps
- `attn_multihead.png`: all four attention heads (2 layers × 2 shown) on the same input sequence

The attention heatmaps show the expected lower-triangular pattern imposed by the causal mask, letting you inspect what each head attends to for a sample passage.

## Project Structure

```
.
├── 4995A3.ipynb     # Full notebook: data, tokenizer, model, training, plots
└── README.md
```

## How to Run

1. Open `4995A3.ipynb` in Google Colab (the badge in the notebook links directly to it).
2. Set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Run all cells top to bottom. The dataset is downloaded automatically.

### Dependencies

```
torch
tokenizers
numpy
matplotlib
seaborn
requests
```

Install the one extra package with:

```bash
pip install tokenizers
```

## Key Implementation Notes

- **Causal masking** is applied with `torch.triu` and `-inf` filling before the softmax so tokens cannot attend to the future.
- **RMSNorm** normalizes by the root-mean-square of activations, a cheaper alternative to LayerNorm used in modern LLMs.
- **Pre-norm** residual blocks (`x + attn(norm(x))`, `x + ff(norm(x))`) for stable training.
- The tokenizer is trained **only on the training split** to avoid leaking validation data into the vocabulary.
- Seeds are fixed (`torch.manual_seed(42)`, `np.random.seed(42)`) for reproducibility.

## Acknowledgments

- Dataset from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) repository.
- Architecture inspired by "Attention Is All You Need" (Vaswani et al., 2017) and RMSNorm (Zhang & Sennrich, 2019).
