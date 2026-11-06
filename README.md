# Tiny Shakespeare Transformer: A Decoder-Only Language Model Built from Scratch in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cynthiazhang98/COMSW4995-cynthiazhang98-A3/blob/main/4995A3.ipynb)

A small GPT-style Transformer language model implemented from scratch in PyTorch and trained on Shakespeare's plays with a custom-trained Byte-Pair Encoding (BPE) tokenizer. The project covers the full pipeline: data loading, tokenization, model implementation, training, evaluation, and attention visualization.

Assignment 3 for COMS W4995.

## The problem

A language model learns to predict the next token given the tokens before it. This project builds that from the ground up: every component of the Transformer (positional encoding, normalization, causal self-attention, feed-forward blocks) is written by hand instead of imported from a library, then trained to predict the next token in Shakespeare's text.

- **Task:** next-token prediction over a context window of 50 tokens
- **Metric:** cross-entropy loss and perplexity on held-out text (lower is better)
- **Questions:** can a ~0.5M-parameter model learn useful structure from ~1M characters of text, and what do its attention heads look at?

## Dataset

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), from Andrej Karpathy's char-rnn repository. The notebook downloads it automatically.

| | |
|---|---|
| Size | 1,115,394 characters (all ASCII) |
| Split | 80% train / 20% validation, a contiguous cut of the text (892,315 / 223,079 characters) |
| Tokenizer | BPE via Hugging Face `tokenizers`, whitespace pre-tokenizer, vocab size 500 |
| Tokens | 356,261 train / 92,599 validation |
| Training windows | Sliding windows of 50 tokens (5,566 train / 1,447 validation batches of 64) |

## Approach

### 1. Tokenization

A BPE tokenizer is trained on the **training split only**, so no validation text leaks into the vocabulary. Common words become single tokens and rarer words are split into subword pieces (for example, `Citizen` becomes `C`, `iti`, `z`, `en`). The notebook plots the 30 most frequent tokens.

### 2. Dataset

Each training example is a window of 50 consecutive tokens as the input, with the target being the same window shifted forward by one token, so the model predicts the next token at every position.

### 3. Model

`TinyTransformer` (524,660 parameters), with every component implemented from scratch:

| Component | Details |
|---|---|
| Token embedding | `nn.Embedding(500, 128)` |
| Positional encoding | Fixed sinusoidal |
| Transformer blocks | 2 layers, pre-norm residual design |
| Attention | Custom multi-head causal self-attention (4 heads, head dim 32) with an upper-triangular mask |
| Normalization | Custom **RMSNorm** (learnable scale, no mean-centering) |
| Feed-forward | Linear(128→512) → GELU → Linear(512→128) |
| Output | Final RMSNorm + linear LM head over the vocabulary |

The model also returns every layer's attention weights so they can be visualized.

### 4. Training

AdamW (learning rate 3e-4), cross-entropy loss, batch size 64 with a shuffled training set, 10 epochs, and gradient clipping at max norm 1.0. Trained on a Google Colab T4 GPU. Seeds are fixed (`torch.manual_seed(42)`, `np.random.seed(42)`).

## Results

| Epoch | Train loss | Val loss | Val perplexity |
|---|---|---|---|
| 1 | 3.4930 | 4.0104 | **55.17** |
| 5 | 2.2166 | 5.1311 | 169.21 |
| 10 | 1.8886 | 6.1318 | 460.27 |

### What the results show

- **The model overfits immediately.** Training loss falls steadily every epoch, but validation loss rises from the first epoch on. Validation perplexity is best after epoch 1 (55.17) and worsens to 460.27 by epoch 10. For context, guessing uniformly over a 500-token vocabulary gives a perplexity of 500, so the epoch-10 model is barely better than chance on unseen text.
- **The split makes this harder.** The validation set is the last 20% of the file, which contains different plays and characters than the training set, so some of the gap is distribution shift as well as memorization.
- **Likely fixes:** dropout, weight decay, early stopping (which would stop at epoch 1 here), a split by random chunks instead of one contiguous cut, a learning-rate schedule, and a larger or more varied vocabulary.
- **Perplexity is per BPE token,** so it is not comparable to character-level or word-level perplexities from other work.

### Attention visualizations

The notebook plots attention heatmaps for a 50-token passage ("...we know't. First Citizen: Let us kill him, and we'll have corn at our own price..."). All show the lower-triangular pattern forced by the causal mask, so each token can only attend to itself and earlier tokens.

Saved figures:

- `token_freq.png`: top 30 most frequent BPE tokens
- `input_target_table.png`: example input/target pair
- `loss_curves.png`: train vs. validation loss
- `perplexity_curve.png`: validation perplexity by epoch
- `attn_layer1_head1.png`, `attn_layer1_head2.png`, `attn_layer2_head1.png`: individual heatmaps
- `attn_multihead.png`: heads 1 and 2 of both layers side by side on the same input

## Key implementation notes

- **Causal masking** uses `torch.triu` and `-inf` fill before the softmax, so tokens cannot attend to the future.
- **RMSNorm** divides by the root-mean-square of the activations, a cheaper alternative to LayerNorm used in many modern LLMs.
- **Pre-norm** residual blocks (`x + attn(norm(x))`, `x + ff(norm(x))`) for stable training.
- **No leakage:** the tokenizer vocabulary is learned from the training split only.

## Limitations

- **Overfitting.** No regularization (dropout, weight decay) and no early stopping are used, so the final model is much worse on validation text than the epoch-1 model.
- **Whitespace is dropped.** The whitespace pre-tokenizer discards spaces and newlines, so decoded text has no line structure and the model could not reproduce the original formatting if used to generate text.
- **Small scale.** The model has about 0.5M parameters, a 500-token vocabulary, and a single train/validation split with no repeated runs.
- **Only heads 1 and 2 of each layer are plotted.** The model has four heads per layer, but the multi-head figure shows the first two of each.

## Running it

1. Open `4995A3.ipynb` in Google Colab (the badge at the top links to it).
2. Set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Run all cells from top to bottom. The dataset downloads automatically.

Requirements: `torch`, `tokenizers`, `numpy`, `matplotlib`, `seaborn`, `requests`. Colab provides all of them except `tokenizers`, which the first cell installs (`pip install tokenizers`).

## Repository contents

```
4995A3.ipynb   # full pipeline: data, tokenizer, model, training, plots
README.md
```

## Acknowledgments

- Dataset from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) repository.
- Architecture based on "Attention Is All You Need" (Vaswani et al., 2017) and RMSNorm (Zhang & Sennrich, 2019).
