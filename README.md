# Oxidized Transformers

Oxidized Transformers is a Rust transformers library that started out as
a port of [Curated Transformers](https://github.com/explosion/curated-transformers/tree/main).
The foundations are in place and some popular models are implemented, but
Oxidized Transformers is still too volatile to use in projects. Keep an eye
on the repo, since progress is currently fast.

## 🧰 Supported Model Architectures

Supported encoder-only models:

- ALBERT
- BERT
- RoBERTa
- XLM-RoBERTa

Supported decoder-only models:

- GPT-NeoX
- Llama 1/2

All types of models can be loaded from Huggingface Hub. Float16/bfloat16
models can use flash attention v2 on recent CUDA GPUs.