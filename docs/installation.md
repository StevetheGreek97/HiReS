# Installation

## Requirements

- Python ≥ 3.10
- pip

## Install from PyPI

```bash
pip install HiReSeg
```

The import name is `hires` (or `HiReS` — both work).

## Install from source

```bash
git clone https://github.com/StevetheGreek97/HiReS.git
cd HiReS
pip install -e .
```

## GPU support

Install PyTorch separately from the [official PyTorch site](https://pytorch.org/get-started/locally/) before using `--device cuda:0` or `--device mps`.

```bash
# Example: CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Optional dependencies

| Package | Use |
|---------|-----|
| `opencv-python` | Chunking and overlay rendering |
| `streamlit` | Web UI |

```bash
pip install opencv-python streamlit
```

## Verify installation

```bash
hires --help
```

You should see the list of available commands.
