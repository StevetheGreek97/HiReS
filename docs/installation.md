# Installation

## Requirements

- Python ≥ 3.10
- pip

## Install from PyPI

```bash
pip install HiReSeg
```

The distribution name is `HiReSeg`; the import name is `hires`.

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

Core dependencies (including `opencv-python`) are installed automatically. The
trait-analysis functions in `hires.analysis` additionally need `plotnine`:

| Package | Use |
|---------|-----|
| `plotnine` | Trait-comparison plots (`distributions`, `bias`, `bland_altman`, …) |
| `torch` | GPU inference (`--device cuda:0`) — install via the PyTorch site |

```bash
pip install plotnine
```

## Verify installation

```bash
hires --help
```

You should see the list of available commands.
