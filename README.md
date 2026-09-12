<h1 align="center">Streaming Flow Policy</h1>
<h3 align="center">Simplifying diffusion/flow-matching policies by treating<br> <i>action trajectories as flow trajectories</i></h3>
<h4 align="center"><a href="https://streaming-flow-policy.github.io/">🌐 Website</a>  &nbsp;•&nbsp;  <a href=https://arxiv.org/abs/2505.21851>📄 Paper</a> &nbsp;•&nbsp; <a href="https://youtu.be/gqUnEzBCbZE">🎬 Talk</a> &nbsp;•&nbsp; <a href=https://x.com/siddancha/status/1925170490856833180>🐦 Twitter</a> &nbsp;•&nbsp; <a href=https://siddancha.github.io/streaming-flow-policy/notebooks>📚 Notebooks</a></h4>
<div align="center" style="margin: 0px; padding: 0px">
      <img style="width: 90%; min-width: 500px" src="https://github.com/user-attachments/assets/2b7a02c5-585e-40d4-9c5c-95a1948aa9d0"></img>
      <img style="width: 70%; min-width: 500px" src="https://github.com/user-attachments/assets/591fc294-b822-4d2c-9e41-2660b39cc863"></img>
</div>

## Two-group preference steering toy

This fork includes a frozen-SFPS experiment that fits direction (upper/lower)
and width (wide/narrow) preferences from scoped comparisons, then selects latent
candidates during inference. Start with the [method, equations, and code map](docs/research/2026-09-12-preference-steering-method.md),
[actual inference results and GIFs](docs/research/2026-09-12-mode-aligned-steering-results.md),
and [artifact locations and reproducibility manifest](docs/research/2026-09-12-preference-steering-artifacts.md).
The exploratory pilot controls all four centered modes; Gaussian wide retains
misses. See [env/README.md](env/README.md) for the toy commands.

## 🛠️ Installation

1. Create a virtual environment
    ```bash
    python3 -m venv .venv --prompt=streaming-flow-policy
    source .venv/bin/activate
    ```

### Via pip

2. pip-install this repository.
    ```bash
    pip install -e .
    ```

### Via uv (recommended for development)

2. Install [uv](https://docs.astral.sh/uv/).
    ```bash
    pip install uv
    ```

3. Sync Python dependencies using uv:
    ```bash
    uv sync
    ```


## 📚 Building Jupyter Book

The Jupyter Book is built using [jupyter-book](https://jupyterbook.org/intro.html). It lives in the `docs/` directory.

#### Command to clean the build directory.
```bash
jupyter-book clean docs
```

#### Command to build the book.
```bash
jupyter-book build docs
```

#### To add a notebook to the Jupyter book

Add a symlink to the `docs` directory.

#### View Jupyter book locally

The HTML content is created in the `docs/_build/html` directory.
