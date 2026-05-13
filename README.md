# Skip-Connected VAE Representations for Visual Deep RL

This project studies whether decoder skip connections in a Variational Autoencoder improve visual representations for Deep Reinforcement Learning.

Core question:

> Do skip connections improve the latent state representation used by an RL agent, or do they mainly improve image reconstruction?

The project is built around MiniGrid visual observations, pretrained VAE encoders, and PPO baselines.

## Project Structure

```text
configs/                 Experiment configs
src/skip_vae_rl/          Training, models, env wrappers, visualization
notebooks/                Optional analysis notebooks
reports/figures/          Generated plots and reconstruction grids
runs/                     Checkpoints and logs
```

## Main Experiments

Recommended comparison:

```text
1. PPO from raw pixels
2. PPO with a frozen standard VAE encoder
3. PPO with a frozen Skip-VAE encoder
4. PPO with a fine-tuned Skip-VAE encoder
```

Extra ablations:

```text
latent_dim = 16, 32, 64
beta = 0.1, 1.0, 4.0
encoder = frozen vs fine-tuned
```

## Kaggle / Colab Setup

Use this inside a notebook cell:

```bash
!git clone <YOUR_REPO_URL> skip-vae-rl
%cd skip-vae-rl
!pip install uv
!uv sync --extra notebook
```

If `uv` gives trouble in the notebook runtime, use:

```bash
!pip install -e ".[notebook]"
```

## Quick Debug Run

These commands are intentionally small. They prove the pipeline works before running expensive jobs.

```bash
uv run python -m skip_vae_rl.train_vae --config configs/debug_vae.yaml
uv run python -m skip_vae_rl.train_rl --config configs/debug_rl_raw.yaml
uv run python -m skip_vae_rl.train_rl --config configs/debug_rl_skip_vae.yaml
```

## Full Training

Train a standard VAE:

```bash
uv run python -m skip_vae_rl.train_vae --config configs/vae_minigrid.yaml
```

Train a Skip-VAE:

```bash
uv run python -m skip_vae_rl.train_vae --config configs/skip_vae_minigrid.yaml
```

Train PPO from raw pixels:

```bash
uv run python -m skip_vae_rl.train_rl --config configs/ppo_raw.yaml
```

Train PPO using a frozen Skip-VAE encoder:

```bash
uv run python -m skip_vae_rl.train_rl --config configs/ppo_skip_vae_frozen.yaml
```

Visualize latent space:

```bash
uv run python -m skip_vae_rl.visualize_latent --config configs/latent_skip_vae.yaml
```

## What To Paste Back To Me

After running on Kaggle/Colab, send:

```text
runs/.../metrics.csv
runs/.../eval.json
reports/figures/recon_*.png
reports/figures/latent_*.png
```

Also paste any terminal error if a run fails.

## Compute Guidance

MiniGrid does not require a powerful GPU. A free Kaggle or Colab T4 is enough for VAE training. PPO can be CPU-bound because environment stepping is sequential, so do not start with huge runs.

Recommended scale:

```text
debug: 1k frames, 2 VAE epochs, 5k PPO steps
small: 10k frames, 10 VAE epochs, 50k PPO steps
final: 50k-100k frames, 30-50 VAE epochs, 300k-1M PPO steps
```

