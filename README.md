# Wan2GP (Experimental Fork)

This is an experimental fork of [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) for experimental features including specific architectures and toolchains.

Some features may be broken or perform slowly on specific platforms.

Please do not file issues about the gradio interface or the inference pipelines with this project.

## Why

- The upstream project is more focused on quickly adopting new technologies for popular architectures. Unified-memory systems and other cases require bespoke configurations. The uv tool is optimal for managing configuration by architecture and operating system, so a fork making uv first-class is in order.

### Prerequisite - uv

Use local package manager to install or

```bash
curl -LsSf https://astral-sh.github.io/uv/install.sh | sh
```

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Clone this repo

```
git clone https://github.com/autojake/Wan2GP-X
cd Wan2GP-X
```

## Sync platform-specific dependencies 

- [DGX Spark family (gb10 sm121)](DGXSPARK.md): `uv sync --extra dgx-spark` 
- All others: `uv sync`

## Start Wan2GP

```
uv run python wgp.py
```
