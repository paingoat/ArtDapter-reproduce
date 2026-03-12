"""
sanity_check.py — Kiểm tra toàn bộ pipeline trước khi train chính thức.
Chạy < 2 phút. Nếu pass hết thì training gần như chắc chắn không crash.

Usage:
    PYTHONPATH=. python control/sanity_check.py --config configs/pod_train_config.yaml
"""
import os
import sys
import argparse
from pathlib import Path

# Dataset cache → /workspace để persist qua pod restart (nhất quán với train.sh)
os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/hf_cache/datasets")

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def check(name, func):
    global CHECKS_PASSED, CHECKS_FAILED
    try:
        func()
        CHECKS_PASSED += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        CHECKS_FAILED += 1
        print(f"  [FAIL] {name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", type=str, default="configs/pod_train_config.yaml")
    args = parser.parse_args()
    config_path = args.config

    print("=" * 50)
    print("  ArtDapter Sanity Check")
    print("=" * 50)

    # ── Check 1: Import tất cả module ──
    print("\n[1/6] Import modules...")

    def check_imports():
        import torch
        import pytorch_lightning
        import lightning
        import transformers
        import diffusers
        import omegaconf
        import einops
        import wandb
        import datasets
        import safetensors
        import open_clip
        import tqdm
        import PIL

    check("Core dependencies", check_imports)

    def check_project_imports():
        from models import ArtDaptedModel, ArtDapterTSC, ELLA, BaselineModel
        from models.util import load_state_dict, create_model
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.custom_ddim import CustomDDIMSampler
        from utils import freeze, count_parameters, load_weights
        from dataset import CompArt

    check("Project modules", check_project_imports)

    def check_xformers():
        import xformers
        import xformers.ops

    check("xformers (optional)", check_xformers)

    # ── Check 2: CUDA ──
    print("\n[2/6] GPU & CUDA...")

    def check_cuda():
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"         GPU: {name} | VRAM: {vram:.1f} GB | CUDA: {torch.version.cuda}")

    check("CUDA available", check_cuda)

    def check_bf16():
        import torch
        assert torch.cuda.is_bf16_supported(), "bf16 not supported"

    check("bf16 support", check_bf16)

    # ── Check 3: Config ──
    print("\n[3/6] Config...")

    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)

    def check_config():
        assert hasattr(config, "training"), "Missing 'training' section"
        assert hasattr(config, "model"), "Missing 'model' section"
        assert hasattr(config, "dataset"), "Missing 'dataset' section"
        assert hasattr(config, "logger"), "Missing 'logger' section"
        eff_batch = config.training.dataloader.batch_size * config.training.get("accumulate_grad_batches", 1)
        print(f"         Effective batch size: {config.training.dataloader.batch_size} x {config.training.get('accumulate_grad_batches', 1)} = {eff_batch}")
        print(f"         Precision: {config.training.precision} | Steps: {config.training.training_steps}")

    check(f"Load config: {config_path}", check_config)

    # ── Check 4: Weights ──
    print("\n[4/6] Weights...")

    def check_init_ckpt():
        init_path = Path(config.model.init_path)
        assert init_path.exists(), f"init.ckpt not found at {init_path}"
        size_gb = init_path.stat().st_size / 1024**3
        print(f"         {init_path} ({size_gb:.2f} GB)")

    check("init.ckpt exists", check_init_ckpt)

    # ── Check 5: Dataset ──
    print("\n[5/6] Dataset (load 1 batch)...")

    def check_dataset():
        from ldm.util import instantiate_from_config
        from torch.utils.data import DataLoader

        ds = instantiate_from_config(config.dataset)
        dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn, num_workers=0)
        batch = next(iter(dl))
        print(f"         Keys: {list(batch.keys())}")
        print(f"         Image shape: {batch['image'].shape}")
        print(f"         Caption[0]: {batch['caption'][0][:80]}...")

    check("Dataset load", check_dataset)

    # ── Check 6: Model + forward/backward ──
    print("\n[6/6] Model forward + backward (1 step, batch=1)...")

    def check_model():
        import torch
        from ldm.util import instantiate_from_config
        from models.util import load_state_dict as load_sd
        from utils import freeze
        from torch.utils.data import DataLoader

        model = instantiate_from_config(config.model)
        model.load_state_dict(load_sd(config.model.init_path, location="cpu"))
        model = model.cuda()

        if config.model.sd_locked:
            freeze(model.model)

        model.learning_rate = config.training.learning_rate
        model.weight_decay = config.training.weight_decay
        model.sd_locked = config.model.sd_locked

        ds = instantiate_from_config(config.dataset)
        dl = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn, num_workers=0)
        batch = next(iter(dl))

        # Move batch to GPU
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()

        # Forward + backward
        loss, loss_dict = model.training_step(batch, 0)
        loss.backward()

        vram_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"         Loss: {loss.item():.4f}")
        print(f"         Loss dict: { {k: f'{v.item():.4f}' for k, v in loss_dict.items()} }")
        print(f"         Peak VRAM (batch=1): {vram_used:.2f} GB")

        # Cleanup
        del model, batch
        torch.cuda.empty_cache()

    check("Forward + backward pass", check_model)

    # ── Summary ──
    print("\n" + "=" * 50)
    print(f"  Result: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
    print("=" * 50)

    if CHECKS_FAILED > 0:
        print("\nCó check bị FAIL. Sửa lỗi trước khi train.")
        sys.exit(1)
    else:
        print("\nTất cả OK! Sẵn sàng train:")
        print("  bash control/train.sh")
        sys.exit(0)


if __name__ == "__main__":
    main()
