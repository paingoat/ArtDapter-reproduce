"""
download_dataset.py — Tải dataset từ HuggingFace trước khi train/sanity check.
"""
import os
import sys
import argparse
from omegaconf import OmegaConf
import datasets

os.environ.setdefault("HF_DATASETS_CACHE", "/backup/data/art-gen/hf_cache/datasets")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", type=str, default="configs/pod_train_config.yaml")
    args = parser.parse_args()
    
    config_path = args.config
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        print(f"Lỗi: Không thể đọc config tại {config_path}: {e}")
        sys.exit(1)
        
    dataset_path = config.dataset.params.dataset_path
    
    print("=" * 50)
    print("  ArtDapter Download Dataset")
    print("=" * 50)
    print(f"Dataset:   {dataset_path}")
    print(f"Cache dir: {os.environ.get('HF_DATASETS_CACHE')}")
    
    for split in ["train", "test"]:
        print(f"\nDownloading split: {split}...")
        try:
            # Chỉ gọi load_dataset để HuggingFace tự động tải và lưu vào cache
            datasets.load_dataset(dataset_path, split=split)
            print(f"  [OK] Đã tải xong split '{split}'.")
        except Exception as e:
            print(f"  [FAIL] Lỗi khi tải split '{split}': {e}")
            sys.exit(1)
            
    print("\n" + "=" * 50)
    print("Tất cả đã tải xong! Bước tiếp theo:")
    print("  bash control/sanity_check.sh")
    print("=" * 50)

if __name__ == "__main__":
    main()
