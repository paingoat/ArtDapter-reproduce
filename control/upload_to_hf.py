"""
Upload trained checkpoint(s) to Hugging Face Hub.

Usage:
    # Upload a specific checkpoint (recommended: best one from WandB):
    python control/upload_to_hf.py --repo_id your-hf-username/artdapter --ckpt ckpt/trained/your-run-name-epoch=X-step=Y.ckpt

    # Upload ALL checkpoints in ckpt/trained/ (if you're not sure which is best):
    python control/upload_to_hf.py --repo_id your-hf-username/artdapter --all

    # Login first if not already authenticated:
    huggingface-cli login
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def get_args():
    parser = argparse.ArgumentParser(description='Upload ArtDapter checkpoint to Hugging Face Hub')
    parser.add_argument('--repo_id',    '-r',   type=str,   required=True,
                        help='HuggingFace repo id, e.g. your-username/artdapter')
    parser.add_argument('--ckpt',       '-c',   type=str,   default=None,
                        help='Path to a specific checkpoint file to upload')
    parser.add_argument('--all',        '-a',   action='store_true',
                        help='Upload all .ckpt files in ckpt/trained/')
    parser.add_argument('--ckpt_dir',           type=str,   default='./ckpt/trained',
                        help='Directory to search for checkpoints (used with --all)')
    parser.add_argument('--repo_folder',        type=str,   default='trained',
                        help='Subfolder inside the HF repo to upload into')
    parser.add_argument('--private',            action='store_true',
                        help='Create repo as private (only on first upload)')
    return parser.parse_args()


def main():
    args = get_args()
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=args.repo_id, repo_type='model', private=args.private, exist_ok=True)
    print(f'Repo: https://huggingface.co/{args.repo_id}')

    if args.all:
        ckpt_dir = Path(args.ckpt_dir)
        ckpts = sorted(ckpt_dir.glob('*.ckpt'))
        if not ckpts:
            print(f'No .ckpt files found in {ckpt_dir}')
            return
        print(f'Found {len(ckpts)} checkpoint(s):')
        for c in ckpts:
            print(f'  {c.name}  ({c.stat().st_size / 1e9:.2f} GB)')
        confirm = input('Upload all? [y/N]: ').strip().lower()
        if confirm != 'y':
            print('Aborted.')
            return
        files_to_upload = ckpts
    elif args.ckpt:
        files_to_upload = [Path(args.ckpt)]
    else:
        # List available and let user pick
        ckpt_dir = Path(args.ckpt_dir)
        ckpts = sorted(ckpt_dir.glob('*.ckpt'))
        if not ckpts:
            print(f'No .ckpt files found in {ckpt_dir}')
            print('Tip: check WandB for the step with lowest train/loss, then pass --ckpt <path>')
            return
        print('Available checkpoints (tip: check WandB for lowest train/loss):')
        for i, c in enumerate(ckpts):
            print(f'  [{i}] {c.name}  ({c.stat().st_size / 1e9:.2f} GB)')
        idx = input('Enter index to upload (or "a" for all): ').strip()
        if idx.lower() == 'a':
            files_to_upload = ckpts
        else:
            files_to_upload = [ckpts[int(idx)]]

    for ckpt_path in files_to_upload:
        dest = f'{args.repo_folder}/{ckpt_path.name}'
        print(f'\nUploading {ckpt_path.name} → {dest} ...')
        url = api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=dest,
            repo_id=args.repo_id,
            repo_type='model',
        )
        print(f'Done: {url}')

    print('\nAll uploads complete.')
    print(f'View at: https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
