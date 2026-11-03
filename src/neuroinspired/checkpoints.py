"""Checkpoint provenance and compatibility with older weights-only files."""
import hashlib
import platform
import subprocess
from pathlib import Path

import torch


def select_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def run_metadata(project_root, config, device):
    paths = [project_root / 'scripts/train.py', project_root / 'requirements.txt']
    paths += sorted((project_root / 'src/neuroinspired').glob('*.py'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode())
        digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=project_root, stderr=subprocess.DEVNULL,
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        'config': config, 'source_sha256': digest.hexdigest(), 'git_revision': revision,
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'device': device, 'cuda': torch.version.cuda},
    }


def checkpoint_matches(path, metadata):
    """Only reuse checkpoints with matching settings, source and environment."""
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    saved = checkpoint.get('metadata', {})
    return all(saved.get(key) == metadata[key]
               for key in ('config', 'source_sha256', 'environment'))


def load_weights(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return checkpoint.get('model_state_dict', checkpoint)
