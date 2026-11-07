from __future__ import annotations
import torch 
import urllib.error
import urllib.request
from pathlib import Path
from utils.config import load_config

def download_dataset(url: str, dest: str | Path, force: bool = False) -> Path:
    """Download the dataset unless it already exists."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except (urllib.error.URLError, OSError) as exc:
        dest.unlink(missing_ok=True)  # don't leave a partial file behind
        raise RuntimeError(f"Failed to download dataset from {url}: {exc}") from exc
    return dest


def load_text(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m utils.data` to download it."
        )
    return path.read_text(encoding="utf-8")


def split_text(text: str, train_fraction: float = 0.9) -> tuple[str, str]:
    """Contiguous split (no shuffling) so validation text is truly unseen."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    cut = int(len(text) * train_fraction)
    return text[:cut], text[cut:]



def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random (input, target) windows.

    x[b] = data[i : i+block_size]
    y[b] = data[i+1 : i+block_size+1]   (x shifted one position right)
    so at every position t the model must predict the token at t+1.
    """
    if len(data) <= block_size:
        raise ValueError(
            f"Data length ({len(data)}) must be greater than block_size ({block_size})"
        )
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def main() -> None:
    cfg = load_config()
    path = download_dataset(cfg.data.url, cfg.data.path)
    text = load_text(path)
    train, val = split_text(text, cfg.data.train_fraction)
    chars = sorted(set(text))
    print(f"File:        {path}")
    print(f"Characters:  {len(text):,}")
    print(f"Vocab size:  {len(chars)}")
    print(f"Train / val: {len(train):,} / {len(val):,}")
    print("First 200 chars:\n" + text[:200])


if __name__ == "__main__":
    main()
    
    