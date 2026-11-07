from __future__ import annotations
import argparse
import sys
from models.sampling import generate_text
from training.checkpoint import load_model
from utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", default="outputs/runs/gpt_small/best.pt")
    parser.add_argument("--prompt", default="", help="starting text (default: a newline)")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="for reproducible samples")
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    device = get_device()
    try:
        model, tokenizer, ckpt = load_model(args.checkpoint, device)
        print(f"Loaded {ckpt['model_type']} from {args.checkpoint} "
              f"(iteration {ckpt['iteration']}, device {device})")
        for i in range(args.num_samples):
            seed = None if args.seed is None else args.seed + i
            text = generate_text(
                model, tokenizer, args.prompt, args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, device=device, seed=seed,
            )
            if args.num_samples > 1:
                print(f"\n===== sample {i + 1} =====")
            print(text)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()