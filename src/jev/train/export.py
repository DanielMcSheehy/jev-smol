"""Merge a trained jev LoRA adapter back into its base model.

Produces a standalone checkpoint at ``--out`` (default ``runs/merged``)
containing the merged weights plus the processor, loadable via
``AutoModelForImageTextToText.from_pretrained(out)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main(argv=None) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="path to the saved adapter dir (e.g. runs/.../adapter)")
    parser.add_argument("--base", type=str, required=True,
                        help="base model id or path the adapter was trained on")
    parser.add_argument("--out", type=Path, default=Path("runs/merged"),
                        help="output dir for the merged model (default: runs/merged)")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=dtype)
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out)
    processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.out)
    print(f"[export] merged adapter {args.adapter} into {args.out} (dtype={dtype})")
    return args.out


if __name__ == "__main__":
    main()
