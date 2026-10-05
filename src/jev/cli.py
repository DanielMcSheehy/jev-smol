"""Command-line interface: ``jev serve`` and ``jev decide``."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from typing import Any, Dict, Optional

from .api.errors import JevError
from .schema import JevRequest


def _load_questions(arg: str) -> Dict[str, Any]:
    """Accept a path to a JSON file or an inline JSON object."""
    if os.path.exists(arg):
        with open(arg, "r", encoding="utf-8") as fh:
            return json.load(fh)
    try:
        parsed = json.loads(arg)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--questions: {arg!r} is neither an existing file nor valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise SystemExit("--questions must be a JSON object mapping question ids to question specs")
    return parsed


def _load_state(args: argparse.Namespace) -> Any:
    if args.state_file:
        with open(args.state_file, "r", encoding="utf-8") as fh:
            return fh.read()
    return args.state


def _image_to_request_image(path: str) -> Dict[str, str]:
    from PIL import Image

    fmt = Image.open(path).format  # PIL reports PNG/JPEG/WEBP
    content_type = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
    }.get(fmt)
    if content_type is None:
        raise SystemExit(f"--image {path!r}: unsupported format {fmt!r}; use png, jpeg, or webp")
    with open(path, "rb") as fh:
        raw = fh.read()
    return {"content_type": content_type, "base64": base64.b64encode(raw).decode("ascii")}


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    uvicorn.run(
        create_app(backend=args.backend, model=args.model, device=args.device),
        host=args.host,
        port=args.port,
    )
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    from .engine import get_engine

    state = _load_state(args)
    if state is None:
        raise SystemExit("decide requires --state TEXT or --state-file FILE")
    questions = _load_questions(args.questions)
    images = [_image_to_request_image(p) for p in args.image]
    request = JevRequest(
        model=args.model,
        state=state,
        questions=questions,
        images=images or None,
    )
    kwargs: Dict[str, Any] = {}
    if args.backend == "vlm":
        kwargs["device"] = args.device
        if args.model:
            kwargs["model"] = args.model
    elif args.model:
        kwargs["model_id"] = args.model
    engine = get_engine(args.backend, **kwargs)
    try:
        response = engine.decide(request)
    except JevError as exc:
        print(json.dumps({"error": {"type": exc.error_type, "message": exc.message}}), file=sys.stderr)
        return exc.status_code
    print(response.model_dump_json())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev", description="jev — Jev/CLEF-compatible decision model")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="serve the System One HTTP API")
    serve.add_argument("--backend", choices=["mock", "vlm"], default="mock")
    serve.add_argument("--model", default=None, help="model path/id (vlm backend: HF repo or local path)")
    serve.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(func=_cmd_serve)

    decide = sub.add_parser("decide", help="run one decision from the command line")
    decide.add_argument("--state", default=None, help="state as a literal string")
    decide.add_argument("--state-file", default=None, help="read the state from a file")
    decide.add_argument("--questions", required=True,
                        help="questions as a JSON file path or inline JSON object")
    decide.add_argument("--image", action="append", default=[],
                        help="path to a png/jpeg/webp image (repeatable, max 4)")
    decide.add_argument("--backend", choices=["mock", "vlm"], default="mock")
    decide.add_argument("--model", default=None, help="reported model name (vlm backend: HF repo/path)")
    decide.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    decide.set_defaults(func=_cmd_decide)
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
