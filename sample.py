#!/usr/bin/env python3
"""Local GPT-vision sanity-check runner (no ROS required)."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import Iterable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PROMPT = (
    "Classify this single scene image for security. "
    "Rules: military or worker => intruder. "
    "Student or researcher or lab assistant => authorized. "
    "If no relevant person is visible => empty. "
    "If a stop sign is clearly visible => stop_sign. "
    "Return exactly one token: intruder, authorized, empty, or stop_sign."
)
VALID_STATUSES = {"intruder", "authorized", "empty", "stop_sign"}


def iter_images(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTS:
            yield source
        return

    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def parse_status(text: str) -> str:
    lowered = (text or "").strip().lower()
    if "stop_sign" in lowered or "stop sign" in lowered:
        return "stop_sign"
    if "intruder" in lowered:
        return "intruder"
    if "authorized" in lowered:
        return "authorized"
    if "empty" in lowered:
        return "empty"
    return "empty"


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/jpeg"
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def classify_with_openai(client: OpenAI, model: str, prompt: str, image_path: Path) -> tuple[str, str]:
    image_url = image_to_data_url(image_path)

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            max_output_tokens=16,
        )
        raw_text = (getattr(response, "output_text", "") or "").strip()
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=16,
        )
        raw_text = (response.choices[0].message.content or "").strip()

    return parse_status(raw_text), raw_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GPT vision test runner for mission labels.")
    parser.add_argument(
        "--source",
        default="dataset/images/val",
        help="Image file or directory to test.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI vision model name.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for classification.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of images to evaluate (0 = no limit).",
    )
    args = parser.parse_args()

    if OpenAI is None:
        raise SystemExit("Missing dependency: install python package `openai`.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    source_path = Path(args.source)
    if not source_path.exists():
        raise SystemExit(f"Source not found: {source_path}")

    images = list(iter_images(source_path))
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No images found under: {source_path}")

    client = OpenAI(api_key=api_key)

    print(f"Model:  {args.model}")
    print(f"Source: {source_path}")
    print("-" * 120)
    print(f"{'image':45} | {'status':10} | raw_response")
    print("-" * 120)

    status_counts: dict[str, int] = {}
    for image_path in images:
        status, raw_text = classify_with_openai(client, args.model, args.prompt, image_path)
        if status not in VALID_STATUSES:
            status = "empty"
        status_counts[status] = status_counts.get(status, 0) + 1
        print(f"{image_path.name[:45]:45} | {status:10} | {raw_text[:55]}")

    print("-" * 120)
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))


if __name__ == "__main__":
    main()
