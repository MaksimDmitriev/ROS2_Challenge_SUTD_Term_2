#!/usr/bin/env python3
"""Local YOLO sanity-check runner (no ROS required).

Runs a YOLO model on a local image file or directory and prints:
- detected raw labels
- mapped roles
- final mission-style status
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROLE_ALIASES = {
    "military": {"military", "soldier", "officer", "military officer"},
    "researcher": {"researcher", "scientist", "lab coat", "doctor"},
    "student": {"student"},
    "worker": {"worker", "construction worker"},
}
STOP_SIGN_ALIASES = {"stop sign", "stopsign"}


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def map_label_to_role(label: str) -> str | None:
    normalized = normalize_label(label)
    for role, aliases in ROLE_ALIASES.items():
        if normalized in aliases:
            return role
    return None


def classify_labels(labels: list[str]) -> tuple[list[str], str]:
    roles = []
    for label in labels:
        role = map_label_to_role(label)
        if role is not None:
            roles.append(role)

    normalized_labels = [normalize_label(label) for label in labels]

    if any(label in STOP_SIGN_ALIASES for label in normalized_labels):
        return roles, "stop_sign"
    if any(role in roles for role in ("military", "worker")):
        return roles, "intruder"
    if any(role in roles for role in ("researcher", "student")):
        return roles, "authorized"
    return roles, "empty"


def iter_images(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTS:
            yield source
        return

    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def extract_labels(result) -> list[str]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = result.names
    labels = []
    for cls_id in boxes.cls.tolist():
        cls_id = int(cls_id)
        if isinstance(names, dict):
            label = str(names.get(cls_id, cls_id))
        else:
            label = str(names[cls_id]) if 0 <= cls_id < len(names) else str(cls_id)
        labels.append(label)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Local YOLO test runner for mission labels.")
    parser.add_argument(
        "--model",
        default="src/yolo_detector/models/best.pt",
        help="Path to model weights (.pt).",
    )
    parser.add_argument(
        "--source",
        default="dataset/images/val",
        help="Image file or directory to test.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold passed to YOLO.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    source_path = Path(args.source)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not source_path.exists():
        raise SystemExit(f"Source not found: {source_path}")

    model = YOLO(str(model_path))
    images = list(iter_images(source_path))
    if not images:
        raise SystemExit(f"No images found under: {source_path}")

    print(f"Model:  {model_path}")
    print(f"Source: {source_path}")
    print("-" * 120)
    print(f"{'image':50} | {'labels':35} | {'roles':20} | status")
    print("-" * 120)

    status_counts: dict[str, int] = {}
    for image_path in images:
        results = model(str(image_path), conf=args.conf, verbose=False)
        labels = extract_labels(results[0])
        roles, status = classify_labels(labels)

        status_counts[status] = status_counts.get(status, 0) + 1
        labels_text = ",".join(normalize_label(x) for x in labels) or "-"
        roles_text = ",".join(roles) or "-"
        print(f"{image_path.name[:50]:50} | {labels_text[:35]:35} | {roles_text[:20]:20} | {status}")

    print("-" * 120)
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))


if __name__ == "__main__":
    main()
