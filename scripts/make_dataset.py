#!/usr/bin/env python3
"""
데이터셋 도구 — 사진만 생기면 5분 만에 학습 시작
  · init_project: data.yaml + 디렉토리 구조 생성
  · coco2yolo    : COCO JSON 라벨 → YOLO txt 변환
  · split        : train/val/시험 분할
사용:
  python make_dataset.py init --classes 박스,포대,롤 --out /workspace/edge-project/data
  python make_dataset.py coco2yolo --json labels.json --images images/
  python make_dataset.py split --src data/images --ratio 0.8
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def init_project(classes, out):
    cls = [c.strip() for c in classes.split(",")]
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        Path(out, sub).mkdir(parents=True, exist_ok=True)
    yaml = {"path": out, "train": "images/train", "val": "images/val",
            "names": {i: c for i, c in enumerate(cls)}}
    p = Path(out, "data.yaml")
    p.write_text("path: " + out + "\ntrain: images/train\nval: images/val\nnames:\n"
                 + "\n".join(f"  {i}: {c}" for i, c in enumerate(cls)), encoding="utf-8")
    print("data.yaml 생성:", p, "| 클래스:", cls)


def coco2yolo(json_path, images_dir):
    d = json.load(open(json_path, encoding="utf-8"))
    cat2id = {c["id"]: i for i, c in enumerate(d["categories"])}
    w = h = 1
    if d["images"]:
        w, h = d["images"][0]["width"], d["images"][0]["height"]
    for img in d["images"]:
        anns = [a for a in d["annotations"] if a["image_id"] == img["id"]]
        lines = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{cat2id[a['category_id']]} {cx:.6f} {cy:.6f} "
                         f"{bw / w:.6f} {bh / h:.6f}")
        stem = Path(img["file_name"]).stem
        Path(images_dir, stem + ".txt").write_text("\n".join(lines), encoding="utf-8")
    print("COCO→YOLO 변환 완료:", len(d["images"]), "장")


def split(src, ratio=0.8, seed=0):
    import random
    files = sorted(Path(src).glob("*"))
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in files if f.suffix.lower() in exts]
    random.Random(seed).shuffle(files)
    n = int(len(files) * ratio)
    for tag, part in (("train", files[:n]), ("val", files[n:])):
        for f in part:
            dst = Path(src).parent / "images" / tag / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst)
    print(f"분할 완료: train {n} / val {len(files) - n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("init"); p1.add_argument("--classes", required=True)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("coco2yolo"); p2.add_argument("--json", required=True)
    p2.add_argument("--images", required=True)
    p3 = sub.add_parser("split"); p3.add_argument("--src", required=True)
    p3.add_argument("--ratio", type=float, default=0.8)
    a = ap.parse_args()
    if a.cmd == "init":
        init_project(a.classes, a.out)
    elif a.cmd == "coco2yolo":
        coco2yolo(a.json, a.images)
    else:
        split(a.src, a.ratio)
