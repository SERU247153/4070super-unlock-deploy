#!/usr/bin/env python3
"""
공개 건설 현장 데이터셋 드라이런 (LTS 파이프라인 실증)
  · keremberke/construction-safety-object-detection (Roboflow 공개, HF 미러)
  · 17클래스: barricade, dumpster, excavators, gloves, hardhat, mask,
    no-hardhat, no-mask, no-safety vest, person, safety net, safety shoes,
    safety vest, dump truck, mini-van, truck, wheel loader
  · COCO 절대 bbox -> YOLO 정규화 xywh 변환 -> data.yaml
"""
import os

from datasets import load_dataset

OUT = "/workspace/edge-project/data/construction-safety"
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def main():
    for d in ("images", "labels"):
        for s in SPLIT_MAP.values():
            os.makedirs(f"{OUT}/{d}/{s}", exist_ok=True)

    ds = load_dataset("keremberke/construction-safety-object-detection", "full")
    names = ds["train"].features["objects"].feature["category"].names

    for hf_split, yolo_split in SPLIT_MAP.items():
        for i, item in enumerate(ds[hf_split]):
            w, h = item["width"], item["height"]
            img = item["image"].convert("RGB")
            name = f"{yolo_split}_{i:06d}"
            img.save(f"{OUT}/images/{yolo_split}/{name}.jpg", quality=95)
            objs = item["objects"]                       # 컬럼형: {bbox:[...], category:[...]}
            with open(f"{OUT}/labels/{yolo_split}/{name}.txt", "w", encoding="utf-8") as f:
                for bbox, cat in zip(objs["bbox"], objs["category"]):
                    x, y, bw, bh = bbox                  # COCO 절대 xywh
                    f.write(f"{cat} {(x + bw / 2) / w:.6f} "
                            f"{(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")
    counts = {s: len(ds[h]) for h, s in SPLIT_MAP.items()}
    print("변환 완료:", counts, "| 클래스:", len(names))

    yaml = (f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
            + "\n".join(f"  {i}: {c}" for i, c in enumerate(names)))
    with open(f"{OUT}/data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml)
    print("data.yaml 생성 완료")


if __name__ == "__main__":
    main()
