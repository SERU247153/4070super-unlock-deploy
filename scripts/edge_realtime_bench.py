#!/usr/bin/env python3
"""
엣지 실시간 예산 벤치마크 — 단계별 ms 측정 (참조: 데스크톱, 엣지는 동일 스크립트 사용)
루프: 캡처(스킵) → YOLO 검출 → ROI 크롭 → DINOv2 분류 → 판정
목표: 루프 ≤ 50ms (20FPS 판정), 프레임 스킵 2~3
"""
import time

import torch
from PIL import Image


def bench_yolo(model, img, n=20):
    model.predict(img, verbose=False)          # 워밍업
    t = []
    for _ in range(n):
        s = time.perf_counter()
        model.predict(img, verbose=False)
        torch.cuda.synchronize()
        t.append((time.perf_counter() - s) * 1000)
    return sum(t) / len(t)


def bench_embed(backbone, img, tf, n=20):
    x = tf(img.convert("RGB")).unsqueeze(0).cuda().half()
    backbone = backbone.cuda().half().eval()
    with torch.inference_mode():
        backbone(x)
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        with torch.inference_mode():
            backbone(x)
        torch.cuda.synchronize()
        t.append((time.perf_counter() - s) * 1000)
    return sum(t) / len(t)


def bench_embed_compiled(backbone, img, tf, n=20):
    """torch.compile(fullgraph) — 커널 융합으로 추가 단축"""
    x = tf(img.convert("RGB")).unsqueeze(0).cuda().half()
    m = torch.compile(backbone.cuda().half().eval(), fullgraph=True)
    with torch.inference_mode():
        m(x)
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        with torch.inference_mode():
            m(x)
        torch.cuda.synchronize()
        t.append((time.perf_counter() - s) * 1000)
    return sum(t) / len(t)


def bench_embed_cudagraph(backbone, img, tf, n=20):
    """CUDA Graph 캡처 — 커널 런치 오버헤드 제거 (엣지 루프의 최종 형태)"""
    x = tf(img.convert("RGB")).unsqueeze(0).cuda().half()
    m = backbone.cuda().half().eval()
    with torch.inference_mode():
        o = m(x)
    g = torch.cuda.CUDAGraph()
    static = x.clone()
    with torch.cuda.graph(g):                      # 그래프 캡처 (고정 입력 버퍼)
        o = m(static)
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        static.copy_(x)                            # 버퍼 재사용
        s = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        t.append((time.perf_counter() - s) * 1000)
    return sum(t) / len(t)


if __name__ == "__main__":
    import os
    from transformers import Dinov2Model
    from torchvision import transforms
    from ultralytics import YOLO

    img_path = os.environ.get("IMG", "/workspace/edge-project/data/sample.jpg")
    img = Image.open(img_path)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485] * 3, [0.229] * 3)])

    m = YOLO("yolov8n.pt")
    b = Dinov2Model.from_pretrained("facebook/dinov2-small",
                                    cache_dir=os.environ.get("HF_HOME"))
    t_yolo = bench_yolo(m, img)
    t_emb = bench_embed(b, img, tf)
    t_pre = 1.0                              # 전처리 예산(가정)
    print(f"YOLOv8n 검출 : {t_yolo:6.1f} ms")
    print(f"DINOv2-S 임베딩(기본) : {t_emb:6.1f} ms")
    try:
        t_cmp = bench_embed_compiled(b, img, tf)
        print(f"DINOv2-S 임베딩(torch.compile): {t_cmp:6.1f} ms "
              f"({(1 - t_cmp / t_emb) * 100:+.0f}%)")
    except Exception as e:
        print("torch.compile 실패(스킵):", type(e).__name__)
        t_cmp = t_emb
    try:
        t_cg = bench_embed_cudagraph(b, img, tf)
        print(f"DINOv2-S 임베딩(CUDA Graph): {t_cg:6.1f} ms "
              f"({(1 - t_cg / t_emb) * 100:+.0f}%)")
    except Exception as e:
        print("CUDA Graph 실패(스킵):", type(e).__name__)
        t_cg = t_emb
    best = min(t_emb, t_cmp, t_cg)
    loop = t_yolo + best + t_pre
    print(f"최적 루프 합계(전처리 {t_pre:.1f}ms 가정): {loop:6.1f} ms → {1000/loop:5.1f} FPS")
    print("판정: 50ms 목표", "달성 ✓" if loop <= 50 else "초과 — 증류/양자화 검토")
