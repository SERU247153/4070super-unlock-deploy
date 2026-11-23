#!/usr/bin/env python3
"""실습 가이드 노트북 3종 생성 (notebooks/00~02.ipynb)"""
import json
import os

NB = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}}}


def cell(src, kind="code"):
    return {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {})}


nb0 = dict(NB)
nb0["cells"] = [
    cell("# 00 — 언록 성능 확인\nGPU 텐서코어 언록 전/후 성능을 이 노트북에서 직접 재현합니다."),
    cell("""import torch
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
print("언록 상태: journalctl -k -b | grep gpu_unlock 으로 확인 (unlocked)")"""),
    cell("""# BF16 GEMM (언록 대상 경로) — 언록 서버 기준 ~130 TFLOPS
import time, statistics, torch
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3): torch.matmul(a, b)
torch.cuda.synchronize()
ts = []
for _ in range(7):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    torch.matmul(a, b); torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
tflops = 2 * 8192**3 / statistics.median(ts) / 1e12
print(f"BF16 GEMM: {tflops:.1f} TFLOPS (잠긴 카드 기준 71)")"""),
    cell("""# FlashAttention 2 처리량 (fwd) — 잠긴 상한 71 TFLOPS 대비
from flash_attn import flash_attn_func
import torch, time, statistics
B, H, S, D = 1, 32, 2048, 128
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
def step(): flash_attn_func(q, k, v, causal=True); torch.cuda.synchronize()
for _ in range(3): step()
ts = []
for _ in range(7):
    t0 = time.perf_counter(); step(); ts.append(time.perf_counter() - t0)
tflops = 4 * B * H * S * S * D / statistics.median(ts) / 1e12
print(f"FlashAttention fwd: {tflops:.1f} TFLOPS")"""),
]
nb0["metadata"]["title"] = "00_성능확인"

nb1 = dict(NB)
nb1["cells"] = [
    cell("# 01 — 학습 템플릿 (12GB 최대 활용)\nYOLOv8 자재 검출 + DINOv2 분류 — 소량 데이터 최적화 조합."),
    cell("""# YOLOv8 학습 (데이터 준비 후)
# from ultralytics import YOLO
# m = YOLO("yolov8n.pt")
# r = m.train(data="/workspace/edge-project/data/data.yaml",
#             cfg="/workspace/edge-project/scripts/edge_yolo_config.yaml")
# print(r.results_dict)"""),
    cell("""# DINOv2 선형 프로브 (소량 데이터 분류 — 2026 문헌: 미세조정 능가)
from scripts.dinov2_classifier import get_backbone, train_probe
# backbone = get_backbone()
# head = train_probe(backbone, train_loader, n_classes, epochs=20)
# torch.save({"head": head.state_dict()}, "probe.pt")"""),
    cell("""# Long-tail 대응 (E-IRFS 샘플링 + CB-Focal)
from scripts.longtail import analyze, eirfs_weights, cb_focal_loss
# stats = analyze(train_labels)            # 불균형 지수 확인
# w = eirfs_weights(img_labels, n_inst)    # WeightedRandomSampler 가중치
# loss = cb_focal_loss(logits, targets, counts)"""),
    cell("""# PyTorch 문서 권장 설정 (이미 노트북 시작 시 자동 주입됨)
import torch
torch.set_float32_matmul_precision("high")   # TF32 (언록 경로 2배)
torch.backends.cudnn.benchmark = True
print("설정 확인 완료")"""),
]
nb1["metadata"]["title"] = "01_학습템플릿"

nb2 = dict(NB)
nb2["cells"] = [
    cell("# 02 — 추론 실습\n검출 → 분류 → 판정(3회 연속) → 이벤트 기록의 전체 흐름."),
    cell("""# YOLO 검출 + ROI 크롭 (자재 사진으로 테스트)
from ultralytics import YOLO
m = YOLO("yolov8n.pt")
r = m.predict("/workspace/edge-project/data/sample.jpg", verbose=False)[0]
print("검출:", [(m.names[int(c)], round(float(s), 2)) for c, s in zip(r.boxes.cls, r.boxes.conf)])"""),
    cell("""# DINOv2 임베딩 + 어텐션 맵
from transformers import Dinov2Model
from PIL import Image
from scripts.attention_map import dinov2_attention_rollout, overlay
b = Dinov2Model.from_pretrained("facebook/dinov2-small", cache_dir="/workspace/.hf-cache")
heat, _ = dinov2_attention_rollout(b, Image.open("/workspace/edge-project/data/sample.jpg"))
overlay(Image.open("/workspace/edge-project/data/sample.jpg"), heat).save("attention.png")
print("어텐션 맵 저장: attention.png")"""),
    cell("""# 판정 상태머신 (3회 연속 확정 + 입출고 이벤트)
from scripts.edge_infer import DecisionBuffer, StockStateMachine, lora_packet
buf = DecisionBuffer(3); stk = StockStateMachine()
print(buf.update(0, True), buf.update(0, True), buf.update(0, True))
ev = stk.update(0, 15)
print("이벤트:", ev, "| LoRa 패킷:", len(lora_packet(1, 1, ev)), "B")"""),
]
nb2["metadata"]["title"] = "02_추론실습"

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
os.makedirs(outdir, exist_ok=True)
for name, nb in (("00_perf_check.ipynb", nb0), ("01_train_template.ipynb", nb1),
                 ("02_inference_demo.ipynb", nb2)):
    with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("생성:", name)
