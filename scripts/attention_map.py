#!/usr/bin/env python3
"""
어텐션 맵 모듈 — 판정의 근거 시각화 (해석가능성)
  · DINOv2 : 마지막 블록 CLS attention rollout → 패치 히트맵
  · YOLOv8 : 백본 SPPF 활성화 CAM → 객체 반응 영역
용도: 논문 Figure, 오판 진단, 판정 로그에 근거 이미지 자동 첨부
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485] * 3, [0.229] * 3),
])


def dinov2_attention_map(model, pil_img, patch=16):
    """[B=1 기준] (224,224) 정규화 히트맵. 자재에 집중하는 영역 확인용."""
    x = _TF(pil_img.convert("RGB")).unsqueeze(0).cuda().half()
    model = model.cuda().half().eval()
    with torch.inference_mode():
        out = model(x, output_attentions=True)
    att = out.attentions[-1].mean(1)          # (B, 1+257, 1+257) heads 평균
    a = att[0, 0, 1:].reshape(patch, patch).float()   # CLS→패치
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    h = F.interpolate(a[None, None], size=(224, 224), mode="bilinear")[0, 0]
    return h.cpu().numpy()


def yolo_cam(model, pil_img):
    """YOLOv8 백본 마지막(SPPF) 활성화 채널평균 CAM (그라디언트 불필요, 경량)"""
    from ultralytics import YOLO
    import cv2
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    # ultralytics 전처리 재사용 (letterbox 640)
    t = model.predictor  # 훅 없이 활성화만 캡처
    acts = {}

    def hook_fn(m, i, o):
        acts["f"] = o.detach()

    target = model.model.model[9]             # yolov8n SPPF
    h = target.register_forward_hook(hook_fn)
    try:
        model.predict(img, verbose=False)
    finally:
        h.remove()
    if "f" not in acts:
        return None
    f = acts["f"]                             # (1, C, H, W)
    cam = f.mean(1)[0]                        # 채널 평균
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = F.interpolate(cam[None, None], size=img.shape[:2], mode="bilinear")[0, 0]
    return cam.cpu().numpy()


def overlay(pil_img, heat, alpha=0.5, cmap="jet"):
    """원본 + 히트맵 오버레이 PIL 이미지"""
    import cv2
    img = np.array(pil_img.convert("RGB"))
    heat = cv2.resize((heat * 255).astype(np.uint8), (img.shape[1], img.shape[0]))
    color = cv2.applyColorMap(heat, getattr(cv2, f"COLORMAP_{cmap.upper()}"))
    blend = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 1 - alpha, color, alpha, 0)
    return Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "data/sample.jpg"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/attention.png"
    from transformers import Dinov2Model
    m = Dinov2Model.from_pretrained("facebook/dinov2-small")
    h = dinov2_attention_map(m, Image.open(img))
    overlay(Image.open(img), h).save(out)
    print("저장:", out)
