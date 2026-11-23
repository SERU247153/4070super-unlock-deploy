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


def dinov2_attention_rollout(model, pil_img, patch=16, normalize_keys=True, all_heads=True):
    """전층 attention rollout + 키 정규화 (DINOv2 어텐션맵 patchy 현상 보정)
    · Abnar & Zuidema(2020) rollout: 잔차 고려 층간 누적 곱
    · key norm 정규화: 자기지도 ViT의 CLS 주의 산만 보정 (문헌 표준)
    → (224,224) 정규화 히트맵, + 어텐션 가중 풀링용 패치 가중치(256,)"""
    x = _TF(pil_img.convert("RGB")).unsqueeze(0).cuda().half()
    model = model.cuda().half().eval()
    with torch.inference_mode():
        out = model(x, output_attentions=True)
    atts = out.attentions                    # L × (B,H,257,257)
    L = len(atts)
    mat = torch.eye(257, device=atts[0].device, dtype=torch.float32)
    for a in atts:
        A = a[0].mean(0).float()             # heads 평균 (B=1)
        if normalize_keys:
            kn = A.sum(0)                    # key 기준 정규화 (열합)
            A = A / (kn + 1e-8)
        mat = 0.5 * mat + 0.5 * (A @ mat)    # rollout (잔차 0.5)
    w = mat[0, 1:]                           # CLS→패치
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    h = F.interpolate(w.reshape(1, 1, patch, patch),
                      size=(224, 224), mode="bilinear")[0, 0]
    return h.cpu().numpy(), w


def attention_weighted_embed(model, pil_img):
    """어텐션 가중 풀링 임베딩: (CLS + Σ w_p · patch_feat) 정규화
    → 자재 영역에 집중된 384차원 벡터 (판별력 향상, 프로토타입 등록용)"""
    x = _TF(pil_img.convert("RGB")).unsqueeze(0).cuda().half()
    model = model.cuda().half().eval()
    with torch.inference_mode():
        out = model(x, output_attentions=True)
    feats = out.last_hidden_state[0]         # (257, 384)
    att = out.attentions[-1][0].mean(0).float()   # 마지막 층 heads 평균
    w = F.softmax(att[0, 1:] * 3.0, dim=0)   # CLS→패치 (온도로 선명화)
    pooled = feats[0] + (feats[1:] * w[:, None]).sum(0)
    return F.normalize(pooled.float(), dim=0)


def yolo_hirescam(model, pil_img):
    """HiResCAM (grad ⊙ activation 요소별 곱 → 채널 합산)
    — 충실도 최상위 CAM 계열 (Draelos & Carin, arXiv:2011.08891, 2020/21)
    ultralytics 전처리 재사용, DetectionModel 학습모드 forward로 그라디언트 확보"""
    import cv2
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    im = model.predictor.preprocess([img])[0].float()      # (1,3,640,640)
    im = im.detach().clone().requires_grad_(True)

    acts, grads = {}, {}

    def fwd(m, i, o):
        acts["f"] = o

    def bwd(m, gi, go):
        grads["g"] = go[0]

    target = model.model.model[9]                          # yolov8n SPPF
    f = target.register_forward_hook(fwd)
    g = target.register_full_backward_hook(bwd)
    was_train = model.model.training
    model.model.train()
    try:
        out = model.model(im)                              # 학습모드: 3개 헤드 출력 리스트
        score = sum(o[..., 4:].max() for o in out)         # 클래스 점수 최대
        model.model.zero_grad()
        score.backward()
    finally:
        f.remove(); g.remove()
        model.model.train(was_train)
    if "f" not in acts or "g" not in grads:
        return None
    act, grad = acts["f"].detach(), grads["g"].detach()    # (1,C,H,W)
    cam = (grad * act).sum(1)[0]                           # HiResCAM: 요소별 곱 채널 합산
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = F.interpolate(cam[None, None], size=img.shape[:2],
                        mode="bilinear")[0, 0]
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
    pil = Image.open(img)
    h, _ = dinov2_attention_rollout(m, pil)
    overlay(pil, h).save(out)
    e = attention_weighted_embed(m, pil)
    print("저장:", out, "/ 어텐션가중 임베딩:", tuple(e.shape))
