#!/usr/bin/env python3
"""
Long-tail 대응 모듈 — 건설 자재 데이터의 클래스 불균형 처리
  · analyze      : 클래스 빈도/불균형 지수 분석
  · eirfs_weights : E-IRFS (IEEE IROS 2025) 지수형 반복 샘플링 가중치
  · cb_focal_loss : Class-Balanced Focal Loss (Cui et al., CVPR 2019)
                    — DINOv2 분류 헤드 학습용 (tail 클래스 그래디언트 보강)
※ SPGM(CAIT 2025)의 프로토타입 가이드 개념은 본 시스템의
  'DINOv2 대표벡터 등록'이 그대로 구현 — 등록 장수별 정확도 실험이 그 평가
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F


def analyze(labels):
    """labels: 클래스 인덱스 리스트 → 빈도 통계 + 불균형 지수"""
    c = Counter(labels)
    n_max = max(c.values())
    n_min = min(c.values())
    return {
        "counts": dict(sorted(c.items())),
        "imbalance_ratio": n_max / n_min if n_min else float("inf"),
        "classes": len(c),
    }


def eirfs_weights(image_labels, n_instances, gamma=0.7):
    """E-IRFS (IROS 2025): 이미지별 반복 샘플링 가중치
    image_labels: [클래스 인덱스 리스트,...] (이미지별 등장 클래스)
    n_instances:  [클래스별 인스턴스 총수]
    → w_img = exp(max_c∈img (1/sqrt(n_img_c·n_inst_c))^γ)  — rare 클래스 위주 증폭"""
    import math
    weights = []
    for classes_in_img in image_labels:
        s = max((1.0 / math.sqrt(n_instances[c]) for c in classes_in_img), default=1.0)
        weights.append(math.exp(s ** gamma))
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum()          # 확률로 정규화 (WeightedRandomSampler용)


def cb_focal_loss(logits, targets, counts, beta=0.9999, gamma=2.0):
    """Class-Balanced Focal Loss — 유효샘플수 기반 클래스 가중 + 포컬
    counts: [클래스별 샘플수]
    weight_c = (1-β)/(1-β^{n_c}), sigmoid focal CE"""
    n = torch.tensor(counts, dtype=torch.float32, device=logits.device)
    eff = (1.0 - beta) / (1.0 - beta ** n + 1e-8)
    w = eff / eff.sum() * len(n)               # 정규화된 클래스 가중치
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = w[None, :] * ((1 - pt) ** gamma) * ce
    return loss.mean()


if __name__ == "__main__":
    print("longtail 모듈 준비 — analyze()/eirfs_weights()/cb_focal_loss()")
