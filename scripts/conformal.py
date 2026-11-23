#!/usr/bin/env python3
"""
Conformal Prediction 판정 모듈 — 신뢰도 보장 예측 (교정 후 분류)
근거: 2025-2026 CVPR/NeurIPS 다수 (보정·불확실성 정량화 표준 기법)
동작:
  calibrate : 교정셋 점수 분포로 임계값 q 산출 → 예측 집합 크기로 신뢰도 보장
  predict   : 유사도 → p-value → 집합. 빈 집합이면 "판정 보류" (모른다)
  edge loop : "3회 연속 동일 + 집합 크기 1" 일 때만 확정 — 불확실성 정식화
"""
import numpy as np


class ConformalClassifier:
    """스코어 = 1 - 최대클래스 유사도 (적합성 점수, 작을수록 확신)"""

    def __init__(self, alpha=0.05):
        self.alpha = alpha          # 목표 오류율 (기본 5%)
        self.q = None
        self.protos = None

    def calibrate(self, feats, labels, protos):
        """feats[N,384], labels[N], protos[C,384] → q 산출 (교정 1회)"""
        self.protos = protos
        sim = feats @ protos.T
        scores = 1 - sim.max(axis=1).values          # 비적합 점수
        n = len(labels)
        q_idx = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.q = np.sort(scores)[min(q_idx, n - 1)]
        return self.q

    def predict(self, feats):
        """feats[K,384] → (집합 리스트, p-value). 집합 크기 1이면 확정"""
        if self.q is None:
            raise RuntimeError("먼저 calibrate() 수행 필요")
        sim = feats @ self.protos.T
        pvals = 1 - sim
        sets = [(pvals[i] <= self.q).nonzero()[0].tolist() for i in range(len(feats))]
        return sets, pvals

    def decide(self, feats, n_confirm=3):
        """엣지 루프용: 3회 연속 단일원소 집합 + 동일 클래스 → 확정, 아니면 보류"""
        # 상태 유지는 호출 측에서 (연속 카운터) — 여기선 1회 판정 + 가이드만
        sets, _ = self.predict(feats)
        result = []
        for s in sets:
            if len(s) == 0:
                result.append(("보류", None, "집합 없음(모른다)"))
            elif len(s) == 1:
                result.append(("후보", s[0], "단일 후보 — 연속 %d회 확인 필요" % n_confirm))
            else:
                result.append(("보류", s, "다중 후보(유사 자재 혼동)"))
        return result


if __name__ == "__main__":
    print("ConformalClassifier 준비됨 — 교정셋 확보 후 calibrate() → predict()")
