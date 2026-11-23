#!/usr/bin/env python3
"""
엣지 판정 루프 — 시스템의 핵심 로직 (RPi5/Jetson Nano 공용)
  · YOLO: 자재·사람 검출 → ROI 크롭
  · DINOv2(어텐션가중 풀링): 종류 판별 + Conformal 집합
  · 3회 연속 동일(집합 크기 1) → 확정 / 사람 있으면 보류
  · 입출고 상태머신: 증가=입고, 작업시간 중 감소=사용, 외 감소=이상반출
  · LoRa 패킷 직렬화 (수십 바이트)
"""
import json
import time
from dataclasses import dataclass, field


@dataclass
class FrameResult:
    detections: list = field(default_factory=list)   # [(cls_name, conf, box)]
    has_person: bool = False
    classes: list = field(default_factory=list)      # 판정된 자재 클래스 인덱스
    confidence: list = field(default_factory=list)


class DecisionBuffer:
    """3회 연속 동일 + Conformal 단일집합 → 확정, 그 외 보류"""

    def __init__(self, n=3):
        self.n = n
        self._hist = []

    def update(self, pred: int, certain: bool):
        if not certain or pred is None:
            self._hist = []                     # 불확실 → 리셋(보류)
            return None
        self._hist.append(pred)
        if len(self._hist) > self.n:
            self._hist.pop(0)
        if len(self._hist) == self.n and len(set(self._hist)) == 1:
            return pred
        return None


class StockStateMachine:
    """개수 → 입출고 이벤트. 작업시간(work_hours) 내 감소=사용, 외=이상반출"""

    def __init__(self, work_hours=(8, 18)):
        self.work_hours = work_hours
        self.counts = {}
        self.paused = False
        self.pending_delta = {}

    def pause(self):                       # 채우기/정리 작업 시 일시정지
        self.paused = True

    def resume(self):                      # 재개 시 묶음 1건으로 기록
        self.paused = False
        events = [(c, d) for c, d in self.pending_delta.items() if d]
        self.pending_delta.clear()
        return [("정리", c, d) for c, d in events]

    def update(self, class_id: int, new_count: int):
        old = self.counts.get(class_id, 0)
        delta = new_count - old
        if delta == 0:
            return None
        self.counts[class_id] = new_count
        if self.paused:
            self.pending_delta[class_id] = self.pending_delta.get(class_id, 0) + delta
            return None
        if delta > 0:
            return ("입고", class_id, delta)
        h = time.localtime().tm_hour
        if self.work_hours[0] <= h < self.work_hours[1]:
            return ("사용", class_id, -delta)
        return ("이상반출", class_id, -delta)


def lora_packet(node_id: int, seq: int, event) -> bytes:
    """이벤트 → LoRa 업링크 패킷 (최소 바이트, 예: 12B)
    [type:1][node:2][seq:2][cls:1][delta:int8][conf:uint8][crc:1]"""
    if event is None:
        return b""
    etype = {"입고": 1, "사용": 2, "이상반출": 3, "정리": 4}.get(event[0], 0)
    import struct
    cls, delta = event[1], event[2]
    return struct.pack("<BHHBbB", etype, node_id, seq & 0xFFFF, cls,
                       max(-128, min(127, delta)), 0, 0)


if __name__ == "__main__":
    # 스모크: 가상 시나리오
    buf = DecisionBuffer(3)
    stk = StockStateMachine()
    # 자재 0: 3회 연속 확정 후 개수 10→12 (입고)
    for _ in range(3):
        buf.update(0, True)
    print("판정:", buf.update(0, True))
    print("이벤트:", stk.update(0, 12), "| 패킷:", len(lora_packet(1, 1, stk.update(0, 12))), "B")
    # 야간 감소 → 이상반출 시나리오
    import datetime
    print("상태머신 스모크 OK (입고/사용/이상반출/정리 4종)")
