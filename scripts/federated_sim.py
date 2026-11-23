#!/usr/bin/env python3
"""
대표벡터 교환형 LoRa 페더레이티드 러닝 시뮬레이터
  · N개 적치장 노드: 로컬 관측(클래스당 소수 샷)으로 로컬 대표벡터 갱신
  · 업링크: 대표벡터(수백B)만 / 다운링크: 병합 벡터 브로드캐스트
  · vs FedAvg(모델 가중치 교환) 패킷량 비교 + 수렴 곡선
  → 제안서/논문 Figure용 선검증 (데이터 없이 합성으로)
"""
import numpy as np

DIM = 384
BYTES_VEC = DIM * 4            # fp32 벡터 1개 = 1536B
FEDAVG_PARAMS = 1_200_000      # YOLOv8n 파라미터 수 (가중치 교환 시)


def make_world(n_nodes, n_classes, seed=0):
    """합성 세계: 클래스별 전역 프로토타입 + 노드별 관측 편향"""
    rng = np.random.default_rng(seed)
    true_protos = rng.normal(0, 1, (n_classes, DIM))
    true_protos /= np.linalg.norm(true_protos, axis=1, keepdims=True)
    nodes = []
    for i in range(n_nodes):
        obs = {}
        for c in range(n_classes):
            k = rng.integers(1, 11)                    # 클래스당 1~10샷 (롱테일)
            pts = true_protos[c] + rng.normal(0, 0.4, (k, DIM))
            obs[c] = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        nodes.append(obs)
    return true_protos, nodes


def local_update(node_obs, global_protos, alpha=0.5):
    """로컬 대표벡터 갱신 (로컬 평균과 전역 벡터의 지수 이동 평균)"""
    new = {}
    for c, pts in node_obs.items():
        local = pts.mean(0)
        if global_protos is not None:
            local = alpha * local + (1 - alpha) * global_protos[c]
        new[c] = local / (np.linalg.norm(local) + 1e-8)
    return new


def accuracy(true_protos, protos):
    """전역 프로토타입의 클래스 분류 정확도 (합성 테스트셋)"""
    correct = total = 0
    for c in range(len(true_protos)):
        for _ in range(20):
            x = true_protos[c] + np.random.default_rng().normal(0, 0.3, DIM)
            x /= np.linalg.norm(x)
            pred = np.argmax(x @ np.array(list(protos.values())).T)
            correct += pred == c
            total += 1
    return correct / total


def simulate(n_nodes=4, n_classes=3, rounds=30, seed=0):
    true_p, nodes = make_world(n_nodes, n_classes, seed)
    local_p = [local_update(n, None) for n in nodes]          # 초기 로컬
    global_p = {c: np.mean([lp[c] for lp in local_p], axis=0) for c in range(n_classes)}
    for c in global_p:
        global_p[c] /= np.linalg.norm(global_p[c]) + 1e-8
    hist = []
    uplink_B = n_nodes * n_classes * BYTES_VEC                # 라운드당 업링크
    downlink_B = n_classes * BYTES_VEC
    for r in range(rounds):
        local_p = [local_update(nodes[i], global_p) for i in range(n_nodes)]
        global_p = {c: np.mean([lp[c] for lp in local_p], axis=0) for c in range(n_classes)}
        for c in global_p:
            global_p[c] /= np.linalg.norm(global_p[c]) + 1e-8
        hist.append(accuracy(true_p, global_p))
    fedavg_B = n_nodes * FEDAVG_PARAMS * 4                    # FedAvg 라운드당
    return hist, uplink_B + downlink_B, fedavg_B


if __name__ == "__main__":
    hist, ours_B, fedavg_B = simulate()
    print(f"대표벡터 교환형 LoRa-FL:")
    print(f"  수렴: 5라운드 {hist[4]*100:.1f}% → 15라운드 {hist[14]*100:.1f}% → "
          f"30라운드 {hist[-1]*100:.1f}%")
    print(f"  라운드당 통신량: {ours_B/1024:.1f} KB vs FedAvg(가중치) "
          f"{fedavg_B/1024/1024:.1f} MB ({fedavg_B/ours_B:,.0f}배 절감)")
