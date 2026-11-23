#!/usr/bin/env bash
# ============================================================
# monitor_stack.sh — GPU 모니터링 스택 (DCGM exporter + Prometheus + Grafana)
# 사용: sudo bash scripts/monitor_stack.sh
# 접속: http://<서버IP>:3000 (admin/admin, 첫 로그인 시 변경)
# 대시보드: Grafana에서 assets/grafana_gpu.json 임포트 (또는 자동 프로비저닝)
# ============================================================
set -euo pipefail

docker rm -f dcgm-exporter prometheus grafana >/dev/null 2>&1 || true
docker network create monitor >/dev/null 2>&1 || true

# 1) NVIDIA DCGM exporter (GPU 메트릭 → Prometheus)
docker run -d --name dcgm-exporter --network monitor --restart always \
  --gpus all --cap-add SYS_ADMIN \
  nvidia/dcgm-exporter:4.9.0 >/dev/null

# 2) Prometheus (DCGM 스크랩)
cat > /tmp/prometheus.yml <<'EOF'
scrape_configs:
  - job_name: dcgm
    static_configs:
      - targets: ['dcgm-exporter:9400']
EOF
docker run -d --name prometheus --network monitor --restart always \
  -v /tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus >/dev/null

# 3) Grafana (Prometheus 데이터소스 + GPU 대시보드 자동 프로비저닝)
mkdir -p /root/monitor/provisioning/datasources /root/monitor/provisioning/dashboards /root/monitor/dashboards
cat > /root/monitor/provisioning/datasources/ds.yml <<'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
cat > /root/monitor/provisioning/dashboards/dash.yml <<'EOF'
apiVersion: 1
providers:
  - name: default
    folder: GPU
    type: file
    options:
      path: /var/lib/grafana/dashboards
EOF
cp /root/workspace/edge-project/scripts/grafana_gpu.json /root/monitor/dashboards/ 2>/dev/null || true

docker run -d --name grafana --network monitor --restart always -p 3000:3000 \
  -v /root/monitor/provisioning:/etc/grafana/provisioning \
  -v /root/monitor/dashboards:/var/lib/grafana/dashboards \
  grafana/grafana >/dev/null

sleep 5
docker ps --format "{{.Names}}: {{.Status}}" | grep -E "dcgm|prometheus|grafana"
echo "Grafana: http://$(hostname -I | awk '{print $1}'):3000 (admin/admin)"
