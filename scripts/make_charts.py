#!/usr/bin/env python3
"""
실측 데이터(results/*.json)로 보고서용 그래프 4종 생성 → assets/
  chart_unlock.png      언록 전/후 GEMM 처리량
  chart_power_sweep.png 전력 제한 스윕
  chart_clock_sweep.png 클럭 스윕 (+효율)
  chart_attention.png   FlashAttention 처리량
실행: python3 scripts/make_charts.py  (저장소 루트에서)
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

TEAL = "#0d9488"
EMERALD = "#10b981"
SLATE = "#64748b"
NAVY = "#0f172a"


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)


# ---------- 1) 언록 전/후 GEMM ----------
before = load("before.json")
after220 = load("after_220w.json")
after245 = load("after.json")

fig, ax = plt.subplots(figsize=(8, 4.6))
labels = ["FP16", "BF16"]
x = range(len(labels))
w = 0.26
vals = {
    "언록 전": [before["results"]["fp16"]["tflops"], before["results"]["bf16"]["tflops"]],
    "언록 후 220W": [after220["results"]["fp16"]["tflops"], after220["results"]["bf16"]["tflops"]],
    "언록 후 245W": [after245["results"]["fp16"]["tflops"], after245["results"]["bf16"]["tflops"]],
}
colors = [SLATE, TEAL, EMERALD]
for i, (name, v) in enumerate(vals.items()):
    bars = ax.bar([xi + (i - 1) * w for xi in x], v, w, label=name, color=colors[i])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9, color=NAVY)
ref_b = before["calibration"]["reference"]["tflops"]
ref_a = after220["calibration"]["reference"]["tflops"]
ax.set_title("언록 전/후 GEMM 처리량 (8192³, FP32 누적, 클럭 2505 MHz 고정)",
             fontsize=12, color=NAVY, pad=12)
ax.text(0.995, 0.06,
        f"참조 경로(FP32·CUDA): {ref_b:.1f} → {ref_a:.1f} TFLOPS (비율 {ref_a/ref_b:.2f}) — 조건 동등 확인",
        transform=ax.transAxes, ha="right", fontsize=8.5, color=SLATE)
ax.set_xticks(list(x), labels)
ax.set_ylabel("TFLOPS")
ax.set_ylim(0, 165)
ax.legend(frameon=False, fontsize=9)
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "chart_unlock.png"), dpi=160)
plt.close(fig)

# ---------- 2) 전력 스윕 ----------
p_files = sorted(glob.glob(f"{RESULTS}/calibration/p*.json"),
                 key=lambda p: int(os.path.basename(p)[1:-5]))
powers = [int(os.path.basename(p)[1:-5]) for p in p_files]
bf = [load(p)["results"]["bf16"]["tflops"] for p in p_files]
fp = [load(p)["results"]["fp16"]["tflops"] for p in p_files]

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(powers, bf, "o-", color=EMERALD, label="BF16", linewidth=2)
ax.plot(powers, fp, "s--", color=TEAL, label="FP16", linewidth=2)
ax.annotate("스위트 스팟\n(245W와 동급 성능)", xy=(180, bf[1]), xytext=(196, 98),
            fontsize=9, color=NAVY,
            arrowprops=dict(arrowstyle="->", color=SLATE))
ax.set_title("전력 제한별 처리량 (자연 부스트)", fontsize=12, color=NAVY, pad=12)
ax.set_xlabel("전력 제한 (W)")
ax.set_ylabel("TFLOPS")
ax.set_ylim(0, 165)
ax.legend(frameon=False, fontsize=9)
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "chart_power_sweep.png"), dpi=160)
plt.close(fig)

# ---------- 3) 클럭 스윕 ----------
c_files = sorted(glob.glob(f"{RESULTS}/calibration/c*.json"))
clocks = [load(p)["results"]["bf16"]["clock_mhz"] for p in c_files]
cbf = [load(p)["results"]["bf16"]["tflops"] for p in c_files]
ceff = [load(p)["results"]["bf16"]["efficiency"] * 100 for p in c_files]

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(clocks, cbf, "o-", color=EMERALD, linewidth=2, label="BF16")
ax.annotate("~2400 MHz에서 포화\n(2505/2700 요청 모두 수렴)", xy=(clocks[-2], cbf[-2]),
            xytext=(clocks[0] + 40, max(cbf) - 18), fontsize=9, color=NAVY,
            arrowprops=dict(arrowstyle="->", color=SLATE))
ax.set_title("클럭별 처리량 (전력 245W 고정, 효율 표기)", fontsize=12, color=NAVY, pad=12)
ax.set_xlabel("실측 SM 클럭 (MHz)")
ax.set_ylabel("BF16 TFLOPS")
for cx, cy, ce in zip(clocks, cbf, ceff):
    ax.text(cx, cy + 3, f"{ce:.0f}%", ha="center", fontsize=8, color=TEAL)
ax.set_ylim(0, 165)
ax.legend(frameon=False, fontsize=9)
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "chart_clock_sweep.png"), dpi=160)
plt.close(fig)

# ---------- 4) FlashAttention ----------
att = load("attention.json")["results"]
cases = []
for r in att:
    key = f"S{r['seqlen']}" if r["batch"] == 1 else f"B{r['batch']}·S{r['seqlen']}"
    if key not in [c[0] for c in cases]:
        cases.append([key, None, None])
    for c in cases:
        if c[0] == key:
            if r["mode"] == "fwd":
                c[1] = r["tflops"]
            else:
                c[2] = r["tflops"]

fig, ax = plt.subplots(figsize=(8, 4.6))
x = range(len(cases))
w = 0.38
fwd = [c[1] or 0 for c in cases]
bwd = [c[2] or 0 for c in cases]
b1 = ax.bar([xi - w / 2 for xi in x], fwd, w, label="fwd", color=EMERALD)
b2 = ax.bar([xi + w / 2 for xi in x], bwd, w, label="fwd+bwd", color=TEAL)
ax.axhline(71.0, color="#dc2626", linestyle=":", linewidth=1.4)
ax.text(len(cases) - 0.45, 71 + 4, "잠긴 상태 상한 ≈ 71 TFLOPS", fontsize=8.5, color="#dc2626")
for bars in (b1, b2):
    for b in bars:
        if b.get_height() > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                    f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=8, color=NAVY)
ax.set_title("FlashAttention 2 처리량 (BF16, causal)", fontsize=12, color=NAVY, pad=12)
ax.set_xticks(list(x), [c[0] for c in cases])
ax.set_ylabel("TFLOPS")
ax.set_ylim(0, 250)
ax.legend(frameon=False, fontsize=9)
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "chart_attention.png"), dpi=160)
plt.close(fig)

print("생성 완료:")
for f in ("chart_unlock.png", "chart_power_sweep.png", "chart_clock_sweep.png", "chart_attention.png"):
    print(" ", os.path.join("assets", f))
