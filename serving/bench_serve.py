#!/usr/bin/env python3
"""OpenAI 호환 서빙 벤치마크 (표준 라이브러리만 사용 — 호스트·컨테이너 어디서나 실행)

측정: TTFT(첫 토큰 지연) · TPOT(토큰 간 지연) · 요청당 tok/s · 동시성별 총 처리량 · 구조화 출력(JSON) 유효율
워크로드 3종:
  qa      : 짧은 질의응답 (일반 디코드)
  rewrite : 주어진 글을 거의 그대로 고쳐 쓰기 → 출력이 입력을 반복 → n-gram Speculative Decoding 이 유리
  prefix  : 동일한 긴 시스템 프롬프트 + 다른 질문 → 접두어 캐싱(vLLM APC / SGLang Radix) 이 유리
사용법:
  python3 bench_serve.py --label vllm --out results/serving/vllm.json
  python3 bench_serve.py --compare results/serving
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import statistics as st
import time
import urllib.request

QA = [
    "GPU 텐서코어가 무엇인지 3문장으로 설명해 줘.",
    "파이썬에서 리스트와 튜플의 차이를 설명해 줘.",
    "혼합 정밀도 학습의 장점과 주의점을 알려줘.",
    "KV 캐시가 LLM 추론 속도에 미치는 영향은?",
    "도커 볼륨과 바인드 마운트의 차이는?",
    "그래디언트 체크포인팅은 어떤 원리로 메모리를 줄이나?",
    "양자화(4비트)가 모델 품질에 주는 영향을 설명해 줘.",
    "배치 크기와 학습률의 관계를 설명해 줘.",
]
PASSAGE = (
    "def train(model, loader, optimizer, epochs):\n"
    "    for epoch in range(epochs):\n"
    "        for x, y in loader:\n"
    "            x = x.cuda()\n"
    "            y = y.cuda()\n"
    "            optimizer.zero_grad()\n"
    "            out = model(x)\n"
    "            loss = torch.nn.functional.cross_entropy(out, y)\n"
    "            loss.backward()\n"
    "            optimizer.step()\n"
    "        print('epoch', epoch, 'loss', loss.item())\n"
)
REWRITE = [
    f"아래 코드에서 .cuda() 를 .to(device, non_blocking=True) 로 바꾼 전체 코드를 그대로 출력해. 설명 없이 코드만.\n\n{PASSAGE}",
    f"아래 코드에 bf16 autocast 를 적용한 전체 코드를 출력해. 나머지는 그대로 두고 코드만.\n\n{PASSAGE}",
    f"아래 코드의 변수명 out 을 logits 로 바꾼 전체 코드를 출력해. 코드만.\n\n{PASSAGE}",
    f"아래 코드에 각 줄마다 한국어 주석을 달아 전체를 출력해.\n\n{PASSAGE}",
]
SYSTEM_LONG = (
    "당신은 GPU 서버 실습 조교입니다. 다음 규칙을 따르세요. " +
    " ".join(f"규칙 {i}: 답변은 정확하고 간결하게, 필요하면 예시 코드를 포함하며, 실습 환경(RTX 4070 SUPER 12GB, "
             f"NGC PyTorch, vLLM)을 고려합니다." for i in range(1, 41))
)
PREFIX_Q = ["torch.compile 은 언제 쓰나요?", "QLoRA 설정 예시를 보여줘.", "OOM 이 나면 무엇부터 보나요?",
            "DataLoader num_workers 는 몇이 좋나요?", "TF32 는 정확도에 영향이 있나요?", "MPS 는 무엇인가요?"]

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "vram_gb": {"type": "integer"},
        "techniques": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
        "difficulty": {"type": "string", "enum": ["쉬움", "보통", "어려움"]},
    },
    "required": ["name", "vram_gb", "techniques", "difficulty"],
    "additionalProperties": False,
}


def build_messages(kind, i):
    if kind == "qa":
        return [{"role": "user", "content": QA[i % len(QA)]}]
    if kind == "rewrite":
        return [{"role": "user", "content": REWRITE[i % len(REWRITE)]}]
    return [{"role": "system", "content": SYSTEM_LONG}, {"role": "user", "content": PREFIX_Q[i % len(PREFIX_Q)]}]


def post(base, key, payload, stream):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    return urllib.request.urlopen(req, timeout=600)


def one_stream(base, key, model, messages, max_tokens):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = None
    stamps = []
    with post(base, key, payload, True) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    stamps.append(now)
                    chunks += 1
    e2e = time.perf_counter() - t0
    out_tok = (usage or {}).get("completion_tokens") or chunks
    tpot = (stamps[-1] - stamps[0]) / max(out_tok - 1, 1) if len(stamps) > 1 else None
    return {"ttft": ttft, "e2e": e2e, "out_tokens": out_tok, "tpot": tpot,
            "prompt_tokens": (usage or {}).get("prompt_tokens")}


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo = int(k)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (k - lo)


def run_load(base, key, model, kind, conc, n, max_tokens):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda i: one_stream(base, key, model, build_messages(kind, i), max_tokens), range(n)))
    wall = time.perf_counter() - t0
    toks = sum(r["out_tokens"] for r in res)
    return {
        "kind": kind, "concurrency": conc, "requests": n, "wall_s": round(wall, 2),
        "throughput_tok_s": round(toks / wall, 1),
        "ttft_p50_ms": round(pct([r["ttft"] for r in res], .5) * 1000, 1),
        "ttft_p90_ms": round(pct([r["ttft"] for r in res], .9) * 1000, 1),
        "tpot_p50_ms": round((pct([r["tpot"] for r in res], .5) or 0) * 1000, 2),
        "per_req_tok_s_p50": round(pct([r["out_tokens"] / r["e2e"] for r in res], .5), 1),
        "avg_out_tokens": round(toks / n, 1),
    }


def structured(base, key, model, n):
    ok = 0
    lat = []
    for i in range(n):
        payload = {"model": model, "temperature": 0, "max_tokens": 200,
                   "messages": [{"role": "user", "content": f"GPU 최적화 실습 과제 #{i+1} 하나를 만들어 JSON 으로 답해."}],
                   "response_format": {"type": "json_schema", "json_schema": {"name": "lab_task", "schema": SCHEMA, "strict": True}}}
        t0 = time.perf_counter()
        try:
            with post(base, key, payload, False) as r:
                body = json.load(r)
            lat.append(time.perf_counter() - t0)
            obj = json.loads(body["choices"][0]["message"]["content"])
            if set(SCHEMA["required"]) <= set(obj) and obj["difficulty"] in SCHEMA["properties"]["difficulty"]["enum"] \
                    and isinstance(obj["vram_gb"], int) and 2 <= len(obj["techniques"]) <= 5:
                ok += 1
        except Exception as e:  # 스키마 미지원 엔진 등
            lat.append(time.perf_counter() - t0)
            print("   structured 오류:", str(e)[:120])
    return {"requests": n, "valid_json_rate": round(ok / n, 3), "latency_p50_ms": round(pct(lat, .5) * 1000, 1)}


def compare(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        d = json.load(open(f))
        for r in d.get("loads", []):
            rows.append((d["label"], r))
        if d.get("structured"):
            rows.append((d["label"], {"kind": "json", **d["structured"]}))
    if not rows:
        print("결과 없음"); return
    print(f"\n{'모드':<12}{'워크로드':<9}{'동시':>4}{'총 tok/s':>10}{'TTFT p50':>10}{'TPOT p50':>10}{'요청 tok/s':>11}")
    print("-" * 66)
    for label, r in rows:
        if r["kind"] == "json":
            print(f"{label:<12}{'json':<9}{'':>4}{'':>10}{r['latency_p50_ms']:>8.0f}ms{'':>10}  유효율 {r['valid_json_rate']*100:.0f}%")
        else:
            print(f"{label:<12}{r['kind']:<9}{r['concurrency']:>4}{r['throughput_tok_s']:>10.1f}{r['ttft_p50_ms']:>8.0f}ms"
                  f"{r['tpot_p50_ms']:>8.1f}ms{r['per_req_tok_s_p50']:>11.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--key", default=os.environ.get("SERVE_API_KEY", ""))
    p.add_argument("--model", default="auto", help="auto = /v1/models 의 첫 모델")
    p.add_argument("--label", default="run")
    p.add_argument("--concurrency", default="1,8")
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--kinds", default="qa,rewrite,prefix")
    p.add_argument("--structured", type=int, default=8)
    p.add_argument("--out")
    p.add_argument("--compare")
    a = p.parse_args()
    if a.compare:
        compare(a.compare); return

    if a.model == "auto":
        req = urllib.request.Request(a.base.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {a.key}"} if a.key else {})
        a.model = json.load(urllib.request.urlopen(req, timeout=10))["data"][0]["id"]
    print(f"[{a.label}] 모델={a.model} 워밍업…")
    one_stream(a.base, a.key, a.model, build_messages("qa", 0), 32)
    loads = []
    for kind in a.kinds.split(","):
        for c in [int(x) for x in a.concurrency.split(",")]:
            n = max(a.requests, c) if c > 1 else max(4, a.requests // 4)
            r = run_load(a.base, a.key, a.model, kind, c, n, a.max_tokens)
            loads.append(r)
            print(f"  {kind:<8} c={c:<3} 총 {r['throughput_tok_s']:>7.1f} tok/s | TTFT p50 {r['ttft_p50_ms']:>6.0f}ms"
                  f" | TPOT {r['tpot_p50_ms']:>5.1f}ms | 요청당 {r['per_req_tok_s_p50']:>5.1f} tok/s")
    s = structured(a.base, a.key, a.model, a.structured) if a.structured else None
    if s:
        print(f"  json     유효율 {s['valid_json_rate']*100:.0f}% | p50 {s['latency_p50_ms']:.0f}ms")
    res = {"label": a.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": a.model,
           "max_tokens": a.max_tokens, "loads": loads, "structured": s}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
        print("저장:", a.out)


if __name__ == "__main__":
    main()
