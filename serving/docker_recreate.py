#!/usr/bin/env python3
"""docker inspect 결과로 동일 구성의 `docker run` 명령을 재구성한다 (+ 2단계 추가 옵션).

사용법:
  python3 docker_recreate.py INSPECT.json --image NEW_IMAGE [--name NAME]
      [--volume hf_cache:/root/.cache/huggingface] [--env HF_HOME=...]
      [--cap SYS_ADMIN] [--shm 16g] [--mps-pipe /tmp/nvidia-mps] [--network gpu-lab]
      [--cpuset 0-7]
출력: 셸에서 그대로 실행 가능한 명령 (한 줄에 인자 하나, 역슬래시 연결)

이미지 설정(Env/Cmd/Entrypoint/WorkingDir/User)은 docker commit 이 이미지에 보존하므로
여기서는 HostConfig(포트·마운트·GPU·재시작 정책·IPC·ulimit 등)를 재현하는 데 집중한다.
"""
import argparse
import json
import shlex
import sys


def size_to_bytes(s):
    s = s.strip().lower()
    mult = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def build(info, a):
    hc = info.get("HostConfig", {}) or {}
    cfg = info.get("Config", {}) or {}
    args = ["docker", "run", "-d", "--name", a.name or info["Name"].lstrip("/")]

    rp = (hc.get("RestartPolicy") or {}).get("Name") or ""
    if rp and rp != "no":
        mr = (hc.get("RestartPolicy") or {}).get("MaximumRetryCount") or 0
        args += ["--restart", f"{rp}:{mr}" if rp == "on-failure" and mr else rp]

    if cfg.get("Hostname") and not cfg["Hostname"].startswith(info.get("Id", "")[:12]):
        args += ["--hostname", cfg["Hostname"]]

    # GPU
    gpus_done = False
    for dr in hc.get("DeviceRequests") or []:
        if "gpu" in [c for caps in (dr.get("Capabilities") or []) for c in caps]:
            ids = dr.get("DeviceIDs")
            cnt = dr.get("Count")
            args += ["--gpus", f'"device={",".join(ids)}"' if ids else ("all" if cnt in (-1, None) else str(cnt))]
            gpus_done = True
    if hc.get("Runtime") and hc["Runtime"] not in ("runc", ""):
        args += ["--runtime", hc["Runtime"]]
        gpus_done = True
    if not gpus_done:
        print("# 경고: 원본에 GPU 설정이 없음 → --gpus all 추가", file=sys.stderr)
        args += ["--gpus", "all"]

    # 포트
    for cport, binds in sorted((hc.get("PortBindings") or {}).items()):
        port, _, proto = cport.partition("/")
        for b in binds or [{}]:
            hip, hport = b.get("HostIp") or "", b.get("HostPort") or ""
            spec = ":".join(x for x in [hip, hport] if x)
            spec = f"{spec}:{port}" if spec else port
            if proto and proto != "tcp":
                spec += f"/{proto}"
            args += ["-p", spec]
    if hc.get("PublishAllPorts"):
        args.append("-P")

    # 마운트: bind 는 HostConfig.Binds 가 원문, volume 은 Mounts 에서
    existing_dest = set()
    for b in hc.get("Binds") or []:
        args += ["-v", b]
        parts = b.split(":")
        if len(parts) >= 2:
            existing_dest.add(parts[1])
    for m in info.get("Mounts") or []:
        if m.get("Destination") in existing_dest:
            continue
        if m.get("Type") == "volume" and m.get("Name"):
            spec = f'{m["Name"]}:{m["Destination"]}'
            if not m.get("RW", True):
                spec += ":ro"
            # 익명 볼륨(64자 해시)도 이름으로 재연결 → 데이터 유지
            args += ["-v", spec]
            existing_dest.add(m["Destination"])
        elif m.get("Type") == "bind":
            spec = f'{m["Source"]}:{m["Destination"]}' + ("" if m.get("RW", True) else ":ro")
            args += ["-v", spec]
            existing_dest.add(m["Destination"])
    for t in (hc.get("Tmpfs") or {}):
        args += ["--tmpfs", t]

    # 추가 볼륨 (HF 캐시 등)
    for v in a.volume or []:
        dest = v.split(":")[1]
        if dest in existing_dest:
            print(f"# 참고: {dest} 는 이미 마운트되어 있어 추가하지 않음", file=sys.stderr)
            continue
        args += ["-v", v]

    # 네트워크
    nm = hc.get("NetworkMode") or "default"
    if nm not in ("default", "bridge"):
        args += ["--network", nm]

    # IPC / SHM
    ipc = hc.get("IpcMode") or ""
    if a.mps_pipe:
        ipc = "host"
    if ipc and ipc not in ("private", "shareable", ""):
        args += ["--ipc", ipc]
    if ipc != "host":
        cur = hc.get("ShmSize") or 0
        want = size_to_bytes(a.shm) if a.shm else 0
        shm = max(cur, want)
        if shm and shm != 64 * 1024 ** 2:
            args += ["--shm-size", str(shm)]

    for u in hc.get("Ulimits") or []:
        args += ["--ulimit", f'{u["Name"]}={u["Soft"]}:{u["Hard"]}']

    caps = list(hc.get("CapAdd") or [])
    for c in a.cap or []:
        if c not in caps and f"CAP_{c}" not in caps:
            caps.append(c)
    for c in caps:
        args += ["--cap-add", c]
    for c in hc.get("CapDrop") or []:
        args += ["--cap-drop", c]
    for so in hc.get("SecurityOpt") or []:
        args += ["--security-opt", so]
    if hc.get("Privileged"):
        args.append("--privileged")
    for d in hc.get("Devices") or []:
        args += ["--device", f'{d["PathOnHost"]}:{d["PathInContainer"]}:{d.get("CgroupPermissions","rwm")}']
    for gh in hc.get("GroupAdd") or []:
        args += ["--group-add", gh]
    for h in hc.get("ExtraHosts") or []:
        args += ["--add-host", h]

    cpuset = a.cpuset if a.cpuset is not None else (hc.get("CpusetCpus") or "")
    if cpuset:
        args += ["--cpuset-cpus", cpuset]
    if hc.get("Memory"):
        args += ["--memory", str(hc["Memory"])]
    if hc.get("NanoCpus"):
        args += ["--cpus", str(hc["NanoCpus"] / 1e9)]

    # 라벨 (compose 내부 라벨 제외)
    for k, v in sorted((cfg.get("Labels") or {}).items()):
        if not k.startswith(("com.docker.compose", "org.opencontainers")):
            args += ["--label", f"{k}={v}"]

    # MPS 파이프 공유
    if a.mps_pipe:
        args += ["-v", f"{a.mps_pipe}:{a.mps_pipe}", "-e", f"CUDA_MPS_PIPE_DIRECTORY={a.mps_pipe}"]

    for e in a.env or []:
        args += ["-e", e]

    args.append(a.image)
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inspect")
    p.add_argument("--image", required=True)
    p.add_argument("--name")
    p.add_argument("--volume", action="append")
    p.add_argument("--env", action="append")
    p.add_argument("--cap", action="append")
    p.add_argument("--shm")
    p.add_argument("--mps-pipe")
    p.add_argument("--cpuset")
    a = p.parse_args()
    data = json.load(open(a.inspect))
    info = data[0] if isinstance(data, list) else data
    args = build(info, a)
    out = []
    for x in args:
        # --gpus "device=0,1" 은 이미 따옴표 포함
        out.append(x if x.startswith('"') else shlex.quote(x))
    print(" \\\n  ".join(out))
    extra = [n for n in (info.get("NetworkSettings", {}).get("Networks") or {})
             if n not in ((info.get("HostConfig") or {}).get("NetworkMode"), "bridge")]
    for n in extra:
        print(f"# 추가 네트워크: docker network connect {shlex.quote(n)} {shlex.quote(a.name or info['Name'].lstrip('/'))}", file=sys.stderr)
        print(f"#NETCONNECT {n}")


if __name__ == "__main__":
    main()
