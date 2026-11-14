"""Where the few-shot rerun is, and when each stage should end, in KST.

    python scripts/ops/rerun_status.py

Read-only: parses campaign.log and the stage logs of scripts/ops/run_fewshot10.sh.
A stage's duration comes from, in order: its own progress so far, the mean of the
finished stages of that kind, the rates measured by the tensor-core gate, and the
planning estimates below. Before the run starts, its start is taken from the end of
Improving-CA-LoRA's campaign, which holds the GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[2]
RUN, RUN_C1 = "sdxl_cityscapes_fewshot10", "sdxl_c1_fewshot10"
SIBLING = Path("/workspace/Improving-CA-LoRA")

FOSTERS = [(RUN, e) for e in ("n1_i0", "n4_i0", "n3_i0", "n5_i1", "n5_i0", "n4_i1", "n4_i2", "n4_i3")] + [
    (RUN_C1, e) for e in ("n4_i0", "n4_i1", "n4_i2", "n4_i3")
]
ARMS = ("no1", "no2", "no3", "c1", "no4", "no5", "no6", "no7", "no8", "no9", "c2n2", "c2")
HARD_ARMS = {"no1", "no2", "c1", "c2n2"}
#: Seconds, used until the gate or a finished stage says otherwise.
FALLBACK = {
    "gate": 20 * 60.0,
    "foster": 17 * 60.0,
    "share": 60.0,
    "hard": 90.0,
    "router": 27 * 60.0,
    "paired_loss": 18 * 60.0,
    "generate": 84 * 60.0,
    "evaluate": 3.5 * 60.0,
}
STEP_RE = re.compile(r"step (\d+)/(\d+)\b")
IMAGE_RE = re.compile(r"generate_images \|\s+(\d+)/(\d+)\s*$")
PROBLEM_RE = re.compile(r"Traceback|out of memory|OutOfMemoryError|loss=nan|loss=inf|"
                        r"Input/output error|Errno 5\b|QUEUE ABORTED")
STALE_AFTER = 600.0


@dataclass
class StageState:
    status: str = "pending"  # pending | running | done | skipped
    started: datetime | None = None
    ended: datetime | None = None
    attempts: int = 0


@dataclass
class Campaign:
    stages: dict[str, StageState] = field(default_factory=dict)
    halted: str | None = None
    started: bool = False
    finished: bool = False
    retries: list[str] = field(default_factory=list)


def plan() -> list[str]:
    """Every stage in the order the runner executes them."""
    return [
        *(f"foster:{run}/{expert}" for run, expert in FOSTERS),
        "share",
        *(f"assemble:{arm}" for arm in ARMS),
        "paired_loss",
        *(f"{kind}:{arm}" for arm in ARMS for kind in ("generate", "evaluate")),
    ]


def kind_of(stage: str) -> str:
    head, _, tail = stage.partition(":")
    if head == "assemble":
        return "hard" if tail in HARD_ARMS else "router"
    return head


def parse_campaign(text: str) -> Campaign:
    campaign = Campaign()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        when = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        event = parts[1]
        if event == "HALT":
            campaign.halted = " ".join(parts[2:])
        elif event == "CAMPAIGN":
            campaign.started = True
            campaign.finished = campaign.finished or parts[2] == "END"
        elif event in ("START", "END", "SKIP", "RETRY"):
            state = campaign.stages.setdefault(parts[2], StageState())
            if event == "START":
                state.status, state.ended = "running", None
                state.attempts += 1
                state.started = state.started or when
            elif event == "END":
                state.ended = when
                if parts[3] == "rc=0":
                    state.status = "done"
            elif event == "SKIP":
                state.status = "skipped"
            else:
                campaign.retries.append(" ".join(parts[2:]))
    return campaign


def stage_log(stage: str, runs: Path) -> Path:
    head, _, tail = stage.partition(":")
    if head == "foster":
        run, expert = tail.split("/")
        return runs / run / "logs" / f"foster_{expert}.out"
    if head in ("assemble", "generate", "evaluate"):
        run = RUN_C1 if tail == "c1" else RUN
        prefix = {"assemble": "assemble", "generate": "gen", "evaluate": "eval"}[head]
        return runs / run / "logs" / f"{prefix}_{tail}.out"
    if head == "paired_loss":
        return runs / RUN / "logs" / "paired_loss.out"
    return runs / RUN / "logs" / f"{head}.out"


def progress(stage: str, text: str) -> float | None:
    """Fraction of a running stage done, from the last progress line of its log."""
    kind = kind_of(stage)
    pattern = STEP_RE if kind in ("foster", "router") else IMAGE_RE if kind == "generate" else None
    if pattern is None:
        return None
    found = None
    for line in text.replace("\r", "\n").splitlines():
        match = pattern.search(line)
        if match:
            found = match
    return int(found.group(1)) / int(found.group(2)) if found else None


def gate_estimates(runs: Path) -> dict[str, float]:
    path = runs / f"{RUN}_gate" / "tensor_core_gate.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("estimates", {})


def project(campaign: Campaign, runs: Path, now: datetime, start: datetime) -> dict[str, datetime]:
    """Expected end of every unfinished stage, walking the runner's order from ``start``."""
    defaults = {**FALLBACK, **gate_estimates(runs)}
    finished: dict[str, list[float]] = {}
    for stage, state in campaign.stages.items():
        if state.status == "done" and state.started and state.ended:
            finished.setdefault(kind_of(stage), []).append((state.ended - state.started).total_seconds())

    def typical(kind: str) -> float:
        done = finished.get(kind)
        return sum(done) / len(done) if done else defaults[kind]

    cursor, ends = max(start, now), {}
    for stage in plan():
        state = campaign.stages.get(stage, StageState())
        if state.status in ("done", "skipped"):
            continue
        remaining = typical(kind_of(stage))
        if state.status == "running" and state.started:
            elapsed = (now - state.started).total_seconds()
            log = stage_log(stage, runs)
            fraction = progress(stage, log.read_text(errors="replace")) if log.exists() else None
            remaining = elapsed / fraction - elapsed if fraction else max(remaining - elapsed, 0.0)
        cursor += timedelta(seconds=remaining)
        ends[stage] = cursor
    return ends


def _shell(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=cwd).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(failed: {exc})"


def sibling_end(now: datetime) -> tuple[datetime | None, str]:
    """When the campaign holding the GPU should end, from its own status tool."""
    tool = SIBLING / "scripts" / "ops" / "campaign_status.py"
    if not tool.exists():
        return None, "상태 도구 없음"
    text = _shell(["/workspace/venv-calora/bin/python", str(tool)], cwd=SIBLING)
    match = re.search(r"campaign end: (\d+)/(\d+) (\d+):(\d+) KST", text)
    if "campaign end: finished" in text:
        return now, "캠페인 로그상 종료"
    if not match:
        return None, "종료 시각을 읽지 못함"
    month, day, hour, minute = map(int, match.groups())
    end = datetime(now.astimezone(KST).year, month, day, hour, minute, tzinfo=KST)
    problems = re.search(r"problems: (.*)", text)
    return end, f"문제: {problems.group(1) if problems else '확인 불가'}"


def problems(campaign: Campaign, runs: Path) -> list[str]:
    found = []
    if campaign.halted:
        found.append(f"HALT: {campaign.halted}")
    found += [f"재시도: {entry}" for entry in campaign.retries]
    for run in (RUN, RUN_C1):
        for log in sorted((runs / run / "logs").glob("*")):
            if not log.is_file():
                continue
            hits = PROBLEM_RE.findall(log.read_text(errors="replace"))
            if hits:
                found.append(f"{log.name}: {len(hits)}건, 예: {hits[0]}")
    for stage, state in campaign.stages.items():
        if state.status == "running":
            log = stage_log(stage, runs)
            age = time.time() - log.stat().st_mtime if log.exists() else None
            if age is not None and age > STALE_AFTER:
                found.append(f"{stage}: 로그가 {age / 60:.0f}분째 멈춤")
    stuck = [line for line in _shell(["ps", "-eo", "stat,etime,cmd"]).splitlines()
             if line.startswith("D") and "CA-LoRA-TSM" in line]
    found += [f"D 상태: {line[:120]}" for line in stuck]
    return found


def render(campaign: Campaign, ends: dict[str, datetime], runs: Path, now: datetime, note: str = "") -> str:
    def kst(moment: datetime) -> str:
        return moment.astimezone(KST).strftime("%m/%d %H:%M")

    def cell(stage: str) -> str:
        state = campaign.stages.get(stage, StageState())
        if state.status in ("done", "skipped"):
            return "완료"
        return kst(ends[stage]) + (" (진행 중)" if state.status == "running" else "")

    lines = [f"기준 시각: {kst(now)} KST"]
    if note:
        lines.append(note)
    fosters = [f"foster:{run}/{expert}" for run, expert in FOSTERS]
    left = [stage for stage in fosters if stage in ends]
    done = len(fosters) - len(left)
    lines.append(f"expert 학습: {done}/{len(fosters)} 완료"
                 + (f", 예상 종료 {kst(ends[left[-1]])}" if left else ""))
    lines += ["", "| arm | 학습 (번들) | 추론 | 평가 |", "|---|---|---|---|"]
    for arm in ARMS:
        lines.append(f"| {arm} | {cell(f'assemble:{arm}')} | {cell(f'generate:{arm}')} | {cell(f'evaluate:{arm}')} |")
    lines.append("")
    lines.append(f"전체 예상 종료: {'완료' if campaign.finished else kst(max(ends.values())) + ' KST'}")
    for stage, state in campaign.stages.items():
        if state.status == "running":
            log = stage_log(stage, runs)
            fraction = progress(stage, log.read_text(errors="replace")) if log.exists() else None
            lines.append(f"진행 중: {stage} " + (f"{100 * fraction:.1f}%" if fraction is not None else "(진행률 없음)"))
    workers = [line[:80] for line in _shell(["ps", "-eo", "stat,etime,cmd"]).splitlines()
               if "catsm.cli" in line or "scripts/generate_images.py" in line or "scripts/evaluate.py" in line
               or "scripts/paired_loss.py" in line]
    lines.append(f"프로세스: {workers or '없음'}")
    lines.append("GPU: " + _shell(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                   "--format=csv,noheader"]))
    found = problems(campaign, runs)
    lines.append("문제: " + ("; ".join(found) if found else "없음"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    log = args.runs / RUN / "campaign.log"
    campaign = parse_campaign(log.read_text() if log.exists() else "")
    now = datetime.now(timezone.utc)
    start, note = now, ""
    if not campaign.stages:
        end, detail = sibling_end(now)
        gate = gate_estimates(args.runs) and 0.0 or FALLBACK["gate"]
        start = (end or now) + timedelta(seconds=gate)
        when = end.astimezone(KST).strftime("%m/%d %H:%M") if end else "불명"
        note = f"GPU 대기 중: 다른 세션 캠페인 예상 종료 {when} KST ({detail}). 종료 신호를 받으면 게이트부터 시작."
    print(render(campaign, project(campaign, args.runs, now, start), args.runs, now, note))


if __name__ == "__main__":
    main()
