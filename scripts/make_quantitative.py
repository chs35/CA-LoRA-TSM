"""Regenerate docs/QUANTITATIVE.md from the raw run outputs (stage 6).

    python scripts/make_quantitative.py            # writes docs/QUANTITATIVE.md
    python scripts/make_quantitative.py --out -    # print to stdout

Every number in the document is read from a JSON file under ``runs/`` (or the upstream
``Optimizing-CA-LoRA`` eval.json files for the base control and the 10K reference rows); nothing is
typed by hand. Interpretation lives in docs/RESULTS.md, not here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/sdxl_cityscapes"
C1 = ROOT / "runs/sdxl_c1"
UPSTREAM = Path("/workspace/Optimizing-CA-LoRA/runs/sdxl_cityscapes")
WEATHER = ["foggy", "night-time", "rainy", "snowy"]

# (label, eval.json, description)
ARMS = [
    ("base", UPSTREAM / "control_base/eval.json", "base SDXL (0%), 상위 저장소 재사용"),
    ("No.1", RUN / "no1/eval.json", "n1 (t=81, 전 구간, 4K import)"),
    ("No.2", RUN / "no2/eval.json", "n4 하드 라우팅"),
    ("No.3", RUN / "no3/eval.json", "n1+n4, core n4, 라우터"),
    ("No.4", RUN / "no4/eval.json", "n1+n3+n4, core n4, 라우터"),
    ("No.5", RUN / "no5/eval.json", "n1+n4+n5, core n4, 라우터"),
    ("No.6", RUN / "no6/eval.json", "n3+n4+n5, core n4, 라우터"),
    ("No.7", RUN / "no7/eval.json", "n1+n4+n5, core n5, 라우터"),
    ("No.8", RUN / "no8/eval.json", "n1+n3+n4+n5, core n4, 라우터"),
    ("No.9", RUN / "no9/eval.json", "n1+n3+n4+n5, core n5, 라우터"),
    ("C1", C1 / "c1/eval.json", "n4 구간, 선택 t=81 고정, 하드"),
    ("C2-n2", RUN / "c2n2/eval.json", "No.2 − n4_i3 ([500,999]=base), 하드"),
    ("C2-n3", RUN / "c2/eval.json", "No.3 − n4_i3 ([500,999]=n1_i0), 라우터 재학습"),
]
REFS = [
    ("A t=81 style", "A_t81_style"), ("B top-3 style", "B_top3_style"), ("C t=41 style", "C_t41_style"),
    ("D t=1 style", "D_t1_style"), ("E_all full LoRA", "E_all"), ("E_random 2%", "E_random"),
]
# (label, left arm, right arm, what differs)
CONTRASTS = [
    ("선택 집합 (구간별 vs t=81 고정)", "No.2", "C1", "구간·step·seed 동일, 선택만 다름"),
    ("구간별 학습 (TSM) 단독", "C1", "No.1", "선택 t=81 동일, 4 × 4K 구간별 vs 1 × 4K 전 구간"),
    ("[500,999] expert 제거 (하드)", "C2-n2", "No.2", "n4_i3 비활성, 그 구간은 base"),
    ("[500,999] expert 제거 (라우터)", "C2-n3", "No.3", "n4_i3 비활성, 그 구간은 n1_i0"),
    ("라우터로 n1 추가", "No.3", "No.2", "No.2 + context n1"),
    ("라우터로 n4 추가", "No.3", "No.1", "No.1 + core n4"),
    ("core n5 vs n4 (3 set)", "No.7", "No.5", "sets 동일"),
    ("core n5 vs n4 (4 set)", "No.9", "No.8", "sets 동일"),
    ("n1 제외", "No.6", "No.8", "n3+n4+n5 vs n1+n3+n4+n5"),
]
BUNDLES = [
    ("No.3", "n1+n4-core-n4"), ("No.4", "n1+n3+n4-core-n4"), ("No.5", "n1+n4+n5-core-n4"),
    ("No.6", "n3+n4+n5-core-n4"), ("No.7", "n1+n4+n5-core-n5"), ("No.8", "n1+n3+n4+n5-core-n4"),
    ("No.9", "n1+n3+n4+n5-core-n5"), ("C2-n3", "n1+n4-core-n4-disable-n4_i3"),
]
PAIRED_INTERVALS = [(0, 61), (62, 124), (0, 124), (125, 249), (0, 249), (250, 499), (500, 999), (0, 999)]
PAIRED_OWN = {(0, 61): "n5_i0", (62, 124): "n5_i1", (0, 124): "n4_i0", (125, 249): "n4_i1", (0, 249): "n3_i0",
              (250, 499): "n4_i2", (500, 999): "n4_i3", (0, 999): "n1_i0"}
PAIRED_C1 = {(0, 124): "n4_i0", (125, 249): "n4_i1", (250, 499): "n4_i2", (500, 999): "n4_i3"}


def load(path: Path):
    return json.loads(Path(path).read_text())


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *["| " + " | ".join(r) + " |" for r in rows], ""]


def f(x, nd=3, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def metrics(d: dict) -> dict:
    pc = d["clip_score"]["per_condition"]
    m = {"cmmd": d["cmmd"]["cmmd"], "in": pc["in_domain"]["clip_score"], "weather": d["clip_score"]["weather_mean"],
         "overall": d["clip_score"]["overall"]["clip_score"], "n": d["num_images"]}
    for c, v in pc.items():
        m[c] = v["clip_score"]
        m[c + "_sem"] = v["clip_score_std"] / math.sqrt(v["num_pairs"])
        m[c + "_std"] = v["clip_score_std"]
    # weather mean = mean of 4 condition means -> SEM by propagation
    m["weather_sem"] = math.sqrt(sum(m[c + "_sem"] ** 2 for c in WEATHER)) / len(WEATHER)
    return m


def section_eval(out: list[str]) -> dict:
    M = {label: metrics(load(path)) for label, path, _ in ARMS}
    desc = {label: d for label, _, d in ARMS}
    base, no1 = M["base"], M["No.1"]

    out += ["## 1. 평가 지표 (5단계)", "",
            "출처: 각 arm의 `eval.json`. CMMD는 in_domain 500장 vs Cityscapes val 500장, CLIP-Score는 프롬프트-이미지 쌍"
            "(in_domain 500, 날씨 조건당 100). weather CLIP = 4조건 평균의 평균.", "",
            "### 1.1 주 결과", ""]
    out += table(["구성", "설명", "CMMD ↓", "CLIP in", "CLIP weather ↑", *WEATHER, "n"],
                 [[k, desc[k], f(m["cmmd"]), f(m["in"], 2), f(m["weather"], 2), *[f(m[c], 2) for c in WEATHER], str(m["n"])]
                  for k, m in M.items()])

    out += ["### 1.2 순위", ""]
    arms = [k for k in M if k != "base"]
    by_c = sorted(arms, key=lambda k: M[k]["cmmd"])
    by_w = sorted(arms, key=lambda k: -M[k]["weather"])
    out += table(["순위", "CMMD ↓", "값", "weather CLIP ↑", "값"],
                 [[str(i + 1), by_c[i], f(M[by_c[i]]["cmmd"]), by_w[i], f(M[by_w[i]]["weather"], 2)] for i in range(len(arms))])
    pareto = [k for k in arms if not any(M[o]["cmmd"] <= M[k]["cmmd"] and M[o]["weather"] >= M[k]["weather"] and o != k
                                          and (M[o]["cmmd"] < M[k]["cmmd"] or M[o]["weather"] > M[k]["weather"]) for o in arms)]
    out += [f"(CMMD, weather CLIP) 2축 Pareto 비지배 구성: **{', '.join(sorted(pareto, key=lambda k: M[k]['cmmd']))}**. "
            "나머지는 이 중 적어도 하나에 두 축 모두에서 진다.", ""]

    out += ["### 1.3 base 및 No.1 대비", "",
            "정렬 이득 = (base CMMD − arm CMMD) / base CMMD. 제어성 보존 = arm weather CLIP / base weather CLIP.", ""]
    out += table(["구성", "ΔCMMD vs base", "정렬 이득", "Δweather vs base", "제어성 보존", "ΔCMMD vs No.1", "Δweather vs No.1", "Δin vs No.1"],
                 [[k, f(m["cmmd"] - base["cmmd"], 3, True), f"{100 * (base['cmmd'] - m['cmmd']) / base['cmmd']:.1f}%",
                   f(m["weather"] - base["weather"], 2, True), f"{100 * m['weather'] / base['weather']:.1f}%",
                   f(m["cmmd"] - no1["cmmd"], 3, True), f(m["weather"] - no1["weather"], 2, True), f(m["in"] - no1["in"], 2, True)]
                  for k, m in M.items() if k != "base"])

    out += ["### 1.4 CLIP-Score 표본 오차", "",
            "이미지 간 표준편차에서 계산한 평균의 표준오차(SEM = std/√n). 생성 seed 간 분산이 아니라 **한 seed 안의 프롬프트 간 분산**이다. "
            "arm들은 같은 프롬프트·같은 초기 노이즈를 공유하므로(짝지은 설계) arm 간 차이의 실제 오차는 이보다 작을 가능성이 높다. "
            "CMMD는 집합 통계라 오차 추정이 없다.", ""]
    out += table(["구성", "in (std)", "in SEM", "weather SEM", *[f"{c} SEM" for c in WEATHER]],
                 [[k, f"{m['in']:.2f} ({m['in_domain_std']:.2f})", f(m["in_domain_sem"]), f(m["weather_sem"]),
                   *[f(m[c + "_sem"]) for c in WEATHER]] for k, m in M.items()])

    out += ["### 1.5 통제 대조", "",
            "Δ = 왼쪽 − 오른쪽. z는 weather CLIP 차이를 두 arm의 SEM 합성(독립 가정, 보수적)으로 나눈 값.", ""]
    rows = []
    for name, a, b, note in CONTRASTS:
        A, B = M[a], M[b]
        dw = A["weather"] - B["weather"]
        z = dw / math.hypot(A["weather_sem"], B["weather_sem"])
        rows.append([name, f"{a} − {b}", note, f(A["cmmd"] - B["cmmd"], 3, True), f"{100 * (A['cmmd'] / B['cmmd'] - 1):+.1f}%",
                     f(A["in"] - B["in"], 2, True), f(dw, 2, True), f(z, 1, True),
                     *[f(A[c] - B[c], 2, True) for c in WEATHER]])
    out += table(["대조", "arm", "차이", "ΔCMMD", "ΔCMMD %", "Δin", "Δweather", "z", *[f"Δ{c}" for c in WEATHER]], rows)

    routers = ["No.3", "No.4", "No.5", "No.7", "No.8", "No.9"]
    out += ["### 1.6 n1 포함 라우터 구성군의 퍼짐", ""]
    rows = []
    for key, nd in (("cmmd", 3), ("in", 2), ("weather", 2), *[(c, 2) for c in WEATHER]):
        v = [M[k][key] for k in routers]
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
        rows.append([key, f(min(v), nd), f(max(v), nd), f(mean, nd), f(sd, nd), f(M["No.1"][key], nd), f(M["No.2"][key], nd)])
    out += table(["지표", "min", "max", "mean", "std (6 arm)", "No.1", "No.2"], rows)

    out += ["### 1.7 참고: 상위 저장소 10K step 단일 LoRA (직접 비교 대상 아님)", ""]
    rows = []
    for label, arm in REFS:
        p = UPSTREAM / arm / "eval.json"
        if p.exists():
            m = metrics(load(p))
            rows.append([label, f(m["cmmd"]), f(m["in"], 2), f(m["weather"], 2), *[f(m[c], 2) for c in WEATHER]])
    out += table(["구성", "CMMD ↓", "CLIP in", "CLIP weather ↑", *WEATHER], rows)
    return M


def section_diagnosis(out: list[str]) -> None:
    d = load(RUN / "diagnosis/diagnosis.json")
    ts = d["timesteps"]
    J = d["jaccard_matrix"]["values"]
    out += ["## 2. 진단: 선택 집합의 t 의존 (2단계)", "",
            f"출처: `{rel(RUN / 'diagnosis/diagnosis.json')}`. 축 {d['select_axis']}, {d['granularity']} 단위 {d['num_units_total']}개 중 "
            f"상위 {d['fraction']:.0%}({d['per_timestep'][str(ts[0])]['num_units']['style']}개), t {len(ts)}개. "
            "split-half = 같은 t에서 샘플을 반으로 나눠 각각 뽑은 집합 간 Jaccard(측정 잡음의 상한선 역할).", ""]
    rows = []
    for i, t in enumerate(ts):
        p = d["per_timestep"][str(t)]
        rows.append([str(t), f(p["split_half_jaccard"]), f(p["jaccard_vs_81"]), f(J[i][i + 1]) if i + 1 < len(ts) else "—",
                     f(list(p["cross_axis_jaccard"].values())[0])])
    out += table(["t", "split-half J", "J vs t=81", "J vs 다음 t", "style∩viewpoint J"], rows)

    rep = [41, 81, 201, 375, 750]
    idx = [ts.index(t) for t in rep]
    out += ["대표 t 사이의 Jaccard (expert 선택 집합 간 겹침):", ""]
    out += table(["t", *map(str, rep)], [[str(rep[a]), *[f(J[i][j], 2) for j in idx]] for a, i in enumerate(idx)])


def section_experts(out: list[str]) -> None:
    out += ["## 3. expert와 선택 집합 (3단계)", "",
            "출처: `experts/*/train_summary.json`, `experts/*/selection.json`. 선택 sha가 같으면 같은 head 집합이다. "
            "모든 expert: rank 64, 4000 step, batch 1, lr 1e-4.", ""]
    rows, seen = [], {}
    for run, tag in ((RUN, ""), (C1, "C1/")):
        for ed in sorted((run / "experts").iterdir()):
            ts_path = ed / "train_summary.json"
            if not ts_path.exists():  # n1_i0: imported upstream step-4000 checkpoint
                sel = load(run / "selections/n1/n1_i0.json")
                imp = load(run / "IMPORTED.json")["checkpoint"]["path"]
                rows.append([tag + ed.name, "[0,999]", "[81]", sel["meta"].get("selection_sha", "55e73f3c")[:8] if isinstance(sel.get("meta"), dict) else "—",
                             str(sel["num_modules"]), f"{sel['params_selected'] / 1e6:.1f}M", f"{100 * sel['fraction_achieved']:.2f}%", "—",
                             f"import: 상위 `{imp}`"])
                continue
            s = load(ts_path)
            md, fin = s["metadata"], s["final"]
            sel = load(ed / "selection.json")
            key = (md["t_lo"], md["t_hi"], md["selection_sha"])
            note = ""
            if key in seen:
                first, elapsed = seen[key]
                note = f"= {first} 공유 (재학습 안 함)" if elapsed == fin["elapsed"] else f"{first}와 동일 구성 재실행"
            seen.setdefault(key, (tag + ed.name, fin["elapsed"]))
            rows.append([tag + ed.name, f"[{md['t_lo']},{md['t_hi']}]", md["ca_timesteps"], md["selection_sha"][:8],
                         str(sel["num_modules"]), f"{sel['params_selected'] / 1e6:.1f}M", f"{100 * sel['fraction_achieved']:.2f}%",
                         f"{fin['elapsed'] / 60:.0f}", note])
    out += table(["expert", "구간", "선택 t", "sha", "모듈 수", "선택 파라미터", "attn 파라미터 대비", "학습 분", "비고"], rows)


def section_paired(out: list[str]) -> None:
    out += ["## 4. paired L_Diff (3b단계)", "",
            "출처: `experts/*/paired_loss.json`, `experts/n1_i0/lora-final.paired_loss_t*.json`, `runs/sdxl_c1/experts/*/paired_loss.json`. "
            "구간마다 같은 96개 (이미지, t, 노이즈)에서 base 대비 짝지은 차이. 셀 = 상대 변화 (±SEM, 개선된 표본 비율).", ""]

    def cell(path: Path, base_mean=None):
        if not path.exists():
            return "—", None
        p = load(path)
        sem_rel = p["paired_delta_sem"] / p["base_mean"]
        return f"{100 * p['relative_change']:+.2f}% (±{100 * sem_rel:.2f}, {100 * p['frac_improved']:.0f}%)", p

    rows = []
    for lo, hi in PAIRED_INTERVALS:
        own = PAIRED_OWN[(lo, hi)]
        c_own, p_own = cell(RUN / f"experts/{own}/paired_loss.json")
        c_n1, p_n1 = (c_own, p_own) if own == "n1_i0" else cell(RUN / f"experts/n1_i0/lora-final.paired_loss_t{lo}-{hi}.json")
        c_c1, p_c1 = cell(C1 / f"experts/{PAIRED_C1[(lo, hi)]}/paired_loss.json") if (lo, hi) in PAIRED_C1 else ("—", None)
        base = next(p for p in (p_own, p_n1, p_c1) if p)["base_mean"]
        rows.append([f"[{lo},{hi}]", f"{base:.5f}", f"{own}: {c_own}", c_n1, c_c1])
    out += table(["구간", "base L_Diff", "구간 expert", "n1_i0 (그 구간만)", "C1 expert (t=81 선택)"], rows)


def section_router(out: list[str]) -> None:
    out += ["## 5. 라우터 (4단계)", "",
            "출처: `bundles/*/bundle_stats.json`, `assemble_summary.json`. 라우터 = FC(mean z_t) + E(t) + bias, 모듈별 raw 게이트, 초기값 1.0, "
            "1000 step, batch 8, lr 1e-3. g0 = bias + E(t) (라텐트 항 제외한 게이트), 전 모듈 × 전 t 통계.", ""]
    rows, set_rows = [], []
    for label, name in BUNDLES:
        bd = RUN / "bundles" / name
        if not (bd / "bundle_stats.json").exists():
            continue
        s = load(bd / "bundle_stats.json")
        r, a = s["router"], load(bd / "assemble_summary.json")
        g = r["gate_g0_all"]
        hist = a["history"]
        rows.append([label, "+".join(r["context_sets"]), str(len(s["experts"])), str(r["num_modules"]), f(g["mean"]), f(g["std"]),
                     f(g["min"], 2), f(g["max"], 2), f(g["frac_negative"], 2), f(r["bias_abs_shift_mean"]), f(r["t_embed_abs_max"]),
                     f(r["fc_weight_frobenius"], 2), f(hist[0]["loss"], 4), f(a["final"]["loss"], 4), f"{a['final']['elapsed'] / 60:.0f}",
                     f"{s['file_mb']:.0f}"])
        for cs, ps in r["per_set"].items():
            iv = ps["by_core_interval"]
            set_rows.append([label, cs, f(ps["bias"]["mean"]), f(ps["bias"]["min"], 2), f(ps["bias"]["max"], 2),
                             *[f(v["mean"]) for v in iv.values()], " / ".join(iv)])
    out += table(["구성", "context", "expert 수", "게이트 모듈", "g0 mean", "g0 std", "min", "max", "음수 비율", "bias 이동 mean", "E(t) 최대 절대값",
                  "‖fc.W‖_F", "loss@50", "loss@1000", "학습 분", "MB"], rows)
    out += ["context set별 게이트 (bias 평균과 core 구간별 g0 평균; 구간 간 차이가 t 의존의 크기):", ""]
    width = max(len(r) for r in set_rows)
    out += table(["구성", "set", "bias mean", "min", "max", *[f"구간 {i + 1}" for i in range(width - 6)], "구간 정의"],
                 [r[:-1] + ["—"] * (width - len(r)) + r[-1:] for r in set_rows])
    out += ["라우터 학습 loss는 batch마다 (이미지, t) 추첨이 지배하므로 loss@50 → loss@1000 비교는 학습 신호가 아니다(참고용).", ""]


def section_cost(out: list[str]) -> None:
    out += ["## 6. 비용", ""]
    # shared experts carry a copy of the same train_summary -> identical elapsed; count each run once
    foster = sorted({load(p)["final"]["elapsed"] for run in (RUN, C1) for p in (run / "experts").glob("*/train_summary.json")})
    asm = [load(RUN / "bundles" / n / "assemble_summary.json")["final"]["elapsed"] for _, n in BUNDLES
           if (RUN / "bundles" / n / "assemble_summary.json").exists()]
    out += table(["항목", "횟수", "평균 (분)", "합계 (시간)"], [
        ["fostering (train_summary 기준, 4K step)", str(len(foster)), f"{sum(foster) / len(foster) / 60:.1f}", f"{sum(foster) / 3600:.1f}"],
        ["라우터 학습 (1K step, batch 8)", str(len(asm)), f"{sum(asm) / len(asm) / 60:.1f}", f"{sum(asm) / 3600:.1f}"],
    ])
    out += ["생성(900장 ≈ 3.9 h/arm, 15.5 s/장)과 CA 측정(76 min/t)은 로그 기반 수치로 `docs/RESULTS.md` §8 참고.", ""]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "docs/QUANTITATIVE.md"))
    args = parser.parse_args(argv)

    proto = load(RUN / "no2/eval.json")["protocol"]
    out = ["# 정량 결과 — CA-LoRA with TimeStep Master (SDXL + Cityscapes)", "",
           "> `python scripts/make_quantitative.py`로 원본 JSON에서 자동 생성. 손으로 고치지 말 것. 해석은 `docs/RESULTS.md`, 정성은 `docs/QUALITATIVE_2026-09-19.md`.", "",
           "## 0. 프로토콜", "",
           f"- 생성: {proto['scheduler']}, {proto['num_inference_steps']} step, guidance {proto['guidance_scale']}, {proto['resolution']} px, "
           f"{proto['model']['dtype']}, seed {proto['seed']} (**1개, seed 간 분산 미측정**), 프롬프트 \"{proto['prompt']}\" + 클래스명.",
           f"- 구성당 900장 = in_domain 500 + 날씨 {len(proto['weather_conditions'])}조건({', '.join(proto['weather_conditions'])}) × 100.",
           f"- CMMD: {proto['clip_model']}, 참조 Cityscapes {proto['reference_split']} {proto['reference_num_images']}장, σ={proto['cmmd_sigma']}, ×{proto['cmmd_scale']:.0f}. "
           "CLIP-Score 같은 모델. mIoU 미진행.", ""]
    section_eval(out)
    section_diagnosis(out)
    section_experts(out)
    section_paired(out)
    section_router(out)
    section_cost(out)

    text = "\n".join(out).rstrip() + "\n"
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
