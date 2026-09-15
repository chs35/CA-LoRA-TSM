# GPU 실행 지시서 (A100 pod)

> 작성: 2026-09-14, Fable 5.1. CPU에서 toy 파이프라인 4단계와 테스트 46개(35 통과 + GPU용 11 skip)를 확인한 상태에서
> GPU pod에 붙는 에이전트가 따라야 할 순서. 각 단계의 **완료 조건**과 **보고 규칙**을 지킨다.
> 근거 문서: `docs/draft.md`(연구 노트 v2), `docs/upstream/RUNBOOK_A100.md`, `docs/upstream/PROJECT_LOG.md`(실측 시간). 상위 저장소 `/workspace/Optimizing-CA-LoRA`는 그대로 있고(생성 이미지와 CMMD reference PNG만 2026-09-14에 삭제), 필요한 파일은 이 저장소로 복사돼 있다.

## 준비 상태 요약 (CPU pod에서 확인, 2026-09-14)

| 항목 | 상태 |
|---|---|
| 코드: select / foster / assemble / bundle | CPU toy E2E 통과 (`runs/tiny_cpu_e2e/`) |
| GPU 테스트 | `tests/test_gpu.py` 11개 작성. **실제 CUDA에서는 미실행** (CPU 치환본으로 10개 통과, SDXL 스모크는 config 파싱만 확인) |
| SDXL 가중치 | `/workspace/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0` (26 GB) |
| Cityscapes | `/workspace/data/cityscapes` (23 GB, gtFine + leftImg8bit) |
| CLIP (CMMD/CLIP-Score) | `models--openai--clip-vit-large-patch14` 캐시됨 |
| **import된 n1 expert (다른 세션, 2026-09-14 07:01)** | `runs/sdxl_cityscapes/experts/n1_i0/lora-final.safetensors` (style) 및 `runs/sdxl_cityscapes_viewpoint/…` (viewpoint): 상위 A_t81 10K run의 **step-4000 체크포인트**, catsm 메타데이터로 재스탬프(`IMPORTED.json`). 같은 프로토콜의 4K 학습이지만 drift 스냅숏의 merge/unmerge 때문에 새 foster와 bit-동일하지 않음. `selections/n1/n1_i0.json`(t=81, 208 unit)과 강건성 sweep용 `n1_i0_t0001/t0041/t0201.json`도 있음 |
| **기존 CA 측정 (재사용 가능)** | `runs/sdxl_cityscapes/upstream_tsweep_head/measurements/` (상위 저장소에서 복사, `PROVENANCE.md` 참조) — head 단위, t ∈ {1, 41, 81, 121, 201, 301, 401, 481, 601, 701, 801, 901, 981}, 256 샘플 × 4 aug, fp32, seed 0. 이 저장소 로더로 로드 확인(10,400 unit). 모델·데이터·측정 프로토콜이 `configs/sdxl_cityscapes.yaml`과 동일(timesteps 목록만 다름) |
| **CPU에서 미리 작성한 도구 (2026-09-14)** | `scripts/diagnose_ca.py`(2단계: 분포 히스토그램·layer heatmap·Jaccard 행렬·결정 템플릿, 점수 없음), `scripts/generate_images.py --bundle`(5단계: denoising step마다 게이트를 설정하는 UNet pre-hook), `scripts/evaluate.py`(CMMD + CLIP-Score), `assemble --disable <expert>`(C2). 전부 toy로 테스트됨(`tests/test_diagnose.py`, `tests/test_scripts.py`, C2 테스트). **mIoU는 Improving CA-LoRA와 동일하게 미진행(사용자 확정)** → CMMD + weather CLIP-Score만 보고 |

## pod 환경 체크리스트 (첫 명령)

```bash
cat /sys/fs/cgroup/memory.max          # fp32 SDXL 로드에 CPU RAM ~13 GB 필요. 8 GB면 즉사 → 재배포
nvidia-smi                              # A100 80 GB 확인
nproc                                   # 128코어 pod: 아래 스레드 제한 필수
cd /workspace/CA-LoRA-TSM
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16   # env.sh 기본값은 2(CPU pod용). 128코어에서 무제한이면 소형 CPU 연산이 수백 배 느려짐
source scripts/env.sh                   # venv + PYTHONPATH=src + HF_HOME=/workspace/.cache/huggingface
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

- 모든 장기 실행은 `nohup ... > runs/<name>/logs/<job>.out 2>&1 &` 로 띄우고 PID를 기록한다. **PID는 `$!`가 아니라 `pgrep -f <스크립트>`로 잡는다**: `source env.sh && ... && nohup ... &` 처럼 `&&` 체인 끝에 `&`를 붙이면 `$!`는 서브셸 PID라서 kill해도 python은 살아남는다(2026-09-14 실제 발생). 종료 확인은 `pgrep -af`와 `nvidia-smi --query-compute-apps=pid`로 한다.
- 네트워크 볼륨은 간헐적으로 EIO를 낸다(상위 저장소 실측). 체크포인트·이미지는 저장 직후 다시 열어 검증하고, 손상 파일은 지우고 `--skip-existing`으로 재개한다.
- `--set` 값은 YAML로 파싱된다. `train.mixed_precision=no`는 `False`가 되어 거부되므로 생략(기본값 "no")하거나 `"'no'"`로 인용한다.

## 보고 원칙 (전 단계 공통, 사용자 지시 2026-09-14)

**30분 단위로 보고한다** (`/loop 30m`). **보고의 모든 시각은 서울 시간(KST, UTC+9)으로 적는다**(사용자 지시 2026-09-14; `TZ=Asia/Seoul date`). 로그 파일의 타임스탬프는 UTC이므로 +9h로 환산한다. 각 단계의 개별 보고 규칙(1단계의 "현재 t·완료 t·예상 종료", 3·4단계의 step 줄)은 이 원칙의 세부 형식이다. 매 보고에 다음을 적는다.

```
[HH:MM KST] 단계 <N> — <세부 단계>            예) 1단계 — t=61 측정
  진행도: <세부 단계 %> (<근거: step k/N, 샘플 i/256 등>)
  세부 단계 예상 종료: <시각>
  단계 전체 예상 종료: <시각> (완료: <목록> / 남은 것: <목록>)
  오류: 없음 | <내용 → 대응 결과>
```

- **오류가 나면 30분을 기다리지 않고 즉시 대응하고 보고한다.** 대응 = 원인 확인(로그·프로세스·VRAM·EIO) → 조치(재시작·손상 파일 삭제 후 `--skip-existing` 재개·설정 변경) → 결과. **1단계 종료 이후(2·3단계부터)는 사용자 답장을 기다리지 않는다**(사용자 지시 2026-09-14): 문제가 생기면 보고를 남기고 즉시 대응한다(재시작, 손상 파일 삭제 후 재개, 병렬→순차 전환, 설정 변경 등). 대응 내용과 근거는 보고와 `runs/<run>/logs/`에 남긴다. 예외는 완료된 산출물(체크포인트·측정 파일)을 지우는 것뿐이며, 이 경우에만 보고 후 기다린다.
- 오류 판단 기준: 프로세스 사망, 로그 10분 이상 정체, loss NaN/급증, `leaves the expert's interval`, CUDA OOM, EIO, 저장 직후 재로드 실패.
- 진행도의 근거(로그 줄)와 예상 종료의 계산 근거(실측 s/step 또는 t당 분)를 함께 적는다. 단계가 바뀌거나 세부 단계가 끝나면 30분 주기와 무관하게 한 줄 보고한다.
- 보고 정보원: `runs/<run>/run.log`, `runs/<run>/logs/<job>.out`, `experts/<expert>/train_summary.json`, `nvidia-smi`, `kill -0 <PID>`.

---

## 0단계 — GPU에서 코드 정상 작동 테스트

```bash
python -m pytest tests/ -q -p no:cacheprovider          # 40 CPU 테스트 + 11 GPU 테스트가 GPU에서 실행됨
CATSM_SDXL_SMOKE=1 python -m pytest tests/test_gpu.py -q -k sdxl -p no:cacheprovider   # 실제 SDXL, 256 px, dummy 데이터, 단계별 1 step (SDXL 7회 로드, 10~20분)
```

완료 조건:
- 51개 전부 통과. `assert_close` 허용오차만으로 실패하면(값 자체는 유한하고 rtol 1e-3 이내) 허용오차를 완화하고 기록한다. 그 외 실패는 코드 문제이므로 1단계로 넘어가지 않는다.
- SDXL 스모크에서 확인할 것: gradient checkpointing 아래에서 라우터 gate로 gradient가 흐르는지(`assemble_summary.json`의 `fc.bias`가 변했는지는 4단계 실전에서 재확인), 텍스트 인코더 파킹 로그("text encoders moved to CPU"), 선택 unit 수 208.
- 스모크 산출물(`.pytest_tmp/`)은 지운다.

---

## 1단계 — CA 측정 (미리 정한 t 후보)

**후보 t (config `measure.timesteps`):** 1, 41, **61**, 81, 125, 201, 250, 375, 500, 750, 999 — 2026-09-14에 config에 61을 추가해 두었다.
draft §4의 13점 중 625·875는 블록 구조가 불명확할 때만 추가.

**재사용 여부(사용자 결정 필요).** 기존 측정 13개는 같은 이미지·노이즈·seed로 만들어졌으므로 그대로 pairing이 성립한다. 이전 import 세션에서 측정 파일은 "필요 없음"으로 제외됐지만, 복사 한 번이면 t=121·301·401·481·601·701·801·901·981이 진단(Jaccard 행렬)에 공짜로 들어오고 측정 시간이 약 14시간(11개 전부)에서 약 9시간(7개)으로 준다. 전부 새로 재려면 아래에서 `cp`를 빼고 `--skip-existing`도 뺀다.

```bash
RUN=runs/sdxl_cityscapes
mkdir -p $RUN/measurements $RUN/logs
cp $RUN/upstream_tsweep_head/measurements/t*.safetensors $RUN/measurements/
# 새로 잴 t: 61, 125, 250, 375, 500, 750, 999 (기존 13개는 --skip-existing으로 건너뜀)
nohup python scripts/measure_ca.py --config configs/sdxl_cityscapes.yaml --skip-existing \
  --set 'measure.timesteps=[1,41,61,81,121,125,201,250,301,375,401,481,500,601,701,750,801,901,981,999]' \
  > $RUN/logs/measure.out 2>&1 &
```

- 실측 속도(상위 저장소, head 단위, fp32): **t당 약 75분, VRAM 약 67 GB**. 7개 → 약 9시간. 기존 것까지 다 새로 재면 약 25시간.
- `measure_summary.json`은 `--skip-existing`으로 병합되므로 복사한 t의 summary는 비어 있다. 필요하면 `$RUN/upstream_tsweep_head/measure_summary.json`에서 가져온다.
- 완료 후 검증: `python -c "from catsm.select import load_measurements; r=load_measurements('runs/sdxl_cityscapes/measurements'); print(sorted(r), len(next(iter(r.values())).names))"` → 20개 t, 10,400 unit. 로드 실패한 파일(상위 저장소에서 t=41이 한 번 손상됐음)은 지우고 그 t만 재측정.
- 각 t의 `direction_concentration_mean`이 로그에 찍힌다. 1/√n 랜덤 바닥보다 낮은 축이 있으면 보고한다.
- **경계 t 추가 측정 규칙(2단계 결과에 따라):** 어떤 구간의 대표 t가 경계에 붙어 결정되면(예: [0,61]이 61로 결정) 인접 구간 쪽 경계 t(62)도 재서, 경계를 사이에 둔 두 선택 집합이 실제로 달라지는지 Jaccard로 확인한다. 같은 명령에 t를 추가하고 `--skip-existing`으로 돌리면 된다.

**무인 핸드오프(사용자 지시 2026-09-14):** 1단계가 끝나면 별도 지시 없이 2단계(진단, 시각화만)와 3단계 일부(n4_i0·n3_i0·n5_i1을 t=81 선택 `selections/n1/n1_i0.json`으로, C1 n4_i0..i3를 `--name sdxl_c1`로)를 병렬로 시작한다. `scripts/stage1_to_3_handoff.sh <measure_pid>`가 이를 자동화한다(PID 종료 대기 → 20 t 검증 → `diagnose_ca.py` → `scripts/foster_queue.sh` 큐 A/B; 큐 B는 A 종료 후 순차 — 병렬은 실측상 손해, 3단계 참조). 로그: `logs/handoff.out`, `logs/foster_queueA.out`, `logs/foster_queueB.out`.

**보고:** 측정은 t 하나가 75분이므로 30분 보고에 "현재 t, 완료된 t 목록, 남은 t × 75분 = 예상 종료"를 적는다.

---

## 2단계 — 구간별 대표 t 결정 (CA-LoRA의 시각적 휴리스틱)

**휴리스틱 = 그림을 보고 사람이 결정한다(사용자 결정, 2026-09-14).** 점수는 계산하지 않는다. **에이전트는 이 단계에서 시각화만 하고 결정을 내리지 않는다**(사용자 지시 2026-09-14): 그림·`DECISION_TEMPLATE.md`·참고 수치를 보고하고 멈춘다. `DECISION.md` 작성, config 반영, n3/n4/n5 `select` 실행은 사용자 결정 후에만 한다. 그동안 GPU를 쓰지 않도록 `select`에는 `--set model.device=cpu`를 붙인다. `scripts/diagnose_ca.py`가 측정 파일에서 다음 그림을 만든다(13 t × 10,400 unit에 약 40초).

```bash
python scripts/diagnose_ca.py --config configs/sdxl_cityscapes.yaml            # -> runs/sdxl_cityscapes/diagnosis/
```

- `ca_distribution.png`: t마다 style-CA와 viewpoint-CA의 정규화 분포 히스토그램(log 스케일). 원 논문이 t=81을 고른 근거인 "두 개념의 분포 차이"를 그대로 보는 그림.
- `ca_heatmap_<axis>_layers.png` / `_layers_shared.png`: attention layer(140행) × {q,k,v,out} 정규화 CA, t당 한 패널(논문 Fig. 11 배치). 패널별 스케일과 t 간 공유 스케일 두 벌.
- `ca_heatmap_<axis>_blocks.png`: 같은 것을 block 그룹(down1/down2/up0/up1/mid)으로 합친 거친 판.
- `jaccard_matrix.png`: top-2% 집합의 t × t Jaccard(draft §4: 집합이 급변하는 t가 자연스러운 경계).
- `DECISION_TEMPLATE.md`: scheme별·구간별 후보 t와 빈 `decision:` 줄. 참고 수치로 split-half Jaccard(샘플링 안정성, draft §4.3)와 t=81 집합과의 Jaccard만 적혀 있다.

CPU pod에서 기존 13개 t로 미리 돌린 결과(`diagnosis.json`)에서 결정에 참고할 사실만 적는다:
- [0,61]: t=1의 split-half Jaccard 0.857(전체 최고), t=41은 0.684. 61은 아직 측정 전.
- [125,249]: 기존 측정에는 201만 속한다(121은 [62,124]). 125를 재야 경합이 성립한다.
- t≥601의 top-2% 집합은 81의 집합과 Jaccard 0.03~0.05로 거의 겹치지 않고, t=401~701의 split-half Jaccard는 0.33~0.47로 **집합의 절반 가량이 샘플링 노이즈**다. draft §3.4의 우려(높은 구간 선택 불안정 → random-sliced/미적용/C2 검토)가 데이터로 확인된 셈이므로 결정 보고에 포함한다.

**이미 확정된 것(사용자 지시):**

| 구간 | expert | 대표 t |
|---|---|---|
| [0, 124] | n4_i0 | **81 (확정)** |
| [62, 124] | n5_i1 | **81 (확정)** |
| [0, 61] | n5_i0 | 1 / 41 / 61 경합 |
| [125, 249] | n4_i1, n5_i2 | 125 / 201 경합 |
| [0, 249] | n3_i0 | 81이 포함되므로 81 (확정과 같은 논리, 이견 있으면 보고) |
| [250, 499] | n3_i1, n4_i2, n5_i3 | 250 / 375 / 481(기존) 경합 |
| [500, 999] | n3_i2, n4_i3, n5_i4 | 500 / 750 / 999 (+601, 701, 801, 901, 981 기존) 경합 |
| [0, 999] | n1_i0 | 81 (CA-LoRA baseline과 동일) |

주의: 기존 측정의 121은 [62,124]에 속하므로 [125,249]의 대표가 될 수 없다. 500은 [500,999]의 첫 t다.

**진단 스크립트:** 위 1~4를 `scripts/diagnose_ca.py`가 계산한다 (13 t × 10,400 unit에 약 17초).

```bash
python scripts/diagnose_ca.py --config configs/sdxl_cityscapes.yaml            # -> runs/sdxl_cityscapes/diagnosis/
#   diagnosis.json, kld_by_t.png, jaccard_matrix.png, ca_heatmap_<axis>[_shared].png, DECISION_TEMPLATE.md
```

**CPU pod에서 기존 13개 t로 미리 돌린 결과 (style vs viewpoint, top 2%):**

| t | KLD | JS | Jaccard vs 81 | split-half Jaccard |
|---|---|---|---|---|
| 701 | 0.0452 | 0.0056 | 0.025 | 0.470 |
| 801 | 0.0399 | 0.0050 | 0.053 | 0.546 |
| 901 | 0.0366 | 0.0046 | 0.043 | 0.684 |
| 981 | 0.0331 | 0.0041 | 0.043 | 0.755 |
| 601 | 0.0327 | 0.0040 | 0.053 | 0.333 |
| 401 | 0.0303 | 0.0038 | 0.178 | 0.455 |
| 481 | 0.0252 | 0.0031 | 0.106 | 0.425 |
| 301 | 0.0227 | 0.0028 | 0.257 | 0.524 |
| 201 | 0.0177 | 0.0022 | 0.316 | 0.576 |
| 121 | 0.0172 | 0.0021 | 0.588 | 0.631 |
| 1 | 0.0165 | 0.0021 | 0.470 | 0.857 |
| 81 | 0.0159 | 0.0020 | 1.000 | 0.664 |
| 41 | 0.0112 | 0.0014 | 0.657 | 0.684 |

읽는 법과 주의:
- 전 분포 KLD는 t에 대해 거의 단조 증가한다. 논문의 t=81은 13개 중 12위다. 즉 "분포 차이가 가장 큰 t"를 **전체 분포 KLD로 읽으면 논문의 선택이 재현되지 않는다**. 논문의 대비는 상위 CA head의 heatmap 구조에서 읽은 것이므로, KLD는 **구간 내 후보 비교의 한 입력**으로만 쓰고 heatmap(`ca_heatmap_*`), Jaccard vs 81, split-half 안정성을 함께 본다. 구간을 넘어 t를 고르는 데 KLD를 쓰지 않는다.
- `ca_heatmap_*`의 행은 block 그룹(down1/down2/up0/up1/mid, 각 그룹의 attention layer를 합산)이라 논문 Fig. 11(layer별)보다 거칠다. 논문식 layer별 패널이 필요하면 `diagnose_ca.py`의 `block_kind_table`을 `group_indices("block_id")` 대신 attention 모듈 단위로 바꾸면 된다(측정 파일 재사용, 재측정 불필요).
- [0,61]: t=1은 split-half 0.857(전체 최고)·KLD 0.0165, t=41은 0.684·0.0112. 61은 아직 측정 전.
- [125,249]: 기존 측정에는 201만 속한다(121은 [62,124]). 125를 재야 경합이 성립한다.
- t≥601의 선택 집합은 81의 집합과 Jaccard 0.03~0.05로 거의 겹치지 않고, t=401~701의 split-half Jaccard는 0.33~0.47로 **top-2% 집합의 절반 가량이 샘플링 노이즈**다. draft §3.4의 우려(높은 구간 선택 불안정 → random-sliced 또는 미적용 검토, C2)가 데이터로 확인된 셈이므로 결정 보고에 포함한다.

**결정 완료(2026-09-15, 사용자):** n3 [81, 375, 750], n4 [81, 201, 375, 750], n5 [41, 81, 201, 375, 750]; 경계는 절반 자르기 그대로. 근거와 sha는 `runs/sdxl_cityscapes/diagnosis/DECISION.md`. config는 `interval_select.mode: representative`로 바뀌었고 `selections/n3,n4,n5/`가 생성됨.

**결정 후 config 반영:**

```yaml
interval_select:
  mode: representative
  representative:
    n1: [81]
    n3: [81, <250|375|481>, <500|750|999>]
    n4: [81, <125|201>, <...>, <...>]
    n5: [<1|41|61>, 81, <125|201>, <...>, <...>]
```

```bash
for s in n3 n4 n5; do python -m catsm.cli select --config configs/sdxl_cityscapes.yaml --scheme $s; done
# n1은 import된 selections/n1/n1_i0.json(t=81, sha 55e73f3c…)을 그대로 쓴다. 다시 만들면 같은 내용이어야 하며 sha가 다르면 보고한다.
# 통제 C1: 선택은 t=81 고정, 학습만 구간별 → 별도 run 이름으로 분리(같은 run이면 selections/n4/를 덮어씀)
python -m catsm.cli select --config configs/sdxl_cityscapes.yaml --name sdxl_c1 --scheme n4 --mode fixed \
  --measurements runs/sdxl_cityscapes/measurements
```

완료 조건: `selections/<scheme>/intervals.json`에 expert마다 `timesteps`, `selection_sha`, `num_units`(208), `logsnr_lo/hi`가 있다. 결정 근거(KLD 값, Jaccard, 선택한 figure)를 `runs/sdxl_cityscapes/diagnosis/DECISION.md`에 남기고 **사용자에게 보고 후** 3단계로 간다.

---

## 3단계 — Fostering (구간별 sliced LoRA 학습)

> **상태(2026-09-15 16:45 KST): 3단계 완료.** `runs/sdxl_cityscapes/experts/` 13개(n1_i0 import, n3_i0·n4_i0..i3·n5_i0·n5_i1 학습, n3_i1·n3_i2·n5_i2..i4 공유) + `runs/sdxl_c1/experts/` n4_i0..i3. 실측: expert당 70~76분(1.05~1.15 s/step), VRAM 14.5 GB, 학습 12회 총 약 14.5시간. 모든 파일 재로드·유한값·t 범위 확인. loss 곡선 `runs/loss_curves.png`(`scripts/plot_loss.py`), paired loss `experts/*/paired_loss*.json`(`scripts/paired_loss.py`). 다음: 4단계.

**같은 구간·같은 선택 sha의 expert는 한 번만 학습하고 가중치를 공유한다(사용자 결정 2026-09-15).** 절반 자르기 경계에서는 n4_i1=n5_i2([125,249]@201), n3_i1=n4_i2=n5_i3([250,499]@375), n3_i2=n4_i3=n5_i4([500,999]@750)가 정의상 같은 학습이므로(같은 sha·구간·seed; 따로 돌리면 GPU 비결정성만 더해짐 — n4_i0 vs C1 n4_i0 최대 차이 0.04로 확인) 고유 expert는 13개가 아니라 8개다. 복사는 `python scripts/share_expert.py --config $CFG --from n4_i2 --to n3_i1 n5_i3`: 텐서는 bit-동일, safetensors 헤더의 `expert`/`interval_scheme`만 재스탬프하고 `shared_from`을 남긴다(`SHARED_FROM.json`). 구간이나 sha가 다르면 거부한다. `scripts/stage3_queueC.sh`가 남은 4개(n5_i0, n4_i1, n4_i2, n4_i3) 학습 후 공유까지 자동으로 한다. 리포트에는 "명목 13개, 실제 학습 8개(32K step)"로 적는다.

학습 순서는 사용자의 지시에 따른다.

**n1_i0는 이미 있다** (준비 상태 표 참조: import된 step-4000 체크포인트, `foster`를 돌리면 덮어쓰므로 돌리지 않는다). 새로 foster한 expert와 학습 조건이 완전히 같지는 않으므로(drift 스냅숏 merge/unmerge) 시간이 남으면 마지막에 `foster --expert n1_i0`로 새로 학습해 두 파일의 결과 차이를 리포트에 적는다. 그때는 기존 파일을 `experts/n1_i0_imported/`로 옮겨 보존한다.

```bash
CFG=configs/sdxl_cityscapes.yaml
nohup python -m catsm.cli foster --config $CFG --expert n1_i0 > runs/sdxl_cityscapes/logs/foster_n1_i0.out 2>&1 &
# 순서대로 n4_i0, n4_i1, n4_i2, n4_i3; C1은 --name sdxl_c1 (selections/n4/가 fixed 모드 것)
```

- 설정: rank 64, alpha 64, 4K step, batch 1, lr 1e-4, fp32, gradient checkpointing, checkpoint 1000 step마다.
- 실측 속도(상위 저장소, 같은 조건 10K step): **2.54 s/step → expert당 약 2.8시간.** 8개 ≈ 22시간, n3·n5까지 16개 ≈ 45시간.
- 강건성 sweep(draft §5.3, 여유 시): `foster --expert n1_i0 --selection runs/sdxl_cityscapes/selections/n1/n1_i0_t0041.json --name sdxl_sweep_t41` 식으로 선택 파일만 바꿔 별도 run에 학습한다(t=1, 41, 201 파일이 이미 있음).
- viewpoint 축 실험은 `--name sdxl_cityscapes_viewpoint --set select.axis=viewpoint`로 run dir를 분리한다(draft의 주 실험은 style).
- 동시 실행: **하지 않는다(2026-09-15 실측).** 단독 fostering은 VRAM 14.5 GB, 1.08 s/step(4K step ≈ 72분, 상위 저장소의 2.54 s/step보다 빠름)이지만 GPU 사용률이 이미 100%라, 두 개를 병렬로 돌리면 각각 2.36 s/step → 합산 1.18 s/step으로 순차보다 9% 느리다. 큐는 하나만 돌린다(`scripts/foster_queue.sh`).
- 각 expert 종료 시 확인: `train_summary.json`의 `t_min_seen/t_max_seen`이 구간 안, loss 유한, `lora-final.safetensors` 재로드(`catsm.train.assemble.load_expert_into`로 pristine UNet에 붙여보기 — 4단계가 어차피 한다).

**보고 규칙 (30분 단위, 4단계도 동일):** `/loop 30m`으로 아래를 반복 보고한다.

```
[HH:MM] 진행 중: <expert> step <k>/4000 (<n>%), loss <최근 값>, <s/step>, ETA <종료 예상 시각>
       완료: <expert 목록>   대기: <expert 목록>   전체 예상 종료: <시각>
       이상: 없음 | <내용>
```
- 정보원: `runs/<run>/run.log`의 `step k/4000 loss=... (s/step, ETA ...)` 줄(50 step마다), `experts/<expert>/train_summary.json`.
- 오류 확인 항목: 프로세스 생존(`kill -0 PID`), 로그 마지막 줄이 10분 이상 갱신 없음, loss NaN/급증, `Timestep draw ... leaves the expert's interval`(있으면 즉시 중단·보고), CUDA OOM, EIO. 죽었으면 마지막 `lora-stepNNNNNN.safetensors`부터 재개할 수 없음(재개 기능 없음) → 처음부터 다시 돌리고 원인을 보고한다.

---

## 4단계 — Assembling (router 학습)

주논지 구성(draft §5.1) 순서: **No.1 → No.2 → No.3 → C1**, 여유 시 No.4, 5, 8, C2.

```bash
CFG=configs/sdxl_cityscapes.yaml
python -m catsm.cli assemble --config $CFG --sets n1 --core n1 --no-router          # No.1: CA-LoRA 4K baseline (학습 없음)
python -m catsm.cli assemble --config $CFG --sets n4 --core n4 --no-router          # No.2: n4 하드 라우팅
nohup python -m catsm.cli assemble --config $CFG --sets n1 n4 --core n4 > runs/sdxl_cityscapes/logs/assemble_n1+n4.out 2>&1 &   # No.3
python -m catsm.cli assemble --config $CFG --name sdxl_c1 --sets n4 --core n4 --no-router   # C1
# 여유 시: No.4 --sets n1 n3 n4 --core n4 / No.5 --sets n1 n4 n5 --core n4 / No.8 --sets n1 n3 n4 n5 --core n4
```

- 설정: expert 동결, router(FC + E(t), 모듈별 raw gate, init 1.0) 1000 step, lr 1e-3, batch 1. 매 step UNet forward+backward가 있으므로 fostering과 비슷한 step 비용(≈2.5 s) → **구성당 약 45분**.
- 산출: `bundles/<sets>-core-<core>/bundle.safetensors` (`format=tsm-sliced-v1`), `assemble_summary.json`. 확인: `router.fc.bias`가 초기값 1.0에서 움직였는지, gate 통계(모듈별 평균/부호)를 요약에서 뽑아 보고.
- **C2(높은 구간 expert 비활성):** `assemble --sets n1 n4 --core n4 --disable n4_i3` → `bundles/n1+n4-core-n4-disable-n4_i3/`. 비활성 expert는 번들에 저장은 되지만 모든 t에서 가중치 0으로 라우팅되고(`disabled` 메타데이터로 재로드 시 복원), 그 구간은 남은 set(여기서는 n1_i0)이 맡는다. 라우터는 활성 context expert가 있는 모듈만 게이트한다. "최종 구성"이 정해진 뒤 그 구성에 `--disable <최고 구간 expert들>`을 붙여 만든다.
- 보고 규칙은 3단계와 동일(`router step k/1000` 줄).

---

## 5단계 — Evaluation

**도구:** 이 저장소의 `scripts/generate_images.py`와 `scripts/evaluate.py` (상위 저장소 스크립트를 가져와 config 로더와 `--bundle`을 붙임). 상위 스크립트를 그대로 쓰면 안 된다: diffusers 파이프라인이 UNet을 직접 호출해 gate가 설정되지 않으므로 expert가 전부 무시된 base 이미지가 나온다. `--bundle`은 UNet forward pre-hook으로 매 denoising step `assembly.set_gate_for(t, sample)`를 호출하고(CFG의 2B 배치 그대로), manifest에 `gate_hook_calls`를 기록한다. 이 값이 0이면 번들이 적용되지 않은 것이다(스크립트는 이 경우 즉시 실패한다).

```bash
CFG=configs/sdxl_cityscapes.yaml; RUN=runs/sdxl_cityscapes
python scripts/generate_images.py --config $CFG --bundle $RUN/bundles/n1-core-n1/bundle.safetensors      --run-name no1 --skip-existing
python scripts/generate_images.py --config $CFG --bundle $RUN/bundles/n4-core-n4/bundle.safetensors      --run-name no2 --skip-existing
python scripts/generate_images.py --config $CFG --bundle $RUN/bundles/n1+n4-core-n4/bundle.safetensors   --run-name no3 --skip-existing
python scripts/generate_images.py --config $CFG --name sdxl_c1 --bundle runs/sdxl_c1/bundles/n4-core-n4/bundle.safetensors --run-name c1 --skip-existing
python scripts/generate_images.py --config $CFG --run-name base --skip-existing                            # LoRA 없는 대조군
python scripts/evaluate.py --config $CFG --run-name no3                                                    # -> $RUN/no3/eval.json (CMMD, CLIP-Score)
```

- 첫 SDXL 생성 전에 `--set eval.num_images=2 eval.images_per_condition=1 --run-name smoke`로 한 번 돌려 `gate_hook_calls`가 `2 × num_inference_steps × 배치 수`인지 확인한다.
- `--embedder pixel`은 배관 확인용이며 그 수치는 보고하지 않는다.
- **mIoU는 미진행**(Improving CA-LoRA와 동일, 사용자 확정). `eval.json`의 `miou`는 항상 null이며 draft §8의 "CMMD와 mIoU 함께 보고"는 CMMD + weather CLIP-Score로 대체하고 리포트에 명시한다.

**프로토콜(draft §8):**
- 스크리닝: 시드 1개, 구성별 in_domain 500장 + weather 4조건 × 100장. 실측 생성 속도 1024 px 25 step ≈ 1.2 s/it → **구성당 약 7.5시간**. 시간이 없으면 스크리닝은 in_domain 200 + weather 4×50으로 줄이고 그 사실을 기록한다.
- 생성 seed는 구성 간 공유(paired). `eval.num_inference_steps=25`, `guidance_scale=5.0` 고정·명시.
- 상위 2~3개 + No.1은 시드 3개(생성 seed만; LoRA 학습 seed는 고정).
- 각 구성의 `eval.json`에 CMMD, in_domain CLIP-Score, weather 조건별 CLIP-Score를 남긴다.

---

## 6단계 — 결과 정리

`docs/RESULTS.md`에 다음을 표로 정리한다.
1. 진단: t별 KLD, Jaccard 행렬, 구간별 대표 t 결정 근거.
2. 구성표(draft §5.1 No.1/2/3/C1 + 추가분): expert 수, 명목 파라미터 합(13×2% 상한), timestep당 활성 비율(m×2%), fostering 시간, assembling 시간.
3. 평가: 구성별 CMMD / weather CLIP-Score(조건별) / in_domain CLIP-Score, 시드 3개 구성은 평균±std.
4. router 분석: No.3의 모듈별 gate 평균·부호 분포(t 구간별) — context expert가 어디서 억제/증폭되는지.
5. draft §5.2 시나리오 중 어디에 해당하는지, 특히 **C1 ≈ 최고 구성이면 2절 주장 기각**임을 명시.
6. 실행 로그: 각 단계 실제 소요 시간, 오류·재시작 이력, 프로토콜에서 벗어난 결정(이미지 수 축소 등).

---

## 예상 시간 총합 (주논지 최소 세트)

| 단계 | 내용 | 시간 |
|---|---|---|
| 0 | 테스트 + SDXL 스모크 | 0.5 h |
| 1 | 새 t 7개 측정 | 9 h |
| 2 | 진단 실행(1분) + 그림 보고 사용자 결정 | 사용자 판단 |
| 3 | 8 expert × 2.8 h (n1_i0는 import됨) | 22 h |
| 4 | No.3 router 1개 (No.1/2/C1은 학습 없음) | 1 h |
| 5 | 4 구성 × 7.5 h (+ base 대조군 7.5 h) | 30~38 h |
| 6 | 정리 | 2 h |

n3/n5 추가 시 3단계 +22 h, 4·5단계 구성당 +8 h.
