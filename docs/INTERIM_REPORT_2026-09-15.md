# 중간 점검 보고서 — 0~3단계 (2026-09-14 17:36 ~ 09-15 16:45 KST)

> 작성: GPU pod 세션(Fable 5.1), 2026-09-15 17:00 KST. 근거 문서: `docs/draft.md`(연구 노트 v2), `docs/GPU_RUNBOOK.md`.
> 목적: 4단계(assemble)로 넘어가기 전에 1~3단계 산출물의 상태·근거·이탈 사항을 한 곳에 모아 둔다.

## 0. 한 줄 요약

- **0~3단계 완료, 산출물 점검 이상 없음.** CA 측정 20개 t, 진단 그림 8장, 구간별 대표 t 확정, sliced LoRA expert 13개(고유 8 + 공유 5) + 통제 C1 4개.
- 예정 프로토콜에서 벗어난 결정 4가지(§7)와 "학습 loss는 신호가 아니다"라는 관찰(§5.4)은 리포트에 반영해야 한다.
- 다음: 4단계 assemble(No.1, 2, 3, C1 → 여유 시 No.4, 5, 8, C2).

## 1. 환경과 0단계

| 항목 | 값 |
|---|---|
| pod | A100-SXM4-80GB 1장, 128코어, cgroup 250 GB, `/workspace` 네트워크 볼륨 |
| 스레드 | `OMP_NUM_THREADS=16` (env.sh 기본 2는 CPU pod용) |
| 테스트 | `pytest tests/` 50 passed + 1 skipped(SDXL 스모크는 env 플래그 필요), 78 s |
| SDXL 스모크 | 1 passed, 3분 32초. 선택 unit 208, 텍스트 인코더 CPU 파킹 로그, 라우터 `fc.bias` 80개 전부 초기값 1.0에서 이동(gradient checkpointing 아래서 gate gradient 확인) |

## 2. 1단계 — CA 측정

**프로토콜.** SDXL base 1.0 fp32, Cityscapes train, 256 샘플 × 4 aug, seed 0, `paired_across_timesteps`(모든 t가 같은 이미지·노이즈), head 단위 10,400 unit, split-half 저장. 상위 저장소 측정 13개(t = 1, 41, 81, 121, 201, 301, 401, 481, 601, 701, 801, 901, 981)와 동일 config이므로 그대로 pairing이 성립해 복사·재사용했고, 7개(61, 125, 250, 375, 500, 750, 999)를 새로 쟀다.

**실측.** t당 76분(약 17.7 s/샘플), VRAM 69.9 GB. 사용자 지시로 61·125까지 잰 뒤 한 번 중단했고(11:17 UTC), 나머지 5개를 20:52 KST에 재개해 03:14 KST에 끝냈다.

**산출물** (`runs/sdxl_cityscapes/`):
- `measurements/t{t:04d}[_half0|_half1].safetensors` 60개 (20 t × 3), 각 4.9 MB. 로더로 20 t × 10,400 unit, half는 128 샘플 확인.
- `measure_summary.json` 20개 t 병합. 61·125의 summary는 프로세스가 중단돼 저장 파일에서 오프라인 재계산했고, direction concentration은 로그값(소수 4자리)을 넣었다(`note`에 기록).
- 로그: `logs/measure_part1_t61_t125.out`, `logs/measure.out`.

**direction concentration(style / viewpoint).** 새 t 7개 모두 인접 기존 t의 추세 위에 놓였고 1/√n 랜덤 바닥 아래 축은 없었다.

| t | 41 | 61* | 81 | 121 | 125* | 201 | 250* | 301 | 375* | 401 | 481 | 500* | 601 | 701 | 750* | 801 | 901 | 981 | 999* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| style | .380 | .361 | .358 | .342 | .342 | .333 | .316 | .318 | .311 | .305 | .293 | .284 | .282 | .304 | .341 | .376 | .457 | .647 | .682 |
| viewpoint | .317 | .313 | .314 | .305 | .306 | .298 | .277 | .277 | .236 | .223 | .203 | .194 | .206 | .230 | .243 | .258 | .317 | .427 | .457 |

(* = 이번에 새로 측정. t=1은 .443/.368.)

## 3. 2단계 — 진단과 대표 t 결정

**절차.** `scripts/diagnose_ca.py`(CPU, 41초) → `runs/sdxl_cityscapes/diagnosis/`에 `jaccard_matrix.png`, `ca_distribution.png`, `ca_heatmap_{style,viewpoint}_{layers,layers_shared,blocks}.png`, `diagnosis.json`, `DECISION_TEMPLATE.md`. 에이전트는 시각화와 참고 수치만 제공했고 결정은 사용자가 했다(러너북 2단계 규칙).

**그림에서 읽은 사실.**
- 인접 t의 top-2% 집합은 같은 t를 다시 잰 것(split-half)만큼 비슷하다: 인접 19쌍 중 13쌍이 J(a,b) ≥ 0.95 × min(split-half). 뚜렷한 단절은 1|41(비 0.69)과 801|901|981(0.26, 0.55)뿐. 81|121, 601|701, 750|801은 작은 실제 변화(0.82~0.93).
- 차이는 거리에 따라 누적된다(81 대비 J: 121→0.59, 250→0.33, 500→0.11, ≥601→0.02~0.05). 즉 구간 안의 대표 t보다 구간 폭이 선택 집합을 좌우한다.
- [500,999]는 단일 t로 대표되지 않는다(750 vs 981 J 0.05; 601~750의 split-half 0.33~0.48로 집합의 절반이 샘플링 잡음). draft §3.4의 우려가 데이터로 확인됨.
- heatmap은 t에 따라 매끄럽게 드리프트한다(down2 후반·up0 전반 → mid·up0 후반, ≥701에서 up0 저해상). 분포 히스토그램의 style/viewpoint 모양 차이는 ≥375에서 커지고 ≥981에서 다시 겹친다.

**결정(사용자, 2026-09-15).** 경계는 절반 자르기 유지(81이 포함된 구간은 확정). 대표 t:

| 구간 | expert | t | selection sha |
|---|---|---|---|
| [0,999] | n1_i0 | 81 | 55e73f3c (import된 CA-LoRA baseline 선택과 동일, 재생성 시 sha 일치 확인) |
| [0,249] | n3_i0 | 81 | 55e73f3c |
| [0,124] | n4_i0 | 81 | 55e73f3c |
| [0,61] | n5_i0 | 41 | e6629892 |
| [62,124] | n5_i1 | 81 | 55e73f3c |
| [125,249] | n4_i1 = n5_i2 | 201 | 980ee585 |
| [250,499] | n3_i1 = n4_i2 = n5_i3 | 375 | 5cea9d3c |
| [500,999] | n3_i2 = n4_i3 = n5_i4 | 750 | cde28a12 |

t=1은 81과 뚜렷이 다르지만(J 0.47) 그것만으로 채택 이유가 되지 않는다는 판단(사용자): 선택 집합이 조금만 달라도 학습 구간이 다른 것 자체가 기여다. config는 `interval_select.mode: representative`로 바뀌었고 `selections/{n1,n3,n4,n5}/`가 생성됐다. 근거 전문은 `diagnosis/DECISION.md`.

**통제 C1.** `runs/sdxl_c1/selections/n4/` n4_i0..i3 전부 t=81 고정(mode fixed, sha 55e73f3c).

## 4. 3단계 — Fostering

**설정.** rank 64, alpha 64, 4K step, batch 1, lr 1e-4, fp32, gradient checkpointing, 1000 step마다 체크포인트. 구간별 t 추출은 `randint(lo, hi+1)`(0-based 스케줄러 인덱스, 논문·재구현과 동일).

**실측.** expert당 70~76분(1.05~1.15 s/step, 상위 저장소 실측 2.54 s/step보다 2.3배 빠름), VRAM 14.5 GB. 두 개 병렬은 각각 2.36 s/step → 합산 1.18 s/step으로 순차보다 9% 느려서 순차로만 돌렸다. 학습 12회, 총 약 14.5시간(03:16 → 16:44 KST).

**인벤토리** (`runs/sdxl_cityscapes/experts/`, `runs/sdxl_c1/experts/`; 전부 재로드·유한값·t 추출 범위 확인):

| run | expert | 구간 | t | 텐서 수 | 출처 | final loss |
|---|---|---|---|---|---|---|
| sdxl_cityscapes | n1_i0 | [0,999] | 81 | 195 | import(상위 A_t81 10K run의 step-4000, `IMPORTED.json`) | — |
| | n3_i0 | [0,249] | 81 | 195 | 학습 | 0.239 |
| | n3_i1 | [250,499] | 375 | 174 | 공유 ← n4_i2 | 0.231 |
| | n3_i2 | [500,999] | 750 | 288 | 공유 ← n4_i3 | 0.078 |
| | n4_i0 | [0,124] | 81 | 195 | 학습 | 0.239 |
| | n4_i1 | [125,249] | 201 | 192 | 학습 | 0.326 |
| | n4_i2 | [250,499] | 375 | 174 | 학습 | 0.231 |
| | n4_i3 | [500,999] | 750 | 288 | 학습 | 0.078 |
| | n5_i0 | [0,61] | 41 | 174 | 학습 | 0.260 |
| | n5_i1 | [62,124] | 81 | 195 | 학습 | 0.337 |
| | n5_i2 | [125,249] | 201 | 192 | 공유 ← n4_i1 | 0.326 |
| | n5_i3 | [250,499] | 375 | 174 | 공유 ← n4_i2 | 0.231 |
| | n5_i4 | [500,999] | 750 | 288 | 공유 ← n4_i3 | 0.078 |
| sdxl_c1 | n4_i0 | [0,124] | 81 고정 | 195 | 학습 | 0.239 |
| | n4_i1 | [125,249] | 81 고정 | 195 | 학습 | 0.326 |
| | n4_i2 | [250,499] | 81 고정 | 195 | 학습 | 0.230 |
| | n4_i3 | [500,999] | 81 고정 | 195 | 학습 | 0.077 |

텐서 수 차이는 208 head가 몇 개의 행렬에 걸치느냐의 차이(195 = 65행렬 × 3, 288 = 96행렬 × 3)로 정상. 학습 expert는 `lora-step001000/2000/3000.safetensors`(31 MB, 유한값)도 보존.

**공유 expert.** 같은 구간·같은 sha·같은 seed는 정의상 같은 학습이므로 한 번 학습해 `scripts/share_expert.py`로 복사했다(사용자 결정). 텐서는 원본과 bit-동일, safetensors 헤더의 `expert`/`interval_scheme`만 재스탬프, `shared_from`·`shared_from_sha256`과 `SHARED_FROM.json` 기록. 점검에서 원본 sha256 불변·텐서 동일 재확인. 따로 학습했을 때의 차이는 n4_i0 vs C1 n4_i0(같은 설정 재실행)의 최대 가중치 차이 0.038, loss 곡선 일치로 GPU 비결정성 수준임을 확인했다.

### 4.1 학습 loss에 대한 관찰 (리포트 반영)

batch-1 학습 loss는 어떤 이미지·어떤 t가 뽑혔는지가 거의 전부이고 LoRA의 몫은 1% 미만이다. 상위 저장소 10K run 11개(A_t81, B_top3, C_t41, D_t1, 전체 head E_all, 랜덤 E_random)의 구간별 평균 loss가 소수 셋째 자리까지 같다. loss 수준은 구간이 정한다([0,124] ≈ 0.30, [500,999] ≈ 0.02). 따라서 loss 곡선(`runs/loss_curves.png`, `scripts/plot_loss.py`)은 평탄하며 이것은 이상이 아니다.

학습 여부는 짝지은 비교로 확인했다(`scripts/paired_loss.py`, 같은 96개 (이미지, t, 노이즈), t∈[0,124], base 평균 0.3104):

| expert | 짝지은 차이 | 개선 샘플 |
|---|---|---|
| n4_i0 step 4000 | −0.0017 ± 0.0001 (−0.6%) | 100% |
| n4_i0 step 1000 | −0.0012 (−0.4%) | 100% |
| n1_i0(전 구간 4K) | −0.0012 (−0.4%) | 99% |

학습은 1K→4K 사이에도 진행됐고, 구간 전문 expert가 자기 구간에서 전 구간 expert보다 1.5배 더 낮춘다(draft §2의 첫 신호, 96 샘플이라 최종 expert로 재확인 가치 있음). 결과 파일: `experts/n4_i0/paired_loss.json`, `experts/n4_i0/lora-step001000.paired_loss_t0-124.json`, `experts/n1_i0/lora-final.paired_loss_t0-124.json`.

## 5. 산출물 점검 결과 (2026-09-15 16:50 KST)

| 항목 | 점검 내용 | 결과 |
|---|---|---|
| 측정 파일 60개 | 존재·크기(각 4.9 MB)·로더 20 t × 10,400 unit·half 128 샘플 | 이상 없음 |
| measure_summary.json | 20 t, concentration 전부 존재 | 이상 없음 |
| 진단 11개 파일 | 그림 8 + json + 템플릿 + DECISION.md, diagnosis.json 20 t | 이상 없음 |
| selections n1/n3/n4/n5 + C1 n4 | 208 unit, t·sha가 결정과 일치, C1은 fixed@81 | 이상 없음 |
| config | mode representative, 대표 t 결정과 일치 | 이상 없음 |
| expert 17개 | final 재로드·유한값, 메타데이터 expert명·sha가 `selection.json`과 일치, 학습분은 t_min/max_seen이 구간 안·step 4000, 공유분은 원본 sha256 불변·텐서 bit-동일 | 이상 없음 |
| 체크포인트 | 학습 expert 12개 × 3개 | 이상 없음 |
| 용량 | measurements 283 MB, experts 1.1 GB, C1 484 MB | — |

## 6. 오류·이탈과 대응

| 시각(KST) | 내용 | 대응 |
|---|---|---|
| 09-14 20:17 | nohup을 `&&` 체인 끝에 붙여 `$!`가 서브셸 PID(4784)를 기록. 감시기가 그 PID를 죽였지만 python(4792)은 t=250을 1샘플 진행 | `nvidia-smi --query-compute-apps`로 발견, 직접 종료. t=250 파일 미생성이라 오염 없음. 러너북 체크리스트에 "PID는 pgrep, 종료 확인은 nvidia-smi" 규칙 추가 |
| 09-15 03:31 | 핸드오프 스크립트가 VRAM 14.5 GB를 보고 fostering 2개를 병렬 시작 → 각 2.36 s/step(합산 1.18 > 단독 1.08) | 큐 B를 step 200에서 중단(부분 산출물 없음), 큐 A 종료 후 순차 재시작. 러너북 3단계·핸드오프 스크립트에 "병렬 금지" 기록 |
| 09-15 09:41~09:48 | paired loss 검사가 학습과 GPU 공유 → 해당 구간 학습 속도 1.15~1.22 s/step | 검사 종료 후 복귀. 결과에 영향 없음 |
| 09-15 15:26~16:44 | n4_i3 학습이 1.15 s/step(평소 1.07) | GPU 단독·100%·정상 클럭, 호스트 load 37 → CPU 경합으로 판단, 관찰만 |

## 7. 프로토콜에서 벗어난 결정 (리포트에 명시할 것)

1. **명목 13개 LoRA, 실제 학습 8개(32K step).** draft §7의 "13 × 4K = 52K"는 "동일 구성 expert는 공유"로 고쳐 적는다. 공유는 scheme 간 학습 잡음을 없애므로 No.4/5/8 비교가 더 깨끗해진다(불리한 점 없음).
2. **C1 n4_i0는 n4_i0와 동일 구성인데 따로 학습했다.** 버리지 않고 "같은 설정 재실행의 가중치·평가 차이"를 재현성 참고치로 쓴다.
3. **n1_i0는 새로 학습하지 않고 import했다**(상위 10K run의 step-4000, drift 스냅숏의 merge/unmerge 때문에 새 foster와 bit-동일하지 않음). 시간이 남으면 `foster --expert n1_i0`로 새로 학습해 차이를 기록(러너북 3단계).
4. **mIoU 미진행**(사용자 확정, Improving CA-LoRA와 동일). draft §8의 "CMMD와 mIoU"는 CMMD + weather CLIP-Score로 대체.

그 밖에 사용자 지시로 확정된 운영 규칙: 보고는 30분 단위·KST, 2단계는 시각화만(결정은 사용자), 1단계 이후 오류는 보고 후 즉시 대응(완료 산출물 삭제만 예외), 1단계 종료 후 2·3단계 무인 병렬 진행.

## 8. 코드·문서 변경 (커밋 전, `git status`)

- 수정: `configs/sdxl_cityscapes.yaml`(mode representative), `docs/GPU_RUNBOOK.md`(보고 원칙·KST·무인 핸드오프·PID 규칙·병렬 금지·결정·공유·상태), `.gitignore`.
- 추가 스크립트: `scripts/foster_queue.sh`(순차 큐), `scripts/stage1_to_3_handoff.sh`(1→2/3 무인 전환), `scripts/stage3_queueC.sh`(잔여 4개 + 공유), `scripts/share_expert.py`(공유 재스탬프), `scripts/plot_loss.py`(loss 곡선), `scripts/paired_loss.py`(짝지은 base-vs-expert loss).
- `runs/`는 git 추적 밖(네트워크 볼륨에만 존재).

## 9. 다음 단계

**4단계 assemble** (러너북): No.1 `--sets n1 --core n1 --no-router`, No.2 `--sets n4 --core n4 --no-router`, No.3 `--sets n1 n4 --core n4`(라우터 1000 step ≈ 45분 예상; fostering 속도로 보면 약 20분일 수 있음), C1 `--name sdxl_c1 --sets n4 --core n4 --no-router`. 여유 시 No.4(n1 n3 n4), No.5(n1 n4 n5), No.8(n1 n3 n4 n5), C2(`--disable n4_i3`). 확인 항목: `router.fc.bias` 이동, 모듈별 gate 통계.

**5단계 평가**: 생성 전 `--set eval.num_images=2 eval.images_per_condition=1 --run-name smoke`로 `gate_hook_calls` 확인 → 구성당 in_domain 500 + weather 4×100(약 7.5시간/구성) → CMMD + CLIP-Score.

**열린 질문**: (a) [500,999] expert의 대안(draft 3.2(B) 집계, random-sliced, 미적용)은 C2 결과를 본 뒤 결정. (b) 최종 expert로 paired loss 재확인 여부. (c) n1_i0 재학습 여부.
