# CA-LoRA 재현·최적화 프로젝트 로그

> 이 문서는 **다른 세션의 모델(리포트 작성: Sonnet 5)이 이 프로젝트의 전모를 파악하기 위한 단일 진입점**이다.
> 파이프라인 구조, 연구·구현 결정과 그 근거를 기록한다. Fable 5가 수시로 갱신한다.
> 역할 분담: **Fable 5 = 계획·리뷰·이 로그 관리 / Opus 5 = 구현 / Sonnet 5 = 실험 후 리포트 작성 / 사용자 = 연구 결정, 실험 실행.**

## 1. 프로젝트 개요

- 대상 논문: **Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation** (M. Park et al., CVPR, arXiv:2503.22172). 공식 코드 미공개 → 완전 재구현.
- 논문의 CA 정의: `CA(θ) = E[‖∇θ L_Concept‖ / ‖∇θ L_Diff‖]`,
  `L_Concept = ‖ε_θ(x_t,c,t) − sg[ε_θ(x_t,c_Aug,t)]‖²`, `L_Diff` = 표준 diffusion loss.
  논문은 **이론상의 head 단위**(layer × head × Q/K/V/OUT)로 CA를 집계·선택하고, **상위 2%는 head 단위 개수 기준**("top-k% of projection layers"), LoRA rank 64는 모듈 수준. type A = OUT(입력 열 슬라이스), type B = Q/K/V(출력 행 슬라이스) — 정확한 파라미터화는 §5 해소 항목 참조 (2026-07-30 논문 Appendix A.2·Algorithm 2–4로 확정). **우리 구현은 fused 모듈 단위(560개)로 집계·선택하는 의도적 변형** — 아래 용어 규약 참조. 논문은 t 후보 {1,81,201,481} 중 style-CA와 viewpoint-CA 분포 차이가 가장 큰 **t=81을 휴리스틱으로 선택**.
- **용어 규약 (2026-07-30 명문화)**: 이 프로젝트에서 "projection"은 항상 **구현상의 fused `nn.Linear` 모듈**(attention 모듈당 to_q/to_k/to_v/to_out.0 각 1개, SDXL UNet에 총 560 = 70 block × 2 attn × 4 kind)을 뜻한다. head 단위 개념은 측정·선택·LoRA 부착 어디에도 없다 (`models/projections.py`가 유일한 정의처). LLM과 달리 UNet은 채널 폭(320/640/1280)·cross-attn K/V 입력(2048)이 달라 projection 크기가 ~26배 편차 → 2% 파라미터 예산에서 선택 개수가 arm마다 10~13개로 가변. 논문의 단위는 이론상의 head(2026-07-30 사용자 확인)이므로 **기존 모듈 단위 구현은 논문보다 거친 granularity였음** (사용자 판정: 오류 → head 재편, §4 참조). 기존 구현의 LoRA는 fused 행렬 전체에 표준 LoRA — 논문의 type A/B(슬라이스 부착) 어느 쪽도 아닌 전체 부착이었다. 함의: (a) 우리 CA는 논문 CA의 head-합산이라 정의는 일관되나 논문과 동일한 선택 집합은 원리적으로 재현 불가, (b) 프로젝트 기여 3종(t 정량화·다중 t 집계·drift)은 모듈 단위로 자기 일관적이라 arm 간·t 간 내부 비교는 무영향, (c) 리포트의 재현 주장은 "모듈 단위 변형에서의 상대 순위 재현"으로 명시할 것.
  1. **t 선택의 정량화** — CA는 norm 비율이라 gradient 방향이 소실됨. 방향 보존 통계량 `ĝ = ∇L_Concept/‖∇L_Diff‖`를 style/viewpoint 각각 측정, projection별 교차축 코사인 유사도로 t를 평가. 가설: *유사도가 낮으면서 CA가 높은 projection이 많은 t가 좋은 t*.
  2. **다중 t 집계 projection 선택** (main) vs 단일 t (ablation).
  3. **학습 중 CA 안정성** — 일정 step마다 CA top-k를 재측정해 선택 집합의 drift(Jaccard, Spearman)를 측정.

## 2. 파이프라인 (코드 기준)

```
scripts/measure_ca.py      →  t별·축별 CA + 방향 통계 (ca/measure.py, ca/stats.py)
scripts/score_timesteps.py →  t 스코어링·순위 (ca/select.py: rank_timesteps)
scripts/select_projections.py → top-k 선택, ablation arm별 selection JSON
scripts/train_lora.py      →  선택된 projection에 LoRA 학습 + drift 콜백 (train/lora.py, analysis/stability.py)
scripts/smoke_test.py      →  전체 파이프라인 CPU end-to-end
```

- 모델은 config 주입: CPU 개발 = toy UNet(diffusers와 동일한 모듈 네이밍), A100 실험 = SDXL. 테스트 109개 전부 CPU에서 수 초 내 통과.
- 핵심 불변식 (측정의 유효성 조건):
  - **batch_size=1**: batch>1이면 추정량이 `E[‖Σg_c‖/‖Σg_d‖]`로 바뀜(논문의 양이 아님).
  - **측정은 항상 crop="center"**: t·축 간 페어링을 위해 3개 측정 호출부 모두 명시 강제.
  - **페어링 보장**: 모든 축은 같은 x_t·같은 prediction 공유; 모든 t는 같은 noise 추첨 공유(seed가 t에 비의존).
- 유사도 두 종류: `cos_of_means` = cos(E[ĝ_a], E[ĝ_b]) (샘플별 1/‖g_d‖ 가중이 내재, t-의존) vs `mean_of_cos` = E[cos(ĝ_a, ĝ_b)] (순수 방향 일치, 정규화에 불변). **항상 둘 다 로깅.**

## 3. 결정 로그 (날짜, 결정 주체, 근거)

### 구현기 (2026-07 중순, 계획·구현)

- **[사용자] 재현 범위**: CA 측정 + LoRA 학습 + CMMD/CLIP-Score까지. mIoU는 인터페이스만 두고 추후 결정.
- **[사용자] 데이터**: 논문대로 Cityscapes. 캡션은 템플릿("a photo of a city street scene in {city}") fallback.
- **[사용자] 환경**: 구현은 로컬 CPU(Windows), 실험은 A100 1장. → CPU/GPU 이원화 설계(toy 모델, config 교체만으로 SDXL 전환).
- **[Fable] `mean` reduction 사용**: 논문의 ‖·‖²(sum) 대신 mean — CA는 두 norm의 비율이라 상수배가 소거되므로 통계량 불변, fp16 range 보호 목적.
- **[Fable] t 스코어 3종 동시 리포트**: S(t)=Σ CA_norm·(1−sim), selected_mean_similarity, conflict_count — 실패 양상이 서로 달라서.
- **[Fable] conflict 임계값은 전체 t에 pooled quantile**: t별 quantile이면 conflict_count가 모든 t에서 상수가 되어 지표가 무의미해짐.

### 리뷰 라운드 (2026-07 하순, Fable 리뷰 → Opus 수정, 10건)

- **실버그 1건**: `select_topk` 예산 초과 버그 — 최고점 projection이 예산을 무시하고 통과(SDXL projection 크기 편차 320²~1280×2048이라 실질 영향). 초과 시 skip으로 수정.
- 측정 경로 crop="center" 강제, accumulator 원자성(`_pending_directions`), fp16 CUDA 요구 검증, latent 캐시(수치 동일성 테스트로 보증), smoke_a100을 dummy 데이터로 실행 가능하게 등.
- **테스트가 잡은 성질**: `cos_of_means`는 방향이 t-불변이어도 t-의존일 수 있음(1/‖g_d‖ 가중 때문). 버그가 아니라 지표의 성질로 문서화.
- drift 베이스라인은 스냅샷과 **같은 샘플로 step 0에서 재측정**(추정 노이즈와 진짜 drift 분리). 별도로 `jaccard_vs_trained`.

### 실험 설계 확정 (2026-07-28, 사용자 결정)

- **[사용자] 유사도 주 지표 = `cos_of_means`**. 근거: (a) 1/‖g_d‖ 가중이 CA의 정규화 철학과 일관, (b) 학습은 batch 평균 gradient를 쓰므로 평균 후 cosine이 학습 알고리즘과 호환. `mean_of_cos`는 로깅 유지 — cos_of_means의 t-의존성이 방향 신호인지 가중 artifact인지 리뷰어 방어용 대조군.
- **[사용자] S(t)는 주 지표에서 강등**. 근거: projection 선택은 binary(rank 64 고정, CA 크기는 선택 후 버려짐)인데 S(t)는 CA 크기를 연속 가중치로 유지 → 결정 문제와 지표의 불일치. CA 가중평균이라는 *방식*만 차용.
- **[사용자] 주 지표 2종으로 ablation**: (1) **top-n·k CA-가중 selected mean similarity** (n≈4–5; top-k만 쓰면 선택 경계의 편향 우려, n·k로 확장하되 CA 가중으로 완화), (2) **conflict_count**. 둘 다 돌려보고 좋은 쪽으로 리포트.
- **[사용자] projection 선택은 순수 CA top-k 유지** (방향 지표는 t 선택에만 사용). 근거: 1차 목표가 t 최적화였고, similarity 높은 projection 배제는 원 논문 접근에서 과도하게 이탈. 유사도·CA 둘 다 높은 projection(예: 이미지 세부 정보에 관여해 두 축 모두에서 CA가 큰 경우)도 fine-tuning 가치가 있음.
- **[사용자] conflict quantile sweep = {0.1, 0.2, 0.3, 0.4, 0.5}** (primary 0.3, Fable 제안). t 순위가 quantile에 따라 뒤집히면 conflict_count는 주 지표 자격 상실 — t 스윕 직후 최우선 확인 항목.
- **[Fable 제안, 사용자 승인] split-half 안정성**: 샘플을 짝/홀로 나눠 각각 t 순위 산출 → 순위 일치도(Spearman, top-5 overlap)로 t 선택의 통계적 신뢰성 검증. full bootstrap은 방향 벡터 저장 비용 때문에 기각.
- **[사용자→Fable 제안] 샘플당 augmentation 다중화**: `augs_per_sample=4` (축당 8개 템플릿에서 비복원 추첨). 근거: grad forward + 분모 backward가 샘플 내에서 공유되므로 aug 추가의 한계비용이 낮음(m=4가 m=1 대비 약 3배 비용으로 4배의 concept-gradient 표본). 주의: 같은 샘플 내 기여는 상관 → split-half는 **샘플 단위**로 분할(기여 단위 아님).
- **[사용자] 실험 장비**: RunPod A100 80GB (PCIe/SXM 중 택1, 스토리지가 데이터센터에 묶여 변경 불가). Cityscapes 계정 승인 완료.

### 확장 구현 완료 (2026-07-28, Opus 5 구현 → Fable 검증)

- 스펙 `docs/specs/2026-07-28-measurement-extensions.md`의 A–E 전 항목 구현. **테스트 109 → 142개 전체 통과, smoke test 통과** (m=2·split-half 경로 실제 실행, agreement 블록 출력 확인).
- 새 config 키: `measure.augs_per_sample`, `measure.split_half`, `select.conflict_sim_quantiles`, `select.conflict_primary_quantile`, `select.tscore_selection_multiplier`. 제거: `select.conflict_sim_quantile`.
- `rank_timesteps`의 기본 rank_by가 `expanded_weighted_similarity`로 변경됨(주 지표 결정 반영). `score_timesteps.py`는 한 실행에서 양쪽 similarity source × rank 기준 3종 + quantile sweep + split-half agreement를 모두 JSON으로 저장.
- 구현 중 발견된 회귀 수정: `select_projections.py`의 측정 파일 glob이 half 파일을 집어삼켜 full 측정을 덮어쓸 수 있던 문제 — half 파일 필터링 추가.
- 수치 주의: 방향 버퍼는 fp32라 병합 시 합산 순서에 따른 last-ulp 차이 존재 → 병합 정합성 테스트에서 direction·cos_of_means는 rtol=1e-6, fp64 통계(CA 등)는 1e-10로 검증.

## 4. 실험 매트릭스 (Phase 4 계획)

| arm | projection 선택 기준 |
|---|---|
| A (재현) | 논문 t=81 단일 CA |
| B (main) | 상위 5개 t 집계 CA (normalized_mean) |
| C | 최고 t 단일 CA — t 순위는 expanded weighted similarity 기준과 conflict_count 기준 각각 |
| D×5 | 상위 5개 t 각각 개별 top-k |
| E | 전체 attn LoRA / random top-k (동일 파라미터 예산) |

재현 판정 (2026-07-30 재정의, §5 해소 항목 참조): ~~CA top-2% > 전체 LoRA~~ → **논문 Fig. 6의 축 간 대비 시그니처 재현** — style 어댑터는 CMMD 우위(단 CLIP-Score 붕괴), viewpoint 어댑터는 날씨 CLIP-Score 보존. 절대 수치 일치는 목표 아님. (E_all 제외 결정으로 full LoRA 비교는 스코프 밖; 위 표의 arm 구성도 head 재편 후 §5의 6-run 매트릭스로 대체.)

### c_Aug 템플릿 확정 (2026-07-28, Fable 분석 → 사용자 확정)

`style` 축 8개는 원안 유지. `viewpoint` 축 8개를 전면 교체 (`prompt_augmentations.json` version 2). 근거:

- **의미 오염 제거**: `bird's-eye`의 "bird", `above the intersection`의 "intersection"이 CLIP 텍스트 임베딩에서 의미 질량을 가져, `L_Concept`의 차이가 카메라 기하가 아닌 *장면 내용* 변화에서 발생. (주의: c_Aug는 생성에 쓰이지 않고 no-grad forward 1회에만 쓰이므로 "새가 그려지는" 문제는 아님 — 임베딩 층위의 confound.)
- **축 간 섭동 크기 비대칭 완화**: style 템플릿은 전부 동일 장면의 on-manifold 재렌더링인데 항공 시점은 도심 주행 장면에서 off-manifold. 비대칭이 크면 viewpoint-CA가 개념과 무관하게 부풀어 style/viewpoint 대비(= t 선택의 근거 전체)를 오염시킴. 극단 고공 템플릿 2개 → 완만한 1개.
- **광학 템플릿 1개로 축소**: wide/telephoto는 pose가 아닌 광학이라 축 정체성을 흐림.
- **1인칭 템플릿 의도적 제외** (사용자가 제기했으나 Fable 반대 의견을 수용): Cityscapes가 전부 1인칭이라 그 템플릿만 조건을 실제 이미지 쪽으로 끌어당김 → 축 내부 방향 평균 `E[ĝ_viewpoint]`에서 나머지 슬롯과 상쇄되어 주 지표 `cos_of_means`를 불안정하게 만듦 (m=4 비복원 추첨이라 뽑힐 확률 1/2). ego 관점 커버리지는 후방 시점 템플릿이 담당.
- **[Fable 제안] `direction_concentration` 지표 추가**: `‖E[ĝ_axis]‖ / CA_axis ∈ [0,1]` — `E[‖ĝ‖]`가 곧 CA이므로 이미 저장 중인 값에서 **추가 측정 비용 0**으로 파생. 축 내부 템플릿들의 방향 일관성을 나타내며, 낮으면 템플릿 세트가 서로 다른 방향으로 밀고 있다는 뜻이라 `cos_of_means` 해석에 경고가 됨. A100 첫 측정에서 템플릿 세트 품질을 즉시 판정하는 용도.

구현 완료·검증됨 (Opus 5 구현 → Fable 재검증): **테스트 148개 전체 통과**, smoke test에 t별 concentration 출력. `direction_concentration`은 `MeasurementResult`의 메서드이며 직렬화하지 않음(저장 값에서 파생). `measure_summary.json`에는 `direction_concentration_mean: {axis: float}`로 포함.

> **해석 주의**: smoke test의 concentration 값(~0.3)은 랜덤 가중치 toy 모델의 노이즈 바닥이므로 의미 없음. 실제 판정은 A100의 SDXL 첫 측정에서 할 것. 랜덤 방향 기준선은 대략 1/√(기여 수)이므로, 실측값이 그보다 유의하게 높아야 템플릿 세트가 일관된 방향으로 밀고 있다는 뜻.

### 논문 Appendix A 확인 및 평가 프로토콜 확정 (2026-07-28, Fable 조사 → 사용자 결정)

논문 PDF(arXiv 2503.22172, 28p)의 Table 12·Appendix B.2·Table 15에서 미확정값 대부분 해소.

**fallback이 논문과 일치했던 것**: lr 1e-4, batch 1, AdamW(0.9/0.999, wd 0.01), constant scheduler, rank 64, 10k steps, 해상도 1024². **CA 측정 샘플 수는 논문에도 미기재** → 256 유지, 리포트에 "논문 미기재, 우리 선택"으로 명시.

**논문에서 확인되어 수정한 것 (Table 12)**:
- 학습 캡션은 도시별 템플릿이 아니라 **고정 프롬프트 "photorealistic first-person urban street view"** → config 교체.
- 데이터 증강: random crop + **random horizontal flip** → hflip 구현 추가 (학습 전용; 측정 호출부는 crop="center"와 함께 hflip=False 강제 — 불변식 확장. latent 캐시는 두 방향 모두 인코딩: VAE(flip(x)) ≠ flip(VAE(x))).
- gradient accumulation 4 → **1** (논문 batch 1, 누적 언급 없음).
- 생성(test-time): **25 inference steps, guidance 5.0**. 테스트 프롬프트: in-domain "... with [Class names]"(label map에서 추출), DG "... in [Weather] with [Class names]".

**[사용자 결정] c_Aug를 논문 세트로 통일** (v2 폐기, version 3): 논문 Appendix B.2의 축당 5개 세트 verbatim — style은 화풍(Sketch/Watercolor/Pop-art/Lineart/Oil painting), viewpoint는 top-down/high/low/**eye-level**/close-up. 재현 충실도를 우선한 선택.
- **긴장 관계 기록**: v2에서 1인칭(≈eye-level)을 제외했던 근거(축 내부 방향 상쇄로 `E[ĝ]` 불안정)가 논문 세트에는 적용된다. 단, **논문은 CA(norm)만 쓰므로 상쇄가 무해하고, 방향 지표를 쓰는 것은 우리뿐** — eye-level의 실제 영향은 `direction_concentration`이 실측으로 판정한다. viewpoint의 concentration이 style보다 유의하게 낮으면 그 자체가 "방향 기반 지표에는 템플릿 설계가 중요하다"는 발견으로 리포트 가능. v2 세트의 설계 논리는 이 로그(위 항목)에 보존됨.

**평가 프로토콜 확정** (논문 값 + 미기재분은 우리 결정으로 명시):
- 생성: in-domain 500장(CMMD용, 논문 few-shot 수), 날씨 4종(foggy/night-time/rainy/snowy) × 100장(CLIP-Score용, 논문 프로토콜), guidance 5.0, 25 steps, per-image seed 파생.
- CMMD reference: **Cityscapes val 500장** center-crop (논문 미기재 — val split이 정확히 500장이라 자연스러운 선택).
- CLIP 백본: ViT-L/14 (논문 미기재 — CMMD 정의와 문헌 표준).
- class-name 프롬프트는 재현하되 label map 조건 생성(label generator)은 스코프 밖 — 프롬프트 수준만 재현임을 리포트에 명시.
- 구현: `EvalConfig` + `scripts/generate_images.py` + `scripts/evaluate.py` (스펙: `docs/specs/2026-07-28-paper-alignment-and-eval.md`).

구현 완료·검증됨 (Opus 5 구현 → Fable 재검증): **테스트 194개 전체 통과**(148→+46), smoke test가 생성(7단계)·평가(8단계)까지 실행하고 eval.json을 산출. 구현 중 확정된 세부 설계:

- **flip 추첨은 데이터셋이 아닌 학습 루프에서** (`derive_generator(seed, "train-hflip", step, micro)`): 데이터셋 안에서 뒤집으면 `__getitem__`이 비결정론이 되어 latent 캐시가 무효화됨. 데이터셋의 `hflip`은 허용 선언, 캐시는 두 방향 각각 인코딩. 캐시 유무 수치 동일성 테스트로 고정.
- **[Fable 결정] 학습 crop을 `random`으로** (Table 12의 증강 완전 재현): latent 캐시가 꺼져 step당 VAE 인코딩이 추가되지만(A100에서 arm당 수십 분 추정) 문서화된 증강을 빼면 재현 주장이 약해짐. 속도가 문제 되면 한 줄 revert 가능하나 그 경우 리포트에 편차로 명시할 것. 측정 경로는 어차피 center/no-flip 강제라 무관.
- **Opus가 잡은 실버그**: LoRA `alpha`는 state dict에서 복원 불가능한 순수 배율이라, 체크포인트에 메타데이터(`lora_alpha`/`lora_rank`)를 기록하지 않으면 다른 alpha로 재적재해도 **정상 실행되며 조용히 다른 모델을 평가**하게 됨 → `save_lora`가 기록, `load_lora`가 우선 사용. 또한 `load_lora`는 selection JSON이 아니라 체크포인트 키에서 projection 이름을 복원(가중치가 사실의 원천).
- 오프라인 `PixelEmbedder`는 플러밍 검증 전용(`reportable=False`로 eval.json에 표기) — 리포트 수치는 반드시 CLIP 백본으로.

## 5. 미해결 항목
- [x] ~~실험 메인 세션 인수인계~~ → **완료 (2026-08-03): 8 run 전체 학습·생성·평가 종료** ([HANDOFF_ARM_TRAINING.md](HANDOFF_ARM_TRAINING.md) 절차대로, 중단 조건 발동 0회).
- [x] ~~E arm 스코프 재편입 결정의 소급 기록~~ → **해소 (2026-08-07, 사용자 확정)**: 재편입 근거 = **음성 대조 필요성**(CA 선택의 가치를 변별하려면 full/random 대조가 필요 — 동기) + **8 run 종료 후 GPU 시간 여유**(실행 조건). 08-04 실험 세션이 head 단위로 재실행. **E_all 완료 확인 (2026-08-07, 보고서 세션)**: 900장 생성·평가 종료 — CMMD 0.596(전체 최저), weather CLIP 20.65(foggy 24.31/night 18.87/rainy 21.53/snowy 17.87 — style급 붕괴), drift final Spearman 0.638(min 0.575)·final Jaccard 0.037. REPORT_DRAFT §6.5/§7/§10 갱신 완료.
- [ ] arm당 학습 seed 수: 기본 1, GPU 시간 여력 시 main arm(A, B, E)만 3 seed.
- [ ] A100 smoke run에서 확인: gradient checkpointing×retain_graph(`use_reentrant=False`, 로컬 검증 완료), 캡션 캐시의 인코딩 경로 동등성(per-caption encode vs batch encode — CLIP `padding="max_length"`면 동일), 측정 dtype 정책(bf16 forward + fp32 누적 vs 순수 fp32).
- [ ] mIoU 다운스트림 평가 여부.
- [ ] measure_ca에 저장 직후 재로드 검증 추가 (t=41 파일 손상 사고 후속, 2026-07-31 — 네트워크 스토리지에서 쓰기 손상 방어).
- [ ] **[미재현] 논문 Fig. 10/11의 Q/K≫V/OUT kind 대비 — 그려진 양의 관례 미상** (2026-08-05, 사용자 요청 KLD·히트맵 분석 중 발견, 실험 세션): head 측정값으로 재현한 Fig. 11식 히트맵(`docs/figures/ca_heatmap_qkvout_t*.{pdf,png}`)에서 kind별 CA share가 평평(V=1 기준 Q 0.88 / K 0.96 / OUT 0.95). raw ‖g_c‖는 강한 kind 편향이 있으나 방향이 논문 그림과 **반대**(V/OUT ≈ 3× Q/K)이고, per-unit 비율(Algorithm 2 = 우리 구현)이 이를 상쇄함을 확인 — base gradient를 층 평균/전역 평균으로 거칠게 바꿔도 V/OUT만 진해질 뿐 논문의 Q/K-진한 패턴은 어떤 정규화로도 재현 불가. 공식 코드 미공개라 확정 불가 → 리포트에는 "시각화 관례 미기재로 재현 한계, 파이프라인 유효성은 E2E(Fig. 6 시그니처 4/4)로 별도 입증"으로 기술할 것.
- [x] ~~control_base weather 4×100장 생성 + 재평가~~ → **완료 (2026-08-04, 실험 세션)**: eval.json이 900장 전체 커버 — weather_mean 31.68 (foggy 32.86 / night 30.26 / rainy 32.03 / snowy 31.56). Fig. 6 대비의 0% 기준선 확보.
- [x] ~~선택 축과 평가의 대응~~ → **[사용자 결정 2026-07-30, 논문 재확인으로 구체화] 논문 §4.3 Fig. 6 구도 그대로.**
  - **논문 사실관계**: ① CMMD·CLIP-Score는 Style/Viewpoint CA-LoRA **양쪽 모두에** 측정 (Fig. 6 — 용도 분리는 main mIoU 결과에서의 사용 방식이고 분석은 양축 공통 지표). ② 세부 템플릿(화풍 5종 등)은 Eq. 7에서 **평균되는 확률 변수** — 어댑터는 축당 1개, 템플릿당이 아님.
  - **[사용자] E_all·E_random 제외** (논문 ablation 전체를 따라가지 않음 + GPU 절약). 이에 따라 **재현 판정 재정의**: "CA top-2% vs full LoRA"가 아니라 **Fig. 6의 축 간 대비 시그니처** — style 어댑터는 CMMD 우위(+CLIP-Score 붕괴), viewpoint 어댑터는 CLIP-Score(날씨 제어) 보존. (논문에서도 full LoRA가 CMMD 최고라 CMMD/CLIP 범위에서 "CA > full" 주장은 원래 불성립 — mIoU 스코프 밖.)
  - **arm 매트릭스 (학습 6 run)**: A(t81)·B(top5 집계)·C(최고 t) × {style, viewpoint} 어댑터 각 1개, 전부 CMMD+날씨 CLIP-Score 양쪽 평가. 0% 컨트롤(base SDXL)은 무학습. D×3은 head 재측정 후 t 랭킹 보고 필요시 추가. t 랭킹은 교차축 유사도 기반 축 공통 1개, 축별 CA는 선택 단계에서만.
- [x] ~~granularity 편차 대응 확정~~ → **사용자 판정(2026-07-30): 오류. head 단위로 전면 재편** (§4 재작업 결정 참조).
- [x] ~~type B LoRA 논문 세부 확인~~ → **해소 (2026-07-30, Fable이 논문 PDF Appendix A.2 + Algorithm 2–4에서 직접 확인)**:
  - **파라미터화**: LoRA는 **모듈당 1개**, 선택된 head들의 합집합을 커버. Type B(Q/K/V, 행=출력 슬라이스): A는 풀 사이즈(r×d_in) 공유, B는 선택 head 행만(|선택|·d_h × r), forward에서 `out[:, :, indices] += scale·lora(x)`. Type A(OUT, 열=입력 슬라이스): B는 풀 사이즈(d_out×r), A는 선택 열만 읽음(`lora(x[:, :, indices])`). 즉 type A/B는 "전체 vs 부분"이 아니라 **OUT(입력 열) vs Q/K/V(출력 행)의 슬라이스 방향 구분**. 구 diffusers `LoRACompatibleLinear` forward 몽키패치로 구현돼 있음 — 우리는 peft 기반이므로 등가 커스텀 모듈 필요.
  - **rank = 64는 이 모듈 수준 LoRA에 적용** (원 LoRA와 동일 고정). head별 rank가 아니므로 "full-rank 퇴화" 우려는 해소.
  - **[중요·추가 편차 발견] top-2% 예산은 파라미터가 아니라 head 단위 개수 기준** — 본문 "top-k% of projection layers", Fig. 4 proportion=2/16(레이어 수 분율), sweep 1/2/3/5/10%. 우리 기존 `attn_params` 예산은 두 번째 구현 편차였음 → head 재편에서 `budget_basis=unit_count`를 기본값으로 (스펙 §3 반영, Opus 통보 완료).
  - **측정 코드도 논문과 정합 확인**: Algorithm 2가 `torch.chunk(grad, heads, dim=1 if to_out else 0)` 후 chunk별 RMS(`(g**2).mean().sqrt()`) — 우리 스펙의 행/열 슬라이스 방향·mean reduction과 일치. per-sample 비율 후 평균(E[ratio])도 일치.

## 6. 실험 환경 (RunPod pod, 2026-07-29 점검 완료)

GPU 투입 전 점검에서 환경 전체를 구축·검증했다. 상세와 실행 순서는 **[RUNBOOK_A100.md](RUNBOOK_A100.md)** 참조. 요점:

- venv `/workspace/venv-calora` (torch 2.9.1+cu128, diffusers 0.39.0, transformers 5.14.1, peft 0.20.0). `/workspace`만 재배포에서 살아남으므로 환경·HF캐시·데이터 전부 그 안에 둠.
- Cityscapes `/workspace/data/cityscapes` 추출·CRC 전수 검증 완료 (train 2975/val 500/test 1525). SDXL base + CLIP ViT-L/14 프리다운로드 완료 (fp16 로드 경로 검증됨).
- **중대 발견 1 (해소됨)**: `.gitignore`의 `data/` 패턴이 `src/calora/data/` 패키지를 삼켜 initial commit·원격 모두에 데이터 모듈이 없었음 → 사용자가 로컬에서 push(49161eb)로 해소. 이후 pod에서 **194개 테스트 전체 통과, smoke test 통과, 실측 Cityscapes 로딩(2975/500) 검증 완료** (2026-07-29).
- **중대 발견 2**: mfs 볼륨이 쓰기를 조용히 손상시킴 (venv 6파일, gtFine 2파일 — 전부 해시 검증으로 복구). **이후 모든 대량 쓰기(체크포인트 포함)는 저장 후 검증이 원칙.**
- 컨테이너 cgroup 메모리 한도 8GB — fp32 SDXL 로드(~13GB)가 불가능하므로 GPU pod 배포 시 한도 상향 필수. → **A100 부착 시 250GB로 배포되어 해소 (2026-07-29).**

### GPU day 1 (2026-07-29, A100 SXM 80GB 부착 후)

- **환경 함정 2건 추가 발견·해소** (RUNBOOK에 상세): (a) 재배포로 `HF_HOME` 환경변수가 사라짐 → 모든 실행에 명시 필요. (b) 128코어에서 torch 기본 스레드 128개 → 작은 텐서 CPU 연산이 수백 배 느려짐 (pytest 34s→2h17m 실측) → `OMP_NUM_THREADS=16` 강제.
- **학습 가시성 기능 추가** (사용자 요청, Opus 구현·Fable 리뷰): 학습 로그에 s/step·ETA 표시, 종료 시 `loss_curve.png` 자동 생성 (`calora.utils.plotting.plot_loss_curve`; matplotlib 없으면 조용히 생략, history 스키마 불변). 테스트 6개 추가.
- **호환성 버그 수정**: transformers 5.14.1에서 `CLIPModel.get_image_features/get_text_features`가 텐서 대신 `BaseModelOutputWithPooling`을 반환 (투영된 768차원 임베딩은 `pooler_output`에 있음 — 1024가 아니라 768임을 실측으로 확인). `eval/embeddings.py`에 `_features()` 헬퍼로 4.x/5.x 양쪽 지원. CPU 테스트는 스텁 임베더(PixelEmbedder)만 쓰므로 GPU에서 실 CLIP 첫 실행 때 드러남. 테스트 4개 추가 (총 204개 통과).
- **A100 스모크 파이프라인 (smoke_a100.yaml) 6단계 전체 통과**: fp32 SDXL 측정 (GPU ~35.6GiB, 8샘플 51.8s), 학습 20스텝 (~0.9s/step, drift 콜백 정상), 1024px 생성 (25스텝 환산 ~1.2s/it), 실 CLIP 평가. 체크포인트·PNG는 mfs 원칙대로 재로드 검증함.
- §5의 gradient checkpointing×retain_graph, 캡션 캐시 경로: A100 스모크에서 문제없이 동작 확인. 측정 dtype 정책: 본 측정(tsweep)을 config 기본값 fp32로 시작 (VRAM 여유 충분: 스모크 실측 53GiB > 여유 있음). bf16 전환 논의는 필요 시 재론.
- **본 측정(tsweep_sdxl, 13 t × 256샘플 × m=4) 시작** — 스모크 실측 속도 기준 예상 ~12h.
- **측정 중 OOM 버그 발견·수정** (Opus 구현·Fable 리뷰): measure_ca.py 루프가 직전 t의 accumulator(GPU ~35.6GiB)를 쥔 채 다음 t를 할당해 2벌 공존 → 80GB 초과. 루프 끝 참조 해제 + `empty_cache`로 수정. t=1 결과는 무사해 `--skip-existing`으로 재개, t=1의 summary 항목은 저장 파일+로그값으로 복원 (direction vector는 비직렬화라 concentration만 로그의 4dp 값 사용).

### 본 측정 결과 (tsweep_sdxl, 2026-07-29 완료 — Phase 3의 핵심 산출)

13개 t 전체 측정 완료 (t당 43.7분, 총 ~9.5h). 사전 등록된 3대 판정:

1. **direction_concentration vs 1/√n 바닥 (≈0.044): 통과.** style 0.289~0.663, viewpoint 0.204~0.443 — 전 구간 바닥의 5~10배. eye-level 템플릿 상쇄 우려(§4 긴장 관계)는 실측상 문제 없음. **U자형 프로파일**: 양 극단 t(1, 981)에서 집중도 최고, t=601 부근 최저.
2. **conflict quantile sweep {0.1..0.5}: 순위 불안정 → 사전 규칙대로 conflict_count는 주 지표 자격 상실, 보조 지표로 강등.** t=41 vs t=81 순위가 q=0.1(81>41)과 q=0.3(41>81)에서 역전. split-half spearman도 0.646으로 지표 중 최저 — 일관된 결론.
   - **강등 기제 (리포트용 정리, 2026-07-30)**: 형식적 사유는 사전 등록 규칙(quantile 간 순위 역전 시 자격 상실)의 발동이고, 두 증상(quantile 역전·split-half 불일치)의 공통 원인은 지표의 구조 — **hard threshold(τ) × 작은 정수 count(0~9)**. projection 1개가 τ 경계를 출입하면 count가 ±1 변하는데 값 범위 대비 거대해 순위가 통째로 흔들리고, 13개 t 중 5개가 count 0 동률로 하위권 변별력도 없음. 같은 정보를 연속량으로 읽는 ews는 경계 출입이 CA 가중치만큼만 부드럽게 반영되어 안정(spearman 0.940) — "가설을 문자 그대로 지표화(count)"한 쪽이 "연속 완화(weighted mean)"한 쪽에 진 것이지, 가설 자체가 기각된 것은 아님(1위 t=41은 두 지표가 합의).
3. **split-half 일치도: 주 랭킹 신뢰 가능.** cos_of_means × expanded_weighted_similarity: spearman 0.940, 양쪽 절반의 top-3이 동일 집합 {41,1,81}. 종합 score는 top-5 overlap 1.000.

**t 랭킹 (best first): [41, 81, 1, 601, 481]** — 논문의 t=81은 2위, **1위는 t=41** (논문 후보 밖). 논문 후보 t=201은 13개 중 11위. C arm의 두 기준(유사도/conflict)이 모두 t=41을 지목해 C는 단일 arm으로 수렴.

### Phase 4 arm 확정 (스코어 대입 결과)

8개 run: A=t81(재현), B=top5 집계 [41,81,1,601,481], C=t41, D=t1/t601/t481 (D-41≡C, D-81≡A), E-all(560개 전체), E-random(동일 2% 예산, seed 0). 선택 JSON 8개 생성 완료 — CA 기반 arm 전부 1.97~2.00% 예산으로 일치. **A arm 학습 시작 (2026-07-29 22:29).**

**[사용자 지시 2026-07-30 추가]** 게이트 판정 이후 **B 제외 6개 arm(C, D_t1, D_t601, D_t481, E_all, E_random)을 추가 확인 없이 자동 진행** (train+generate+evaluate 전체). drift_autocast는 게이트 결과를 전 arm에 동일 적용.

**[사용자 지시 2026-07-30]** ① arm B는 사용자가 더 고민 후 결정 — **보류**, A 다음은 C부터. ② GPU 효율 개선: A 학습 중에는 계획만 수립하고, A 종료 후 코드 수정 → C 재개. 계획: [docs/plans/2026-07-30-vram-efficiency.md](plans/2026-07-30-vram-efficiency.md) (실측: VRAM 23%만 사용, 벽시계의 ~50%가 drift 측정. Tier 1 수치동일 최적화 + Tier 2 drift bf16(등가성 게이트) 적용, 학습 dtype은 A와의 비교성 위해 fp32 유지).

### arm B 확정: B-5 단독, B-3는 무학습 발견으로 기록 (2026-07-30, 사용자 제안 → Fable 분석 → 사용자 확정)

사용자가 B를 B-3(top-3 t 집계)와 B-5(top-5 집계)로 분할 제안 — 근거는 ews 랭킹의 3위(t=1, 0.390)/4위(t=601, 0.506) 갭이 13개 t 중 최대라는 것. Fable이 B-3 선택을 실제 생성(`selection_B_top3.json`, 예산 1.998%)해 검증한 결과 **B-3 학습은 기각, B-5 단독 유지** (보류 해제):

- **B-3 선택 집합은 A(t=81 단일)와 10개 중 9개 동일** — arm으로 학습하면 1개 projection 차이를 seed 1개로 시험하는 셈이라 노이즈와 구분 불가.
- **원인은 선택 집합의 클러스터 구조**: 상위 3개 t(41, 81, 1)의 개별 선택은 서로 6~8개 겹치는 단일 클러스터라 집계해도 클러스터 합의점(≈A)으로 수렴. 반면 t=601/481의 선택은 이 클러스터와 겹침 0~1개인 이질 집합 — top-5 집계만이 선택을 실제로 바꾼다(B-5 vs A = 8/10).
- **리포트용 발견 (무학습)**: ① top-3 집계는 논문의 t=81 선택을 9/10 재발견 → "ews 갭 위쪽 t들은 이미 projection 수준에서 합의돼 있어 집계가 불필요하고, 집계가 의미를 가지려면 갭 아래의 이질적 t가 들어와야 한다". ② pairwise 겹침 행렬(아래) 자체가 t 선택의 중요성에 대한 직접 증거: 낮은 t(1~81)와 중간 t(481~601)는 거의 서로소인 projection을 고른다 (D_t481 vs D_t601도 5개만 공유, E_random은 전 arm과 겹침 0).

| ∩ (10개 중) | A_t81 | B_top3 | B_top5 | C_t41 | D_t1 | D_t481 | D_t601 |
|---|---|---|---|---|---|---|---|
| A_t81 | — | 9 | 8 | 8 | 6 | 1 | 0 |
| B_top5 | 8 | 8 | — | 8 | 5 | 2 | 0 |

(D_t481은 12개, D_t601은 13개 선택 — projection 크기 편차로 동일 예산에서 개수 상이.)

### [중대] granularity 오류 판정 → 전면 재작업 결정 (2026-07-30, 사용자)

**사용자 판정: 모듈 단위 선택은 재현 편차가 아니라 오류.** 논문의 단위는 이론상의 head(모듈 행렬의 head 슬라이스)이며, 프로젝트는 head 단위로 재편한다. 조치:

- **진행 중이던 작업 중단**: A_t81은 학습이 이미 완료됐고 이미지 생성 단계였음 — 생성 프로세스(PID 94813, ~2.5h 경과) kill (2026-07-30 13:3x). **다른 세션(/loop, 6 arm 자동 진행 지시 잔존)은 이 세션에서 제어 불가 — 사용자가 직접 중지해야 함.**
- **무효화되는 것**: 모듈 단위 선택 JSON 9개(B_top3 포함), 모듈 단위 t 랭킹·arm 매트릭스, A_t81 학습·부분 생성 산출물. **단, 삭제하지 않고 보존** — `granularity: module` 변형의 데이터 포인트로 리포트에서 head 단위와의 비교(granularity ablation)에 재사용 가능. §4의 arm 확정·B-5 확정·t 랭킹 항목은 모듈 단위 결과로 재해석되며, head 단위에서 재도출 필요.
- **살릴 수 있는 것 (검증 완료)**: ① 측정 파일은 [560] 모듈 스칼라만 저장(방향 벡터 비직렬화)이라 head 수치 파생 불가 → **CA 측정은 전면 재실행(13 t × ~44분+)**. ② 그러나 측정 *코드*의 누적부는 weight-shape 버퍼라 대부분 유지되고 readout만 head 슬라이스로 재편. ③ t 그리드·c_Aug v3·평가 프로토콜·페어링/split-half 설계 그대로. ④ latent·caption 캐시는 단위 무관이라 재사용. ⑤ OOM 수정·VRAM 효율 계획 유효.
- **재편 스펙**: [docs/specs/2026-07-30-head-granularity-refactor.md](specs/2026-07-30-head-granularity-refactor.md) (Fable 작성, Opus 구현). **type B LoRA(부분 선택 행렬의 head-슬라이스 LoRA)는 논문 세부 확인 전까지 구현 보류** — 미해결 항목 참조.

### head 재편 구현 완료·검증 + head tsweep 시작 (2026-07-30 밤, Opus 구현 → Fable 검증)

- 스펙 `2026-07-30-head-granularity-refactor.md` 전 항목 구현. **테스트 230 → 270개 전체 통과(회귀 0), smoke test 8단계 통과** — Fable이 독립 재실행으로 확인. (주의: pytest는 `OMP_NUM_THREADS=8` 제한 필수 — 무제한 스레드면 경합으로 25초짜리가 30분+로 늘어짐.)
- **U = 10,400 확정** (모듈 560 × 블록별 H, 640ch=10/1280ch=20, d_h=64 전역, 파라미터 합 955,187,200으로 분할 완전성 검증). top-2% = ⌊0.02·10400⌋ = **208 단위**.
- 구현 요점: head 버퍼가 모듈 버퍼를 정확히 분할(메모리 동일), CA_head는 모듈 CA의 사후 분해가 아니라 head 수준의 E[‖g_c[h]‖/‖g_d[h]‖] (비율의 평균 ≠ 평균의 비율). 측정 파일에 `unit_granularity` 메타데이터, granularity 혼용 로드 즉시 에러. `budget_basis: auto` → head=unit_count, module=attn_params. 부분 선택 행렬 학습은 type A/B 구현 전까지 `NotImplementedError`로 명시 거부.
- **리포트용 주의**: unit_count 예산이라 실효 파라미터 분율은 고정이 아님 — 208단위 선택 시 어느 단위가 뽑히느냐에 따라 attention 파라미터의 약 1.4~2.7% (640ch head는 1280ch의 절반 폭, cross-attn k/v는 2048 폭). selection JSON에 파라미터 통계 병기.
- **head tsweep 시작 (2026-07-30 14:5x, 이 세션에서 — 사용자 지시)**: `runs/tsweep_sdxl_head/` (기존 모듈 측정 `runs/tsweep_sdxl/`과 분리 보존), 13 t × 256샘플 × m=4, fp32, 예상 ~10–12h. type A/B 학습 구현(Opus 2차)은 병행 진행.

### type A/B 슬라이스 LoRA 구현 완료·검증 (2026-07-30 밤, Opus 구현 → Fable 검증)

- 스펙 `2026-07-30-type-ab-lora-training.md` 전 항목 구현: `models/sliced_lora.py`(SlicedLoRALinear — 모듈당 1개, 선택 head 합집합, type은 `HEAD_SLICE_DIMS`에서 파생), 체크포인트에 head 인덱스는 int64 텐서로(`lora_head_indices`, 텐서가 사실의 원천) + `granularity`/`axis` 메타데이터, 불일치 로드 거부. drift의 `merged_lora`는 peft·sliced 양쪽 병합 지원.
- **테스트 295개 전체 통과, smoke test 통과** (head arm이 type A 2·type B 3·부분 슬라이스 2를 실제로 밟도록 확장, 체크포인트 왕복 포함) — Fable 독립 재실행 확인. 측정 경로(`ca/measure.py`, `ca/stats.py`)는 무변경 — 진행 중인 tsweep과 정합.
- 설계 재량 사항: A 초기화는 `train.lora.init_lora_weights` 따름(양 granularity 동일 규약), 메타데이터 없는 module 체크포인트는 기존 A arm 파일 호환 위해 경고 로드 유지(head는 거부), 생성 시 axis 검증은 `--expect-axis` opt-in(경고는 자동).
- 남은 학습 전 단계: head tsweep 완료 → 스코어링 → arm 확정 → 축별 selection 6종 생성. **tsweep 실측 속도: 샘플당 ~17.6s (head readout 오버헤드로 모듈 대비 +73%), t당 ~75분, 13 t 전체 ~16h 예상.**

### head tsweep 완료 + t=41 파일 손상·재측정 (2026-07-31 아침)

- **13 t 전체 측정 완료 (07:20, 총 ~16.4h — t당 ~75분, head readout 오버헤드 반영).** OOM 없음, VRAM 67GB 안정.
- **t=41 파일 손상 발견**: `t0041.safetensors`가 정상 크기(4.9MB)인데 safetensors 헤더 파손(스토리지 글리치 추정, RunPod 네트워크 볼륨) — 스코어링 로드에서 발견. half 파일은 정상이나 `cos_of_means`는 방향 버퍼(비직렬화) 없이는 half에서 재구성 불가 → **t=41만 재측정** (07:3x 시작, `--skip-existing`, seed t-비의존이라 페어링 보존). 손상 파일은 `corrupt_quarantine/`에 격리. 교훈: **측정 직후 전 파일 로드 검증을 measure_ca에 추가할 가치** (미해결 항목에 추가).

### head 단위 t 랭킹·selection 확정 (2026-07-31 아침 — Phase 3 재산출)

t=41 재측정 후 13개 파일 전부 로드 검증 통과, 스코어링 완료 (`runs/tsweep_sdxl_head/timestep_scores.json`).

- **t 랭킹 (ews, cos_of_means, best first): [41, 1, 81, 481, 601, 121, 801, 401, 901, 701, 201, 301, 981]** — **top-5 집합이 모듈 단위와 동일** {41, 81, 1, 481, 601}, 순서만 2↔3위(1↔81)·4↔5위(481↔601) 스왑. **1위 t=41은 ews·conflict_count 양 지표 재합의** → C arm = t41 유지.
- **split-half 안정성: ews spearman 0.956** (모듈 0.940 → 개선). **conflict_count 0.837** (모듈 0.646에서 대폭 개선) — count 값 범위가 0~9 → 8~120으로 커져 "hard threshold × 작은 정수" 이산성 문제가 head 단위에서 완화됨. **강등 사유의 기제 자체가 granularity 의존이라는 발견** — 리포트 가치 있음 (주 지표는 사전 등록대로 ews 유지, 재론 안 함).
- **selection 6종 생성** (`selection_{A_t81,B_top5,C_t41}_{style,viewpoint}.json`, `runs/tsweep_sdxl_head/`에 생성 후 `runs/sdxl_cityscapes/`로 복사): 전부 208 단위(unit_count 예산), 실효 파라미터 2.09~2.38%.
- **축 간 겹침**: 같은 arm의 style vs viewpoint 선택이 83~97/208(40~47%)만 공유 — 두 축이 실질적으로 다른 head를 고름 (Fig. 6 축 대비 실험의 전제 성립). arm 간(같은 축)은 159~165/208(76~79%) 공유 — 저 t 클러스터 유사성이 head 단위에서도 재현.
- **다음 단계: 사용자 t 랭킹 확인 게이트 → 실험 메인 세션에 학습 6 run 인수인계** (HANDOFF_ARM_TRAINING.md 시작 조건 충족 상태).

### [사용자 확정 2026-07-31] arm B = top-3 집계 [41, 1, 81] (top-5에서 변경)

학습 시작 전, head 랭킹 관찰 후 변경 (결과 누출 없음). 근거 (Fable 분석 → 사용자 확정):

1. **top-5 경계 방어 불가**: 5위 601(ews 0.5393)과 6위 121(0.5398)이 사실상 동률 — 601 포함/121 제외는 자의적.
2. **top-3 경계는 전 구간 최대급 갭 위**: 3위 81(0.383) → 4위 481(0.500), 갭 0.117.
3. **모듈 때의 B-3 퇴화(B3≈A 90%)가 head 단위에서 해소**: B3 vs A 81%, vs C 82% — A vs C 기준선(79%)과 동일 수준으로, 집계가 선택을 실질적으로 움직임 (10,400 단위의 해상도 덕).
4. 사전 등록(top-5)에서의 변경임을 리포트에 명시할 것. `selection_B_top3_{style,viewpoint}.json` 학습 디렉토리 복사 완료, `B_top5_*.json`은 분석용으로만 보존. HANDOFF 문서 갱신 완료.

### [사용자 확정 2026-07-31] Phase 4 최종 arm 매트릭스 — 학습 8 run

**A_t81(논문 재현) · B_top3(main, [41,1,81] 집계) · C_t41(최고 t) · D_t1(집계 분해의 미커버 t) × {style, viewpoint} = 학습 8 run** + 0% 컨트롤(base SDXL, 무학습, control_base 기존 생성물). E_all·E_random은 제외 유지. D_t1 추가로 B_top3의 완전 분해(41=C, 81=A, 1=D)가 성립 — B의 우위가 나올 경우 집계 효과 vs t=1 단독 효과를 가릴 수 있음. selection 8종(`{A_t81,B_top3,C_t41,D_t1}_{style,viewpoint}`) 전부 208 단위(2.09~2.38% params), 학습 디렉토리 복사 완료. HANDOFF_ARM_TRAINING.md 8 run으로 갱신. **인수인계 시작 조건 전부 충족 — 사용자가 실험 메인 세션에 시작 지시만 하면 됨.**

### 학습 전 최종 코드 리뷰 (2026-07-31, Fable — 사용자 요청)

head 재편 + type A/B 전체 diff(+2336/−172, 신규 4파일) 검토. **통과 — 학습 시작 승인.** 확인한 핵심 경로: ① 슬라이스 수학 정합(type B `out[rows] += BA·x` ↔ merge `W[rows,:] += BA`, type A 열 대칭 — forward/merge/unmerge 배치 일치), ② head 단위 gradient 접근은 `bind()`가 단위 이름→모듈 weight 참조 매핑으로 해결(복사 없음), ③ freeze 순서(`unet.requires_grad_(False)` → attach → `lora_` 파라미터만 optimizer) 덕에 커스텀 경로에서도 base 누출 없음, ④ 체크포인트는 키가 사실의 원천(`lora_head_indices` 텐서), granularity·axis 불일치 거부 + 메타데이터-키 교차검증, ⑤ drift `merged_lora`가 peft·sliced 양쪽 병합, 단위 이름 보존 테스트 고정, ⑥ unit_count 예산(동수 208, skip 규칙 불필요). 비차단 관찰 2건: sliced forward가 `*args`를 nn.Linear에 그대로 전달(구식 diffusers `scale=` 인자가 오면 시끄럽게 실패 — fail-fast라 수용), `load_lora`의 `int(lora_alpha)`는 alpha가 int인 현 config 타입에 의존. 구 A_t81 모듈 체크포인트는 head config에서 로드 거부됨(의도된 동작).

### A arm CMMD "과잉 성능" 조사 → 프로토콜 편차로 해명 (2026-08-01, 사용자 제보 → Fable 조사)

A_t81_style CMMD 0.638이 논문 2%(1.420)를 크게 상회하고 논문 100% full LoRA(0.644)와 동급이라는 사용자 지적. 조사 결과 **버그 아님, 학습 데이터 규모의 프로토콜 편차**:

- **캘리브레이션**: 0% 컨트롤을 우리 파이프라인으로 평가 → CMMD 4.078 (논문 0% = 5.063, 같은 자릿수 — CLIP 백본 224px·reference 선택 차이로 ~20% 오프셋, 스케일은 대체로 호환). 개선비: 논문 0%→2% 3.6배 vs 우리 6.4배 — 스케일 문제가 아니라 실제로 더 강한 적응.
- **원인**: 논문 Table 3/4는 **few-shot 0.3% 세팅** — T2I fine-tuning이 해당 서브셋(~9장, 10k step ≈ 1,100 epoch)으로만 수행됨 (Table 12에 데이터 규모 미기재, §4.4 세팅 기술에서 추론 — 신뢰도 높음). **우리는 전체 train 2,975장으로 학습** (~330배): 다양한 데이터의 도메인 흡수는 memorization 없이 CMMD를 크게 개선. 부수 증거: in-domain CLIP-Score 28.8(컨트롤)→22.1(A_style) — 도메인 흡수 방향 정합.
- **함의**: ① 8 run 내부 비교 무영향(전 arm 동일 데이터). ② 논문 Table 3/4 절대값과의 비교는 이중 무효(데이터 량+CMMD 상수) — 리포트에 "few-shot 세팅 미재현(mIoU 스코프 밖 결정의 귀결), T2I 학습은 전체 train" 편차로 명시. ③ 재현 판정을 Fig. 6 상대 시그니처로 잡은 것이 유효한 방어였음.
- 부수 정리: control_base 생성이 07-30 전면 중단 때 436/900에서 끊겨 있던 것 발견 → in-domain 500 완성·평가 완료 (eval.json 생성). weather 400장은 미생성 — 실험 세션 큐 종료 후 보완 필요 (§5 항목 추가).

### 날씨 붕괴(Fig. 6 시그니처) 3-run 중간 관찰 (2026-08-02, 사용자 질의 → Fable 분석)

완료 3 run(A_t81_style/viewpoint, B_top3_style; B_top3_viewpoint 진행 중) 기준 CLIP-Score:

| | in-domain | foggy | night-time | rainy | snowy | weather 평균 |
|---|---|---|---|---|---|---|
| A_style | 22.08 | 23.79 | 20.45 | 20.75 | 19.02 | 21.00 |
| A_viewpoint | 23.09 | 24.83 | 21.77 | 24.58 | 24.46 | 23.91 |
| B_top3_style | 22.64 | 23.15 | 19.70 | 20.80 | 19.11 | 20.69 |

- **논문 Fig. 6의 축 대비(재현 판정 기준)가 방향은 재현됨**: viewpoint가 전 날씨 조건에서 style보다 높음(weather 평균 gap +2.9). B_top3_style ≈ A_style(gap 0.3)로 붕괴가 t 선택과 무관한 style-축 구조적 현상임을 시사.
- **조건별 gap 비대칭 발견**: snowy(+5.44)·rainy(+3.83)에서 gap 최대, foggy(+1.05)·night-time(+1.32)에서 최소. 가설(미검증): Cityscapes 원본이 흐린/안개 낀 촬영분을 포함해 foggy는 원본 스타일과 덜 충돌하는 반면 snowy/rainy는 원본의 맑은 도심 스타일과 가장 이질적이라 style 어댑터의 당김이 프롬프트를 가장 세게 압도. 리포트 후보 발견, 정성 확인(생성 이미지 육안) 권장.
- **미해결(§5)**: 0% 컨트롤의 날씨 CLIP-Score 부재로 "style이 하락" vs "viewpoint가 상승"을 절대 기준으로 가르지 못함 — control_base weather 400장 생성 후 재평가 시급.

### 부록: t 스윕 그리드 13개의 설계 근거 (기록 보강, 2026-07-30)

`tsweep_sdxl.yaml`의 [1, 41, 81, 121, 201, 301, 401, 481, 601, 701, 801, 901, 981] 구성 논리:

- **논문 후보 4개(1, 81, 201, 481) 전부 포함** — 논문의 휴리스틱 선택과 같은 후보 위에서 비교(apples-to-apples).
- **저 t 구간은 간격 40으로 조밀** (1, 41, 81, 121, 201): 논문 후보가 몰려 있고 논문 승자(t=81)가 있는 구간이라 그 주변을 정밀화 — 41(1과 81 사이), 121(81과 201 사이) 추가. 실제로 1위 t=41이 이 정밀화에서만 발견될 수 있었음.
- **상위 구간은 간격 ~100~120으로 성김** (301~981): 저 t 우위 가설을 검증 없이 가정하지 않기 위한 전 스케줄 커버리지. 실제로 t=601이 top-5에 들어 커버리지가 정당화됨. 981은 스케줄 끝(t_max=999) 근방 경계점.
- **13개인 이유는 비용**: CA 측정 1회(실측 t당 43.7분) ≪ LoRA 학습 1회라 조밀 그리드가 감당 가능하되, 13 × ~44분 ≈ 9.5h로 하룻밤 측정 예산에 맞춤. 모든 t가 ≡1 (mod 20)인 것은 논문 후보(1, 81, 201, 481)와의 정렬.

### arm A 완료 + 효율화 구현 (2026-07-30)

**arm A 학습 완료**: 10k 스텝, 7h03m (2.54s/step 실효). 체크포인트 4개+final 전부 재로드 검증(20 tensors, all finite), loss_curve.png 정상. drift 최종 요약: final_jaccard 0.053, min 0.000, spearman 0.90~0.94 유지, 최초 10개 프로젝션 전원 ever_left (아래 Phase 5 신호 참조). 생성 단계에서 mfs EIO로 1회 사망(82번째 이미지) → 손상 1장 삭제 후 `--skip-existing` 재개(이미지별 seed라 출력 동일). 평가는 생성 완료 후.

**효율화 Tier 1+2 구현 완료 (Opus 구현, Fable 리뷰·재검증 — 최종 테스트 230개 통과 + CPU 스모크 8단계 통과, 프리페치·저장 견고화 경로 포함 end-to-end 확인)**:
- **프리페치** (`train.prefetch: true` 기본): 단일 워커 스레드가 다음 micro-batch를 순서 보존 읽기. 안전 근거: crop RNG는 전역이 아니라 `derive_generator(seed, "cityscapes-crop", index, _crop_draws)` — 단 `_crop_draws`가 데이터셋의 읽기 횟수 카운터라 **읽기 순서가 수치에 영향** → 워커 1개가 루프와 동일 순서 소비로 재현. on/off 비트 동일성 테스트 포함(loss 5개 `==`, LoRA 텐서 `torch.equal`, 스레드명 검증으로 실효성 증명).
- **텍스트 인코더 오프로드**: 학습 시작 전 전 캡션 pre-encode(`distinct_captions()` — 메타데이터만, 디코드 0회) 후 텍스트 인코더 2개(~3.3GB)를 CPU로. VAE는 per-step 인코딩 위해 GPU 유지. 미등록 캡션 등장 시 복원→인코드→재퇴장 가드. CUDA에서만 동작(CPU 테스트 비트 동일 유지).
- **drift bf16 옵션** (`measure.drift_autocast: "no"|"bf16"`, 기본 "no"=nullcontext로 비트 동일): stability 콜백 **및** step-0 baseline 양쪽에 동일 적용(한쪽만 bf16이면 정밀도 차가 drift로 오인됨 — 구현자 판단 채택). scripts/measure_ca.py(선택 산출)는 불변. **게이트 통과 전까지 "no" 유지.**
- **생성 저장 견고화**: `save_image_verified`(EIO 지수 백오프 ≤3회, re-open+full decode 검증, 부분 파일 삭제) + `_ImageWriter`(백그라운드 1스레드, 유한 큐, 첫 에러 latch→main thread 재발생, manifest는 전 저장 검증 후에만 기록).
- drift 스냅숏 후 `empty_cache()` (tsweep OOM 교훈의 예방 적용).
- **게이트 스크립트** `scripts/check_drift_autocast_gate.py` (리뷰 완료, 테스트 230개 통과): arm 최종 체크포인트를 단일 `merged_lora` 컨텍스트에서 fp32/bf16 각 1회 측정(동일 샘플·seed, drift와 동일 kwargs), spearman ≥0.99 ∧ top-2% jaccard ≥0.8이면 PASS → `drift_autocast_gate.json` 기록. 임계값은 모듈 상수로 고정(사후 조정 금지). 측정 호출부 crop="center" 강제가 3곳→4곳이 됨(CLAUDE.md 갱신). GPU가 비는 대로 arm A에 실행 예정(~20분).

**[사고 기록 2026-07-30 오전, Fable 과실] arm A 생성 3연발 문제**: ① 06:23 최초 생성 launch에 `--lora` 누락 — 베이스 SDXL로 425장 생성함 (스크립트가 WARNING을 냈으나 launch 직후 로그 확인을 빠뜨림). 해당 이미지는 `control_base/generated`로 이동 보존 (per-image seed 결정론이라 유효한 베이스 대조군 부분집합, 로그도 gen_control_base.log로 보존; 나머지 475장을 채울지는 사용자 비용 판단 대기). ② mfs EIO 2차 사망(08:39, 구코드) 후 감시 셸이 자기 자신을 생존으로 오인해 2h09m 방치 — 감시자를 폐기하고 하니스 직접 통지(run_in_background)로 교체. ③ 재시작 시 상대경로 리다이렉트가 바뀐 cwd에서 실패 1회 — launch는 절대 경로로 통일. **재발 방지 체크리스트: 생성 launch에 --lora 필수 + launch 직후 로그의 WARNING/LoRA 적재 라인 확인.** 진짜 arm A 생성은 10:59 시작 (Loaded LoRA: 10 projections 확인). ④ 13:30 그 프로세스가 외부 SIGTERM(원인 미상, EIO 아님 — 하니스 백그라운드 수명 의심)으로 584/900에서 사망, 이미지는 견고화 저장 덕에 584장 전원 무결. setsid 분리 재시도는 로그 무기록 즉사(원인 미상) → run_in_background로 재재개(13:50대). **점검 명령의 pgrep 자기-매칭 함정 확인 → 프로세스 생존 확인은 `ps aux | grep "[g]enerate_images.py"` 패턴으로 통일.**

**Phase 5 조기 신호 (arm A drift, 잠정)**: baseline(64샘플 재측정) vs 학습대상 세트 jaccard 0.818로 시작 → step 500에 0.111 → step 2000 이후 0.000~0.053 (완전~준완전 분리), 반면 전역 순위 spearman은 0.914~0.937 유지. **"학습이 선택된 프로젝션의 CA를 깎아 top-2%에서 밀어내는" 자기 무효화 패턴** — 전역 CA 지형은 안정. 전 스냅숏·타 arm 재현 여부 확인 후 정식 기록 예정.

### mfs 세 번째 장애 유형: I/O 무기한 행 (2026-07-31/08-01 밤, 실험 메인 세션)

run 1(A_t81_style) 생성 launch 2회 연속으로 프로세스가 **D-state(`folio_wait_bit_common`) 무기한 행** — SDXL 가중치(mmap lazy read, ~14GB)를 GPU로 옮기는 구간에서 mfs 특정 청크 읽기가 영원히 안 돌아옴 (소용량 읽기는 동시에 246MB/s로 정상 → 마운트 전체가 아니라 특정 청크/청크서버 문제). EIO(즉시 실패)·조용한 손상에 이은 세 번째 유형으로, 프로세스가 죽지도 않고 로그도 없어 가장 악질. **해결: SDXL fp32 스냅숏을 로컬 컨테이너 디스크로 선별 복사(14GB, fp16/onnx 제외, per-file 재시도 cp)하고 `HF_HOME=/tmp/hf-local`로 실행** — 이후 모델 로드는 mfs를 안 탐. 주의: 로컬 디스크는 재배포에서 사라지므로(runbook의 /workspace-only 원칙) 복사는 세션마다 재실행 가능해야 하며, 로컬 여유는 4.8GB뿐이라 CLIP까지 옮길지는 평가 단계에서 판단. 진단법 기록: `grep State /proc/<pid>/status`가 `D`이고 wchan이 `folio_wait_bit_common`이면 mfs 행 — 재시도보다 로컬 우회가 답.

### 리포트 전체 초안 완성 (2026-08-07, 보고서 세션 — Fable 작성, 사용자 검토 대기)

**[docs/REPORT.md](REPORT.md)** — 영문 최종 리포트의 전체 초안 (Abstract~Conclusion + References 22종 + Appendix A/B). 사용자가 Abstract·Intro·Related Work·References를 대화에서 확정했고, 나머지(§3~§8, Appendix)는 승인된 계획에 따라 Fable이 일괄 작성. 표 수치는 전부 `runs/` 원본 JSON에서 재계산해 검증(스팟체크 16/16, 인용 22/22 본문 등장). 신규 figure 3종 생성: `figures/{tradeoff_scatter,drift_curves,qualitative_weather_grid}.png`. 재계산 중 draft 대비 정정: arm 간 겹침 범위(같은 축 {A,B,C} 74~82%, D 쌍 50~64%), 축 간 겹침에 D 54% 포함(기존 40~47% → 40~54%), head 단위에서도 q=0.1에서 conflict 41↔81 역전 존재(리포트 §5.1에 명시). 정성 그리드는 snowy #800/#801 자동 선별 — 사용자 교체 가능.

### 전 8 run 완료 — 결과 집계·보고서 초안 (2026-08-04, Fable)

8 run(A_t81·B_top3·C_t41·D_t1 × style/viewpoint) + 0% 컨트롤의 학습·생성·평가 전부 종료 (실험 세션, 07-31~08-03, run당 학습 5.6~6.3h·생성 ~3h, 중단 조건 발동 0회). **전체 수치·해석은 [REPORT_DRAFT.md](REPORT_DRAFT.md) (2026-08-04 작성, 사용자 요청 — Sonnet 최종 리포트의 입력)**. 요점만:

- **재현 판정 성공: Fig. 6 시그니처 4/4** — 전 arm 쌍에서 CMMD style<viewpoint ∧ weather CLIP viewpoint>style. 0% 기준으로는 양 축 모두 하락하되 style이 더 붕괴(−8.5~−11.7 vs −6.1~−7.8)가 정확한 기술.
- **t 효과 = 트레이드오프**: CMMD는 A(t81)가 양축 최적(0.638/0.949) — 논문 휴리스틱이 CMMD 기준 실증 지지. 날씨 CLIP은 C(t41) style 최고(23.16)·B(top3) viewpoint 최고(25.62, 자기 분해 성분 A/C/D 전부 상회). D(t1)는 양 지표 지배당함. **해석 가설(미검증): CMMD~CA 크기(t81이 저클러스터 CA 최대), 제어성 보존~ews(t41 최저)** — ews 순위(41>1>81)는 실전 순위로 단조 이어지지 않음(t1 최하위).
- **drift (기여 3) 확정**: 자기 무효화 8/8 재현 — Jac@500 0.07~0.16 → 10k까지 플래토(0.05~0.17), 전역 spearman 0.77~0.95 안정, 추적 집합 mean CA 유지(경계 경쟁이지 CA 소멸 아님). **축 비대칭**: baseline(64샘플 재측정) vs 학습대상 Jaccard가 style 0.66~0.77 vs viewpoint 0.23~0.37 — viewpoint 선택의 추정 분산이 큼(낮은 CA·concentration과 정합).
- **E arm 재실행 발견 → 스코프 정식 재편입 (08-07 확정, §5 해소 항목 참조)**: E_random 완료 — CMMD 0.681로 **CMMD는 CA 선택 가치를 변별 못 함**을 실증, weather 23.76(viewpoint 급)으로 **style 붕괴가 CA 선택 특이적**임을 보이는 음성 대조군. **E_all 전체 완료 (08-07 확인)**: CMMD 0.596(전체 최저) + weather 20.65(style급 붕괴) — "full LoRA는 도메인 최적이나 제어성 상실"이라는 논문 Fig. 1 전제의 독립 재현. drift final Spearman 0.638(min 0.575)로 전면 학습의 CA 지형 재편 확정(E_random은 0.742~0.761로 2% CA arm 하한 부근).
- 남는 최우선 보강: seed 1개/arm 한계, 정성 이미지 검토, memorization NN 체크 (§10 of REPORT_DRAFT).

### 논문 휴리스틱 KLD 재현 + Fig. 10/11 히트맵 분석 (2026-08-05, 사용자 요청, 실험 세션 — CPU only)

head 측정값(`runs/tsweep_sdxl_head/measurements/`)으로 t ∈ {1, 41, 81} 분석. 산출 figure 3종: `docs/figures/ca_kld_style_viewpoint.{pdf,png}` (분포+KLD), `ca_heatmap_qkvout_t.{pdf,png}` (Fig. 11식, t 간 공유 스케일), `ca_heatmap_qkvout_t_panelnorm.{pdf,png}` (논문식 패널별 정규화). 계산 스크립트는 세션 스크래치패드(휘발) — 재현 필요 시 이 항목의 수치 정의로 재작성 가능(축별 CA를 합으로 나눠 확률화, 자연로그 KLD).
> **히트맵 수정 재생성 (2026-08-10, 보고서 세션)**: 원판은 640ch 블록(10 head — down1 8모듈, up1 12모듈) 열을 20행 그리드에 NaN 패딩해 얕은/깊은 열 하반부가 흰색으로 남았음(사용자 발견 — 논문 Fig. 10/11은 완전 채움). head 값 행 반복(×2)로 열을 밴드 높이에 맞춰 재생성 (`ca_heatmap_qkvout_t{,_panelnorm}.{png,pdf}` 갱신, 정규화·레이아웃 불변). 스크립트는 보고서 세션 스크래치패드 `make_ca_heatmap.py` (여전히 휘발 — repo 커밋 여부는 사용자 판단 대기).

- **KLD (양방향, nats)**: t=1: S‖V 0.0084 / V‖S 0.0081 · t=41: 0.0058 / 0.0055 · t=81: 0.0081 / 0.0077. 기준 분포 선택과 무관하게 순위 동일: **t=1 ≳ t=81 > t=41**. 논문 휴리스틱(분포 차이 최대 t 선택) 기준으로는 t=41이 꼴찌 — **방향 지표(ews, t=41 1위)와 크기 분포 지표(KLD)가 정반대 신호**라는 대비가 정량화됨.
- **전역 vs 꼬리 분리**: 축 간 전역 spearman 0.93~0.97로 분포 전체는 거의 동일하나, **top-2%(208 unit) jaccard는 t=81에서 최소(0.249; t=41 0.304, t=1 0.373)** — "t=81이 축 구분 최대"라는 논문 주장은 전체 분포가 아니라 **선택이 쓰는 꼬리에서 성립**. KLD와 top-k 겹침이 t 순위를 다르게 매긴다는 것 자체가 리포트 소재 (논문 휴리스틱의 실효 성분 = 상위 집합 분리).
- **t=1 병리 부재**: 논문은 "t≈1은 clean 이미지라 gradient 신호 부족"이라 했으나 SDXL 실측에서 ‖∇L_diff‖가 t=1과 t=81에서 동일 수준(2.60 vs 2.59e-4), CA 상대 SEM도 t=1이 최소(1.43%; 1,024 페어링 기여). 논문 Fig. 11의 t=1 점묘는 소표본 추정 분산과 정합적(논문 표본 수 미기재).
- **주의(세션 교훈)**: 처음에 논문 백본을 SD1.5로 오인하고 아키텍처 차이로 분석했다가 사용자 지적으로 정정 — 논문도 SDXL 명시("We use SDXL as the pretrained T2I model"). kind 대비 미재현 문제는 §5 미해결 항목 참조.

## 6b. 로컬 환경 주의사항

- torch가 `2.9.1+cpu`인데 torchvision이 `+cu126`이면 import 연쇄 실패. `pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cpu "torchvision==0.24.1+cpu"` (버전만 지정하면 pip가 no-op 처리함).
- pytest 기본 temp가 권한 잠김 → `pyproject.toml`의 `--basetemp=.pytest_tmp`로 우회 (이미 설정됨).
