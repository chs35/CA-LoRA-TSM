# 참고 자료 보관함

리포트(`docs/report_draft*.md`) 작성에 참고한 자료와 논문의 목록. 새 자료를 참고할 때마다 여기에 추가한다.

- **검증** 열: `웹` = 작성 세션에서 웹 검색으로 제목·arXiv 번호(·학회)를 확인함. `선행` = Improving CA-LoRA 보고서(`REPORT.md`)의 참고문헌에서 그대로 가져옴(그 보고서에서 검증된 서지). `기억` = 모델의 기억에 의존해 적었고 **제출 전 대조 필요**.
- **번호** 열: `report_draft1.md`의 References 번호.
- 저자 전체 목록이 확실하지 않은 항목은 "et al."로 적었다.

최종 갱신: 2026-09-20.

## 1. 내부 자료

| 자료 | 위치 | 어디에 썼는가 |
|---|---|---|
| Improving CA-LoRA 기술 보고서 (August 2026) | `/workspace/Optimizing-CA-LoRA/docs/REPORT.md`, https://github.com/chs35/Improving-CA-LoRA | 리포트의 구조 견본. 2.1·2.2절의 서술과 서지, 3.1절의 측정 단위·추정량·불변 조건, 선행 결과(t=41 vs 81, 선택 drift) 요약. References [26] |
| 연구 노트 v2 (2026-09-11) | `docs/draft.md` | 문제의식, 핵심 가설, 방법 설계(구간, core/context, C1/C2), 구성표, 예상 시나리오 |
| 정량 결과 (자동 생성) | `docs/QUANTITATIVE.md` ← `scripts/make_quantitative.py` | 1절 결과 요약의 모든 수치, 3.1절 실효 파라미터 비율(2.25~2.45%), expert 공유(13 중 8) |
| 결과 해석 초안 (2026-09-19) | `docs/RESULTS.md` | 결과 요약의 해석 방향, 프로토콜 이탈 목록(4절 예정) |
| 정성 평가 | `docs/QUALITATIVE_2026-09-19.md` | (아직 본문에 미사용. 5절 예정) |
| 중간 보고서 | `docs/INTERIM_REPORT_2026-09-15.md`, `docs/INTERIM_REPORT_2026-09-17_stage4.md` | (직접 인용 없음. RESULTS.md의 근거) |
| 대표 t 결정 기록 (2026-09-15) | `runs/sdxl_cityscapes/diagnosis/DECISION.md` | 3.2절의 대표 t와 그 근거(인접 t는 재측정 수준으로 유사, 차이는 거리에 따라 누적, [500,999]는 단일 t로 대표 불가) |
| 구간 정의 코드 | `src/catsm/intervals.py` | 3.2절 절반 자르기 분할, log-SNR 기록 방식 |
| 라우터 코드 | `src/catsm/models/router.py`, `src/catsm/models/multi_sliced_lora.py` | 3.3절 게이트 식, 활성화 없음, 항등 초기화, 행렬별 게이트, CFG에서의 게이트 반복 |
| Fostering·Assembling 학습 코드 | `src/catsm/train/foster.py`, `src/catsm/train/assemble.py` | 3.2절 구간 내 균등 t 추출·구간 이탈 검사, 3.3절 라우터 학습(전 구간 t, expert 동결) |
| 선택 코드 | `src/catsm/select.py` | 3.2절 representative / aggregate 두 방식 |
| 원자료 JSON | `runs/sdxl_cityscapes/*/eval.json`(routing.experts의 logsnr_lo/hi), `diagnosis/diagnosis.json`, `experts/*/paired_loss.json`, `bundles/*/bundle_stats.json` | Table 2의 log-SNR 경계, 진단·paired loss·라우터 수치 |
| 그림 | `docs/figures/cmmd_vs_weather_clip.png` ← `scripts/plot_tradeoff.py` | (5절 예정) 정렬 vs 제어성 산점도 |

## 2. 논문

### 2.1 본 연구의 직접적 기반

| 번호 | 서지 | 링크 | 검증 | 어디에 썼는가 |
|---|---|---|---|---|
| [19] | M. Park, S. Park, J. Lee, H. Park, K. Hwang, F. Porikli, J. Choo, S. Choi. "Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation." CVPR 2025 | arXiv:2503.22172 | 선행 | 전체. CA 정의, sliced LoRA, t=81 휴리스틱 |
| [25] | S. Zhuang et al. "TimeStep Master: Asymmetrical Mixture of Timestep LoRA Experts for Versatile and Efficient Diffusion Models in Vision." ICML 2025 (PMLR v267) | https://arxiv.org/abs/2503.07416 , https://proceedings.mlr.press/v267/zhuang25a.html | 웹 | 2.3, 3.2, 3.3. Fostering/Assembling, core(게이트 없음)·context(timestep 의존 게이트), 적용 범위(UNet/DiT/MM-DiT, 도메인 적응·post-pretraining·증류) |

TSM에 대해 **본문에 쓰지 않은 것**(확인하지 못해서): 구간 개수의 구체적 값, 게이트의 정확한 함수형, expert당 학습 step. `draft.md`는 "구간당 4K(TSM 기준)", "router = FC(z_t) + E(t)"라고 적고 있으나 논문 원문과 대조하지 않았다. 본문에서 인용하려면 원문 확인 필요.

### 2.2 Related Work 2.1 (효율적 파인튜닝)

| 번호 | 서지 | 링크 | 검증 |
|---|---|---|---|
| [2] | S. Basu et al. "On Mechanistic Knowledge Localization in Text-to-Image Generative Models." ICML 2024 | arXiv:2405.01008 | 선행 |
| [7] | R. Gal et al. "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion." ICLR 2023 | arXiv:2208.01618 | 선행 |
| [12] | E. J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022 | arXiv:2106.09685 | 선행 |
| [16] | Y. Lee et al. "Surgical Fine-Tuning Improves Adaptation to Distribution Shifts." ICLR 2023 | arXiv:2210.11466 | 선행 |
| [21] | D. Podell et al. "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." ICLR 2024 | arXiv:2307.01952 | 선행 |
| [22] | R. Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022 | arXiv:2112.10752 | 선행 |
| [23] | N. Ruiz et al. "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR 2023 | arXiv:2208.12242 | 선행 |
| [24] | Q. Zhang et al. "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning." ICLR 2023 | arXiv:2303.10512 | 선행 |

### 2.3 Related Work 2.3~2.4 (noise level의 역할, flow 기반 모델)

| 번호 | 서지 | 링크 | 검증 | 본문에서 인용한 주장 |
|---|---|---|---|---|
| [1] | Y. Balaji et al. "eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers." 2022 | arXiv:2211.01324 | 기억 | 생성 단계별 전문 denoiser 앙상블. 고노이즈 단계는 텍스트 의존이 크고 저노이즈 단계는 텍스트를 거의 무시 |
| [6] | Z. Feng et al. "ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts." CVPR 2023 | arXiv:2210.15257 | 기억 | 단계별 denoising expert 혼합 |
| [3] | J. Choi, J. Lee, C. Shin, S. Kim, H. Kim, S. Yoon. "Perception Prioritized Training of Diffusion Models." CVPR 2022 | arXiv:2204.00227 | 기억 | noise level을 coarse / content / clean-up 단계로 구분, 손실 가중 재설계 |
| [27] | Y. Zhang et al. "ProSpect: Prompt Spectrum for Attribute-Aware Personalization of Diffusion Models." ACM TOG (SIGGRAPH Asia) 2023 | https://arxiv.org/abs/2305.16225 | 웹 | 생성 단계별 프롬프트 임베딩으로 레이아웃·내용·재질/스타일 분리. (검색으로 확인한 것은 "저주파→고주파 순으로 생성, 단계별 프롬프트"까지) |
| [8] | H. Go, J. Kim, Y. Lee, S. Lee, S. Oh, H. Moon, S. Choi. "Addressing Negative Transfer in Diffusion Models." NeurIPS 2023 | https://arxiv.org/abs/2306.00354 | 웹 | noise level 차이가 클수록 과제 친화도 감소, negative transfer 발생, interval clustering |
| [20] | B. Park, S. Woo, H. Go, J. Kim, C. Kim. "Denoising Task Routing for Diffusion Models." ICLR 2024 | https://arxiv.org/abs/2310.07138 | 웹 (저자 목록은 기억) | noise level별로 서로 다른 채널 부분집합 활성화 |
| [9] | T. Hang et al. "Efficient Diffusion Training via Min-SNR Weighting Strategy." ICCV 2023 | arXiv:2303.09556 | 기억 | 과제 간 충돌을 손실 가중으로 완화 |
| [11] | J. Ho, A. Jain, P. Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS 2020 | arXiv:2006.11239 | 선행 | 표준 디퓨전 손실, 이산 timestep 매개화 |
| [14] | T. Karras, M. Aittala, T. Aila, S. Laine. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022 | arXiv:2206.00364 | 기억 | σ를 직접 조건 변수로 사용 |
| [15] | D. P. Kingma, R. Gao. "Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation." NeurIPS 2023 | arXiv:2303.00848 | 기억 | 디퓨전 목적함수들이 log-SNR 축 위의 가중 적분으로 통일 |
| [17] | Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. "Flow Matching for Generative Modeling." ICLR 2023 | arXiv:2210.02747 | 기억 | flow matching |
| [18] | X. Liu, C. Gong, Q. Liu. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR 2023 | arXiv:2209.03003 | 기억 | rectified flow, 직선 경로 x_t = (1−t)x_0 + tε |
| [5] | P. Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." ICML 2024 | arXiv:2403.03206 | 기억 | logit-normal timestep 추출(중간 noise level에 질량)이 최선 |

rectified flow의 log-SNR 환산식 λ = 2 log((1−t)/t), t = 1/(1+e^{λ/2})는 x_t = (1−t)x_0 + tε에서 α = 1−t, σ = t를 λ = log(α²/σ²)에 대입해 직접 유도한 것이며 특정 논문의 인용이 아니다.

### 2.4 데이터·평가

| 번호 | 서지 | 링크 | 검증 |
|---|---|---|---|
| [4] | M. Cordts et al. "The Cityscapes Dataset for Semantic Urban Scene Understanding." CVPR 2016 | arXiv:1604.01685 | 선행 |
| [10] | J. Hessel et al. "CLIPScore: A Reference-free Evaluation Metric for Image Captioning." EMNLP 2021 | arXiv:2104.08718 | 선행 |
| [13] | S. Jayasumana et al. "Rethinking FID: Towards a Better Evaluation Metric for Image Generation." CVPR 2024 | arXiv:2401.09603 | 선행 |

## 3. 검색했으나 본문에 쓰지 않은 것

검색 결과에 함께 나온 관련 논문. 필요하면 2.4절에 추가할 수 있다(모두 미확인, 제목과 번호만).

- "Denoising Task Difficulty-based Curriculum for Training Diffusion Models." arXiv:2403.10348
- "Switch Diffusion Transformer: Synergizing Denoising Tasks with Sparse Mixture-of-Experts." arXiv:2403.09176
- "Diffusion Model Patching via Mixture-of-Prompts." arXiv:2405.17825

## 4. 제출 전 할 일

- [ ] `기억` 표시 항목 9개(eDiff-I, ERNIE-ViLG 2.0, P2, Min-SNR, EDM, Kingma & Gao, Flow Matching, Rectified Flow, SD3)의 서지와, 본문에서 인용한 주장이 원문에 실제로 있는지 대조.
- [ ] "et al." 항목의 저자 전체 목록 채우기(REPORT.md 형식과 통일하려면).
- [ ] TSM 원문에서 구간 개수·게이트 함수형·학습 step 확인(위 2.1의 메모).
