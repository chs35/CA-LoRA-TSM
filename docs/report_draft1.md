# Concept-Aware LoRA with Timestep Experts: 노이즈 구간별 선택 집합은 무엇을 바꾸는가

<!-- 제목은 가제. -->

Heeseung Chung, September 2026.

## Abstract

Concept-Aware LoRA(CA-LoRA) [19]는 T2I 디퓨전 모델에서 아주 적은 어텐션 헤드만을
파인튜닝하는 방법을 제안하여 프롬프트 제어성을 보존하면서도 모델을 타깃 도메인에
적응하도록 하였다. 다만 이 소수의 어텐션 헤드를 선별하는 과정은 단일 timestep *t*에
의존하면서도, 논문에서 이 *t*의 결정이 다소 휴리스틱하였다는 문제가 있다. 우리는
Improving CA-LoRA [26]에서 실험 코드를 재현하면서 이 휴리스틱한 *t*의 결정을 정량화해
보고자 하였다. 그러나 논문이 휴리스틱하게 제안한 *t*를 정량적 점수로 이긴 *t*는 실제
결과에서 우위를 보이지 못했을 뿐만 아니라, 전반적으로 그에 못 미치는 성능을 보였다.
이에 따라 방향을 바꿔, "*t*를 잘 고르는 방법"이 아닌 "*t* 결정이 불완전하여도 괜찮은
방법"을 탐구하기로 하였다. 나아가 CA-LoRA를 다른 도메인이나 백본으로 확장할 때 매번
*t*를 재탐색해야 하는 부담을 없애는 것이 목적이다.

<!-- TODO(결과 확정 후): 방법 한 문장(구간별 선택 + 구간별 expert + 라우터)과 핵심 결과 두세 문장을
     추가. 후보 문장은 1절 네 번째 문단 참고. -->

## 1. Introduction

Text-to-image(T2I) 생성 모델로 dense prediction용 학습 데이터를 합성하려면 두 가지가
동시에 필요하다. 생성 이미지가 타깃 도메인과 닮아야 하고(정렬), 타깃 데이터에 없는
조건(악천후 등)을 프롬프트로 만들어 낼 수 있어야 한다(제어성). 모델을 타깃 도메인에
파인튜닝하면 전자는 얻지만 후자를 잃는다. Concept-Aware LoRA(CA-LoRA) [19]는 이 긴장을
"어디를 학습할 것인가"로 푼다. 각 어텐션 헤드가 특정 concept 축(style, viewpoint 등)의
프롬프트 섭동에 얼마나 민감한지를 concept-awareness(CA)라는 그래디언트 노름 비로 점수화하고,
상위 2% 헤드에만 LoRA [12]를 붙여 학습한다.

문제는 CA가 측정 timestep *t*의 함수라는 점이다. *t*는 학습이 아니라 **선택**에 쓰인다.
즉 어떤 헤드가 학습되는지가 *t* 하나로 정해지고, 그렇게 정해진 헤드 집합은 이후 모든
timestep에서 학습된다. 원 논문은 네 개 후보 {1, 81, 201, 481} 가운데 t = 81을 CA 분포의
시각적 비교로 골랐으며 정량 기준은 제시하지 않았다. 우리의 선행 작업인 Improving CA-LoRA
[26]는 CA-LoRA를 처음부터 재구현한 뒤, 그래디언트 방향을 보존하는 정량 점수로 *t*를
순위화하였다. 결과는 부정적이었다. 점수 1위인 t = 41은 도메인 정렬(CMMD)에서 논문의
t = 81을 넘지 못했고, 점수 2위인 t = 1은 두 지표 모두에서 지배당했다. 점수는 학습 결과를
단조롭게 예측하지 못했고, 남은 것은 "최적의 *t*는 지표에 따라 다르다"는 트레이드오프였다.
같은 작업에서 서로 다른 noise level의 *t*가 상당히 다른 헤드 집합을 고른다는 점도 확인되었다.

이 두 관찰은 질문을 바꾸게 한다. 단일 *t*를 더 잘 고르려는 시도가 실패했고 *t*마다 선택이
다르다면, 단일 *t*에 의존하는 설계 자체를 없애는 편이 낫다. 본 보고서는 **"*t*를 잘 고르는
방법"이 아니라 "*t* 결정이 불완전해도 괜찮은 방법"**을 탐구한다. 실용적 동기도 있다.
CA-LoRA를 다른 도메인이나 다른 백본으로 옮길 때마다 *t*를 다시 탐색해야 한다면, 특히
timestep의 정의 자체가 다른 flow matching 계열 백본 [5, 17, 18]에서는 "t = 81"이라는
결정이 그대로 이전되지도 않는다. 우리의 핵심 가설은 다음과 같다.

> 어떤 파라미터가 concept에 민감한지는 noise level에 따라 달라진다. 따라서 noise level
> 구간마다 LoRA의 **가중치**뿐 아니라 **선택 집합**도 달라야 한다.

이를 위해 TimeStep Master(TSM) [25]의 두 단계 구조를 가져온다. timestep 축을 구간으로
나누고, 구간마다 CA를 따로 측정해 헤드 집합을 따로 고르며, 그 구간의 timestep에서만 학습한
sliced LoRA expert를 만든다(Fostering). 그다음 서로 다른 스케일의 구간 분할에서 나온
expert들을 core-context 라우터로 결합한다(Assembling). TSM과 다른 점은 셋이다. 구간별로
선택 집합이 다르고, 구간이 비균등하며(CA-LoRA가 중요하게 본 저노이즈 영역을 더 촘촘히),
"이득이 구간별 *선택*에서 오는가 구간별 *학습*에서 오는가"를 분리하는 통제 실험을 설계의
일부로 둔다.

<!-- TODO(결과 절 확정 후 다듬기). 현재 정량 결과(docs/QUANTITATIVE.md) 기준 요약: -->
결과는 가설을 부분적으로만 지지한다. 선택 집합의 noise level 의존은 실재한다(t = 81의
상위 2% 집합과의 Jaccard 유사도는 t = 250에서 0.33, t ≥ 601에서 0.05 이하). 그리고
구간별 선택은 결과를 크게 바꾼다. 구간·학습량·시드가 같고 선택 집합만 다른 두 구성을
비교하면, 구간별 선택은 날씨 프롬프트 제어성(weather CLIP-Score)을 +5.1점 높이는 대신
CMMD를 57% 악화시킨다. 즉 효과는 "더 나은 성능"이 아니라 정렬-제어성 트레이드오프 위에서의
**위치 이동**이며, 그 위치는 거의 전적으로 가장 높은 noise 구간을 맡는 expert 하나가
정한다. 반면 라우터로 여러 expert 집합을 섞는 Assembling은 어느 축에서도 단일 LoRA나
하드 라우팅을 넘지 못했다.

본 보고서의 기여는 다음과 같다.

- noise level 구간별로 CA 선택 집합과 LoRA 가중치를 모두 분리하는 방법(구간별 CA 선택,
  비균등 구간 expert, core-context 라우팅)과 그 전체 구현.
- 20개 timestep에 걸친 CA 선택 집합의 의존성 진단: 쌍별 Jaccard 구조와, 측정 잡음의
  기준선으로서의 split-half Jaccard.
- 구간별 *선택*의 효과와 구간별 *학습*의 효과를 분리하는 통제 설계(C1), 그리고 고노이즈
  구간 expert의 기여를 분리하는 제거 실험(C2).
- 부정적 결과를 포함한 정직한 보고: Assembling의 무효과, 그리고 "단일 최적 구성은 없고
  고노이즈 구간의 선택이 손잡이다"라는 실용 규칙.

## 2. Related Work

### 2.1 Efficient Fine-Tuning of T2I Models

대규모 T2I 모델 [21, 22]은 새 concept이나 도메인에 대해 재학습하기보다 적응시키는 것이
일반적이다. 초기 개인화 방법은 양 극단에 있다. Textual Inversion [7]은 모델을 동결하고
텍스트 임베딩만 최적화하며, DreamBooth [23]는 대상에 맞춰 네트워크 전체를 파인튜닝한다.
LoRA [12]로 대표되는 parameter-efficient fine-tuning은 저랭크 가중치 갱신만 학습해 비용을
낮추고, AdaLoRA [24] 같은 변형은 학습 중 파라미터 예산을 가중치별로 재분배한다. 이들은
*얼마나* 학습할지는 줄이지만 *어디를* 학습할지는 묻지 않는다. "어디"의 문제는 두 갈래에서
다뤄졌다. Lee et al. [16]은 분포 이동 하에서 잘 고른 일부 층만 파인튜닝하는 것이 전체
파인튜닝보다 나을 수 있음을 보였고, Basu et al. [2]는 T2I 모델에서 개별 시각 속성을
담당하는 층을 제거 실험으로 국소화하였다. CA-LoRA [19]는 이 둘을 결합해, 수동 국소화를
헤드 단위 그래디언트 점수로 대체하고 LoRA 갱신을 점수가 높은 헤드로 제한한다.

이 계열의 방법들은 공통적으로 학습 대상 파라미터를 **한 번** 정하고 모든 noise level에서
같은 파라미터를 학습한다. 본 연구가 문제 삼는 것이 이 지점이다.

### 2.2 Concept-Aware LoRA

CA-LoRA [19]의 목표는 도메인 정렬된 세그멘테이션 데이터셋 생성이다. T2I 모델을 타깃
데이터셋(Cityscapes [4])에 파인튜닝해 생성 이미지가 타깃 도메인과 닮게 하되, 악천후처럼
데이터에 없는 조건을 합성할 수 있도록 프롬프트 제어성을 보존한다. 각 어텐션 헤드는
concept-awareness

CA = E[ ‖∇_θ L_Concept‖ / ‖∇_θ L_Diff‖ ]

로 점수화된다. L_Concept = ‖ε_θ(x_t, c, t) − sg[ε_θ(x_t, c_Aug, t)]‖²는 프롬프트 c를
concept 축 방향으로 섭동한 c_Aug(style, viewpoint 등의 바꿔 쓰기)에 대해 그 헤드의
파라미터가 얼마나 강하게 반응하는지를 재고, L_Diff는 표준 디퓨전 손실 [11]이다. CA 상위
2% 헤드에 rank-64 LoRA가 붙는다. 어댑터는 융합된 projection 행렬의 해당 헤드 슬라이스
(Q/K/V는 출력 행, OUT은 입력 열)에만 작용하며(sliced LoRA), 이것만 학습된다.

두 그래디언트는 단일 timestep *t*에서 평가된다. 논문은 {1, 81, 201, 481}에서 style과
viewpoint의 CA 분포를 비교해 t = 81을 휴리스틱하게 골랐다. 논문의 분석은 두 축의 대비를
보인다. style로 고른 어댑터는 도메인 정렬(CMMD [13])에서 앞서지만 날씨 프롬프트 제어성을
잃고, viewpoint로 고른 어댑터는 제어성(CLIP-Score [10])을 보존한다.

Improving CA-LoRA [26]는 공식 코드가 없는 이 방법을 SDXL [21] 위에 재구현하고 세 가지를
보고하였다. (i) 13개 timestep 격자에서 교차 축 방향 유사도로 *t*를 순위화하면 논문 후보
밖의 t = 41이 1위가 되지만, CMMD에서는 t = 81이 여전히 최고이고 t = 41은 제어성에서만
앞선다. 점수 2위 t = 1은 지배당한다. (ii) 서로 다른 noise level의 *t*는 상당히 다른 헤드
집합을 고른다. (iii) 선택된 헤드는 학습 약 500 step 안에 CA 상위 2% 밖으로 밀려난다
(선택의 자기 무효화). 본 연구는 (i)과 (ii)를 출발점으로 삼는다. 정량 점수는 *t* 결정
메커니즘으로 쓰지 않으며, (iii)은 본 연구의 설계 근거로 사용하지 않고 후속 과제로만 남긴다.

### 2.3 Timestep-Master

디퓨전 모델의 LoRA 파인튜닝은 보통 모든 timestep에 같은 LoRA를 쓴다. TimeStep Master
(TSM) [25]는 이것이 서로 다른 noise level의 서로 다른 denoising 과제를 하나의 저랭크
갱신에 밀어 넣는 것이라 보고 두 단계의 파인튜닝을 제안한다. **Fostering stage**에서는
timestep 축을 구간으로 나누고 구간마다 별도의 LoRA를 그 구간의 timestep에서만 학습해
TimeStep LoRA expert를 만든다. **Assembling stage**에서는 서로 다른 스케일의 구간 분할
(거친 분할과 세밀한 분할)에서 나온 expert들을 비대칭적으로 결합한다. 각 timestep에서 가장
좁은 구간의 expert가 게이트 없이 **core**로 들어가고, 더 넓은 구간의 expert들은 timestep에
의존하는 게이트를 거쳐 **context**로 더해진다. expert는 동결하고 라우터만 학습한다. TSM은
도메인 적응, post-pretraining, 모델 증류에서, 그리고 UNet, DiT, MM-DiT 구조에 걸쳐 이득을
보고하였다.

noise level별로 모델 용량을 분리한다는 생각 자체는 TSM 이전에도 있었다. eDiff-I [1]와
ERNIE-ViLG 2.0 [6]은 생성 단계별로 전문화된 denoiser 앙상블을 사전학습 수준에서 사용한다.
TSM은 이를 LoRA 수준으로 내려 파인튜닝 비용으로 가능하게 하였다.

이들 방법에서 구간별로 달라지는 것은 **가중치**뿐이다. 모든 expert는 같은 위치(같은 층,
같은 행렬)에 붙는다. 본 연구는 CA-LoRA의 선택 기제와 결합해 expert가 붙는 **위치**까지
구간별로 달라지게 하며, 이 차이가 실제로 무엇을 바꾸는지를 TSM식 구간별 학습만 적용한
통제 구성과 비교해 분리한다.

### 2.4 The Role of Noise Levels in Diffusion and Flow-Based Generative Models

본 절에서는 "timestep" 대신 **noise level**이라는 말을 쓴다. 이산 timestep 인덱스는
DDPM 계열 [11]의 매개화에 묶인 개념이고, 본 연구가 겨냥하는 성질은 매개화와 무관하게
성립해야 하기 때문이다. 디퓨전 모델에서 x_t = α_t x_0 + σ_t ε이라 하면 noise level은
신호 대 잡음비의 로그 λ = log(α_t² / σ_t²)로 요약된다. EDM [14]은 σ를 직접 조건 변수로
쓰고, Kingma & Gao [15]는 여러 디퓨전 목적함수가 log-SNR 축 위의 가중 적분으로 통일됨을
보였다. flow matching [17]과 rectified flow [18]는 x_t = (1 − t) x_0 + t ε (t ∈ [0, 1])의
직선 경로에서 속도장을 회귀하는데, 같은 정의로 λ = 2 log((1 − t) / t)를 얻는다. 즉 noise를
점진적으로 제거하는 생성 모델이라면 학습 방식이 ε-예측이든 속도 예측이든 noise level이라는
공통 축을 가지며, 이 축 위의 위치는 서로 환산된다.

noise level마다 모델이 하는 일이 다르다는 것은 여러 방향에서 관찰되었다.

- **생성 내용의 단계성.** 고노이즈 단계에서는 전역 구조와 레이아웃이, 저노이즈 단계에서는
  질감과 세부가 결정된다. P2 weighting [3]은 이를 coarse, content, clean-up 단계로 나누고
  손실 가중을 재설계하였고, ProSpect [27]는 생성 단계별로 다른 프롬프트 임베딩을 주어
  레이아웃, 내용, 재질·스타일을 단계별로 분리 제어할 수 있음을 보였다. eDiff-I [1]는
  고노이즈 단계가 텍스트 조건에 크게 의존하고 저노이즈 단계는 텍스트를 거의 무시한다고
  보고하였는데, 이는 프롬프트 섭동에 대한 민감도로 정의되는 CA가 noise level에 따라
  달라지리라는 직접적인 근거이다.
- **denoising 과제 간의 간섭.** Go et al. [8]는 디퓨전 학습을 noise level별 과제의
  multi-task learning으로 보고, noise level의 차이가 클수록 과제 간 친화도(그래디언트
  정렬)가 떨어지며 negative transfer가 실제로 발생함을 보였다. 이들은 인접한 noise level을
  구간으로 묶어 다루는데, 이는 TSM과 본 연구의 구간 분할과 같은 발상이다. Denoising Task
  Routing [20]은 같은 관점에서 noise level별로 서로 다른 채널 부분집합을 활성화한다.
  Min-SNR [9]은 과제 간 충돌을 손실 가중으로 완화한다.
- **noise level의 비균등한 중요도.** Stable Diffusion 3 [5]는 rectified flow 학습에서
  timestep을 균등 추출하지 않고 중간 noise level에 질량을 모으는 logit-normal 분포로
  추출할 때 가장 좋은 결과를 얻었다. 모든 noise level이 동등하지 않다는 점은 flow 기반
  모델에서도 그대로 성립한다.

본 연구의 위치는 다음과 같다. 위 연구들은 noise level에 따라 모델의 **기능**이 달라짐을
보였고, 일부는 그에 맞춰 가중치나 경로를 분리하였다. 우리는 같은 축 위에서 **"어떤
파라미터가 주어진 concept에 민감한가"라는 선택 신호**가 어떻게 달라지는지를 측정하고(3.1),
그 차이를 파인튜닝 대상의 선택에 반영한다(3.2). 방법의 모든 구성 요소는 "noise level의
구간"만을 요구하므로, 구간을 log-SNR로 기록해 두면 flow matching 계열 백본에도 같은 정의를
적용할 수 있다. 다만 본 보고서의 실험은 ε-예측 디퓨전 모델인 SDXL 하나에 한정되며, flow
기반 백본에서의 검증은 후속 과제이다.

## 3. Methods

전체 파이프라인은 네 단계이다. (1) 여러 noise level에서 헤드 단위 CA를 측정하고 선택
집합의 의존성을 진단한다(3.1). (2) noise level 축을 비균등 구간으로 나누고 구간마다 선택
집합을 정해 그 구간에서만 학습한 sliced LoRA expert를 만든다(3.2). (3) 여러 분할에서 나온
expert들을 하드 라우팅 또는 학습된 core-context 라우터로 결합한다(3.3). (4) 효과의 출처를
분리하는 통제 구성과 보조 측정 도구를 둔다(3.4). 이하 SDXL의 관례에 따라 t ∈ {0, …, 999}로
쓰며 t가 작을수록 noise가 적다.

### 3.1 CA Measurement at Head Granularity

**선택 단위.** 측정과 선택의 단위는 Improving CA-LoRA [26]와 같다. 단위는 융합된
projection 행렬 안의 이론적 어텐션 헤드이다. SDXL UNet에는 560개의 융합 projection
(어텐션 모듈 70개 × 블록당 어텐션 2개 × {Q, K, V, OUT})이 있고, 헤드 폭 d_h = 64에서
640채널 블록은 projection당 10개, 1280채널 블록은 20개의 헤드를 가져 총 **U = 10,400개의
헤드 단위**가 된다(단위별 파라미터 수의 합이 어텐션 projection 파라미터 955,187,200개와
정확히 일치함을 확인하였다). 단위는 Q/K/V에서는 행 슬라이스, OUT에서는 열 슬라이스이며,
상위 2% 예산은 **단위 개수**로 센다. 선택 하나는 ⌊0.02 × 10,400⌋ = 208개 단위이다. 단위
폭이 블록마다 다르므로 실효 파라미터 비율은 선택마다 조금씩 다르다(본 연구의 선택에서
어텐션 파라미터의 2.25~2.45%).

**추정량.** 헤드별 CA는 표본별 노름 비의 기댓값이다(기댓값의 비가 아니다). 표본마다 헤드
슬라이스에서 ‖g_c[h]‖ / ‖g_d[h]‖를 계산해 평균한다. 측정 1회는 Cityscapes 학습 이미지
256장(center crop, 1024²)에 대해 축마다 표본당 4개의 concept 섭동을 쓴다(논문 Appendix
B.2의 축별 템플릿 5개에서 비복원 추출). 따라서 timestep 하나, 축 하나당 1,024개의 concept
그래디언트가 기여한다. 추정량을 지키기 위한 세 불변 조건도 그대로 유지한다.

- **배치 크기 1.** 배치가 1보다 크면 추정 대상이 E[‖Σg_c‖ / ‖Σg_d‖]라는 다른 양으로
  바뀐다. 측정 경로는 b = 1을 강제한다.
- **축 간 짝짓기.** 한 표본에서 두 축은 같은 noised latent x_t와 같은 기준 예측을 공유한다.
- **timestep 간 짝짓기.** 표본별 noise ε은 *t*와 독립적으로 고정된다. 모든 *t*가 같은
  이미지와 같은 ε을 쓰고 *t*에 따른 스케일링만 다르므로, *t* 사이의 선택 집합 차이는 표본
  차이가 아니라 *t*의 차이만을 반영한다.

모든 측정은 fp32로 수행한다. 본 연구의 선택은 모두 **style 축**의 CA를 사용한다.
<!-- TODO(확인): style 축만 쓴 이유를 한 문장으로. 예: "선행 재현에서 제어성 손실이 가장 컸던
     축이어서 개선 여지를 보기에 적합하다" 또는 "계산 예산". viewpoint는 측정만 하고 선택에는 쓰지 않음. -->

**측정 격자.** *t*는 20개 지점에서 측정한다: {1, 41, 61, 81, 121, 125, 201, 250, 301,
375, 401, 481, 500, 601, 701, 750, 801, 901, 981, 999}. 13개는 선행 작업의 격자를 같은
프로토콜로 재사용한 것이고, 7개(61, 125, 250, 375, 500, 750, 999)는 후보 구간 경계와 구간
중앙을 덮기 위해 추가하였다.

**선택 집합의 의존성 진단.** 학습에 앞서 가설의 전제를 검증한다. *t*마다 style 축의 상위
208개 집합 S_t를 구하고 두 가지를 계산한다.

- **쌍별 Jaccard** J(S_a, S_b) = |S_a ∩ S_b| / |S_a ∪ S_b|. 20 × 20 행렬은 선택 집합이
  noise level에 따라 얼마나, 그리고 어디에서 바뀌는지를 보여 준다. 모든 *t*에서 집합이 거의
  같다면 구간별 선택은 통제 구성과 구별되지 않으므로 방향을 재고해야 한다.
- **split-half Jaccard.** 같은 *t*에서 256개 표본을 표본 단위로 짝수·홀수 절반으로 나누어
  각각 상위 208개를 뽑고 두 집합의 Jaccard를 구한다. 이는 "같은 *t*를 다시 측정했을 때의
  일치도", 즉 측정 잡음만으로 생기는 불일치의 기준선이다. 서로 다른 *t* 사이의 J가 이
  기준선에 가까우면 그 차이는 잡음과 구별되지 않고, 기준선보다 뚜렷이 낮아야 실제 차이로
  읽는다. 또한 split-half 값 자체가 낮은 *t*에서는 상위 2% 집합의 상당 부분이 표본 추출
  잡음이므로, 그 *t*에서의 CA 선택이 사실상 무작위 선택에 가까워질 위험을 알려 준다.

이 진단은 점수를 만들기 위한 것이 아니다. 선행 작업에서 정량 점수로 *t*를 고르는 시도가
실패했으므로, 본 연구는 진단 결과를 (i) 전제의 검증, (ii) 구간 경계와 대표 *t*의 타당성
확인, (iii) 고노이즈 구간 선택의 신뢰도 평가에만 쓴다.

### 3.2 Non-Uniformly Sliced Timestep Expert

**TSM의 Fostering stage.** TSM [25]은 timestep 축 [0, T)를 n개의 구간으로 나누고, 구간
i마다 별도의 LoRA Δ_i를 두어 그 구간에서 추출한 *t*로만 표준 디퓨전 손실을 학습한다.
추론 시에는 현재 *t*가 속한 구간의 LoRA만 활성화된다. 하나의 저랭크 갱신이 모든 noise
level의 denoising 과제를 동시에 감당하지 않아도 되므로, 2.4절의 과제 간 간섭을 구조적으로
피한다. TSM에서 구간은 균등하고, 모든 expert는 같은 위치에 붙는다.

본 연구의 expert는 이 구조를 따르되 두 가지가 다르다.

**(a) 구간별 선택 집합.** 구간 I = [t_lo, t_hi]마다 CA 측정에서 상위 208개 헤드 집합 S_I를
따로 정하고, expert Δ_I를 S_I의 헤드 슬라이스에만 붙는 sliced LoRA로 만든다. 따라서
expert마다 가중치뿐 아니라 **붙는 위치**가 다르다. 구간의 선택 집합을 정하는 방식은 두
가지를 구현하였다.

- *representative*: 구간 안의 대표 timestep 하나 t_I ∈ I의 CA로 선택한다.
- *aggregate*: 구간 안에서 측정된 모든 *t*의 CA를 *t*별로 정규화한 뒤 평균해 선택한다
  (선행 작업의 multi-timestep 집계와 같은 연산). 단일 *t* 의존이 설계에서 사라진다는
  장점이 있다.

본 보고서의 실험은 모두 representative 방식을 쓴다. 대표 *t*는 3.1의 진단 그림을 보고
정하였으며 점수는 계산하지 않았다. t = 81을 포함하는 구간은 CA-LoRA의 81을 그대로 쓰고,
그 밖의 구간은 구간 중앙 부근의 측정 지점(201, 375, 750)을, n = 5의 최저 구간은 41을
쓴다. 근거는 진단에서 인접한 *t*의 선택 집합이 재측정 수준으로 비슷하고 차이가 거리에 따라
누적된다는 점이다. 즉 구간 안에서 어느 *t*를 고르는가보다 구간의 **폭**이 더 중요하므로,
구간이 충분히 좁다면 대표 *t*의 결정은 덜 민감해진다. 이것이 "*t* 결정이 불완전해도
괜찮은 방법"이라는 목표와 구간 분할이 만나는 지점이다. aggregate 방식의 평가는 후속
과제로 남긴다.

**(b) 비균등 구간.** CA-LoRA와 선행 작업에서 유효했던 *t*는 모두 저노이즈 영역(t ≤ 81)에
있다. 그래서 구간을 균등하게 나누지 않고 위에서부터 절반씩 잘라 저노이즈 쪽을 더 촘촘하게
한다. T = 1000에서 구간 수 n에 대한 분할 n⟨n⟩은 다음과 같다.

| 분할 | 구간 (t, 양끝 포함) | 대표 *t* |
|---|---|---|
| n1 | [0, 999] | 81 |
| n3 | [0, 249], [250, 499], [500, 999] | 81, 375, 750 |
| n4 | [0, 124], [125, 249], [250, 499], [500, 999] | 81, 201, 375, 750 |
| n5 | [0, 61], [62, 124], [125, 249], [250, 499], [500, 999] | 41, 81, 201, 375, 750 |

**Table 1.** expert 집합과 대표 timestep. n1은 단일 CA-LoRA와 같은 구성이다. n5는 선행
작업의 두 후보 t = 41과 t = 81을 서로 다른 구간으로 분리한다.

구간의 정의가 백본에 묶이지 않도록 경계마다 log-SNR λ = log(ᾱ_t / (1 − ᾱ_t))를 함께
기록한다. SDXL의 noise schedule에서 경계는 다음과 같다.

| 구간 | [0, 61] | [62, 124] | [125, 249] | [250, 499] | [500, 999] |
|---|---|---|---|---|---|
| λ (시작 → 끝) | 7.07 → 2.75 | 2.73 → 1.84 | 1.83 → 0.73 | 0.73 → −0.96 | −0.96 → −5.36 |

**Table 2.** 구간 경계의 log-SNR. 라우팅은 raw *t*로 하며 λ는 정의의 이식을 위한 기록이다.

표에서 보듯 절반 자르기는 log-SNR 축에서 균등 분할이 **아니다**. 구간의 λ 폭은 저노이즈
쪽부터 4.3, 0.9, 1.1, 1.7, 4.4로, 중간 구간이 좁고 양 끝이 넓다. 따라서 이 분할은 noise
level의 이론적 등분이 아니라 "CA-LoRA가 의미를 둔 저노이즈 영역의 *t* 해상도를 높인다"는
사전 지식에 따른 설계로 서술한다. 다른 백본으로 옮길 때에는 raw *t*가 아니라 이 λ 경계를
옮기면 된다. rectified flow 백본이라면 λ = 2 log((1 − t) / t)를 풀어 t = 1 / (1 + e^{λ/2})로
환산한다.

**학습.** expert 하나의 학습은 CA-LoRA의 학습 설정을 그대로 따르되, 학습 timestep을 전체
범위가 아니라 그 expert의 구간에서 균등 추출한다는 점만 다르다. 손실은 L_Diff만 쓰고
(L_Concept은 측정 시점의 양이다), rank 64, batch 1, 상수 학습률 1e-4, AdamW이다. UNet
본체와 텍스트 인코더는 동결한다. 텍스트 인코더에 LoRA를 붙이지 않는 것은 SDXL 파인튜닝
관행과도 맞고, 구간이 바뀔 때마다 프롬프트를 다시 인코딩해야 하는 비용도 피한다. 학습
step은 **expert당 4,000**으로 통일한다. n1(단일 CA-LoRA)도 4,000 step으로 맞추는데, 이는
구성 간 비교에서 step 수를 교란 변수에서 빼기 위한 것이며, 따라서 n1 구성은 "원 논문의
수치"가 아니라 "동일 step 조건의 내부 기준선"이다. 각 expert의 체크포인트에는 구간, 대표
*t*, 선택 집합의 해시가 메타데이터로 기록되고, 학습 중 추출된 *t*가 구간을 벗어나면 학습이
중단된다.

**expert의 공유.** 서로 다른 분할의 두 expert가 같은 구간과 같은 선택 집합을 가지면(예:
n3, n4, n5의 [500, 999] 구간) 같은 시드에서 같은 학습이 되므로 한 번만 학습해 공유한다.
네 분할의 명목상 expert는 1 + 3 + 4 + 5 = 13개지만 고유한 expert는 8개이다.

### 3.3 Assembling: Core-Context Routing over Multiple Expert Sets

**하드 라우팅.** expert 집합을 하나만 쓰는 구성(예: n4 단독)에는 학습할 것이 없다. 각
*t*에서 그 *t*를 포함하는 구간의 expert만 가중치 1로 활성화한다. 구간들이 [0, T)를 빈틈없이
겹치지 않게 덮으므로 *t*마다 정확히 하나의 expert가 활성화되고, timestep당 활성 파라미터는
단일 CA-LoRA와 같은 2%이다.

**core-context 결합.** 여러 분할을 함께 쓰는 구성에서는 TSM의 비대칭 결합을 따른다. 각
*t*에서 모든 분할은 그 *t*를 포함하는 expert 하나씩을 낸다. 구간이 가장 세밀한 분할이
**core**이고 나머지가 **context**이다. 선택된 헤드 슬라이스에만 쓰이는 expert의 갱신을
Δ_e(x)라 하면, projection 행렬 m의 출력은

out_m = W_m x + Δ_core(t)(x) + Σ_k g_{m,k}(z_t, t) · Δ_{k(t)}(x)

이다. core는 게이트 없이 가중치 1로 들어가고, k번째 context 분할의 expert는 게이트
g_{m,k}로 스케일된다. 분할마다 선택 집합이 다르므로 어떤 행렬에는 core만, 어떤 행렬에는
context만 붙어 있을 수 있다. 이 점이 모든 expert가 같은 위치에 붙는 TSM과 다르다.

**라우터.** 게이트는 noised latent와 timestep의 함수이다.

g(z_t, t) = FC(mean_{h,w}(z_t)) + E(t)

FC는 공간 평균한 latent(SDXL에서 4채널)를 받는 선형층이고 E는 timestep 임베딩 테이블이다.
출력 차원은 (게이트가 걸리는 행렬 수 M) × (context 분할 수 K)로, 게이트는 **projection
행렬별**로 따로 있다. 어떤 블록의 to_q는 거친 expert에 기대고 다른 블록의 to_out은 그것을
무시하는 식의 재배분이 가능하다. 설계상의 선택은 다음과 같다.

- **활성화 함수 없음.** 게이트는 선형 출력 그대로이며 softmax도, 범위 제한도, 부호 제한도
  두지 않는다. context가 하나뿐인 구성(n1 + n4)에서 softmax는 항등적으로 1이 되어 아무것도
  학습하지 못한다. 또 범위나 부호를 제한하면 "context expert는 도움이 된다"는, 실험이
  검증해야 할 사전 가정을 구조에 넣게 된다. 게이트는 context를 약화할 수도, 증폭할 수도,
  뺄 수도 있다.
- **항등 초기화.** FC의 가중치와 E는 0으로, FC의 bias는 1로 초기화한다. 따라서 학습 시작
  시점의 모델은 모든 expert를 가중치 1로 단순 합산한 것과 같고, 라우터는 거기서부터의
  이탈만을 학습한다.
- **라우터는 UNet 밖의 별도 모듈**이다. LoRA state에 섞이거나 projection 이름 규칙에
  걸리지 않는다. classifier-free guidance에서는 조건부·비조건부 절반에 같은 게이트를 쓴다.

Assembling 학습에서는 모든 expert와 본체를 동결하고 라우터만 L_Diff로 학습한다. *t*는 전
구간에서 균등 추출하며 1,000 step, batch 8, 학습률 1e-3이다. batch를 Fostering의 1이 아닌
8로 둔 것은, batch 1에서는 손실이 (이미지, *t*) 추출에 지배되어 라우터의 그래디언트 잡음이
지나치게 컸기 때문이다.

**구성.** expert 집합과 core의 조합으로 아홉 구성을 둔다(Table 3). No.1과 No.2는 라우터가
없고, No.3~No.9는 구성마다 라우터를 하나씩 학습한다. 명목 파라미터는 expert 수 × 2%로 최대
26%이고, timestep당 활성 파라미터는 분할 수 × 2%이다.

| No. | expert 집합 | core | 라우터 | expert 수 | t당 활성 | 의도 |
|---|---|---|---|---|---|---|
| 1 | n1 | n1 | 없음 | 1 | 2% | 단일 CA-LoRA(4K step), 내부 기준선 |
| 2 | n4 | n4 | 없음(하드) | 4 | 2% | 구간별 선택 + 구간별 학습만 |
| 3 | n1 + n4 | n4 | 학습 | 5 | 4% | 기준선에 구간 expert를 더함 |
| 4 | n1 + n3 + n4 | n4 | 학습 | 8 | 6% | 보수적인 중간 분할 추가 |
| 5 | n1 + n4 + n5 | n4 | 학습 | 10 | 6% | 세밀한 분할 추가 |
| 6 | n3 + n4 + n5 | n4 | 학습 | 12 | 6% | n1 제외 |
| 7 | n1 + n4 + n5 | n5 | 학습 | 10 | 6% | core의 스케일(No.5와 대조) |
| 8 | n1 + n3 + n4 + n5 | n4 | 학습 | 13 | 8% | 전체 |
| 9 | n1 + n3 + n4 + n5 | n5 | 학습 | 13 | 8% | core의 스케일(No.8과 대조) |

**Table 3.** Assembling 구성. 이론적 근거가 가장 큰 n4를 core의 기본값으로 둔다.

### 3.4 Attribution Controls and an Auxiliary Loss Instrument

제안 방식은 단일 CA-LoRA와 두 가지가 동시에 다르다. 구간별로 **선택**이 다르고, 구간별로
**학습**이 분리된다. 후자는 TSM이 이미 효과를 보고한 요소이므로, 통제 없이는 어떤 이득도
본 연구의 가설(구간별 선택) 덕분이라고 말할 수 없다. 두 개의 통제를 둔다.

- **C1: 선택 고정, 구간별 학습만.** n4의 구간과 학습 설정, 시드를 그대로 두고 네 expert의
  선택 집합을 모두 t = 81의 집합으로 고정한다. 순수한 "TSM + CA-LoRA"이다. No.2와 C1은 선택
  집합만 다르므로 그 차이가 **구간별 선택의 효과**이고, C1과 No.1은 선택 집합이 같고 구간별
  학습 여부만 다르므로 그 차이가 **구간별 학습의 효과**이다. C1이 최고 구성과 같다면 이득은
  구간별 학습에서 온 것이고 본 연구의 가설은 기각된다.
- **C2: 고노이즈 구간 expert 비활성.** CA-LoRA가 full LoRA를 이긴 논리는 "정렬은 덜 하되
  내용 누출을 막는다"였다. 레이아웃과 전역 구조는 고노이즈 단계에서 결정되므로(2.4),
  [500, 999] 구간의 expert가 타깃 도메인의 레이아웃을 학습해 제어성과 다양성을 깎을 위험이
  있다. 게다가 진단에서 이 구간은 선택의 split-half 일치도가 가장 낮은 영역이다. C2는 학습된
  구성에서 [500, 999] expert만 끈다. 하드 라우팅 구성(No.2)에서는 그 구간이 base 모델로
  돌아가고, 라우터 구성(No.3)에서는 그 구간을 n1 expert가 맡으며 라우터를 다시 학습한다.
  두 경우를 각각 C2-n2, C2-n3로 표기한다.

**짝지은 L_Diff.** batch 1 학습의 손실 곡선은 (이미지, *t*) 추출이 지배해 expert가 실제로
학습되었는지를 알려 주지 못한다. 그래서 구간마다 고정된 96개의 (이미지, *t*, ε) 삼중항에서
base 모델과 expert의 L_Diff를 짝지어 비교하는 보조 측정을 둔다. 같은 삼중항에서 (i) 그
구간의 expert, (ii) 전 구간 expert n1을 그 구간에만 적용한 것, (iii) C1의 expert를 비교하면,
평가 지표와는 독립적으로 "구간 학습이 그 구간의 denoising 손실을 더 낮추는가"와 "어떤 선택
집합이 손실을 더 낮추는가"를 볼 수 있다. 이 도구는 학습의 건전성 확인과 해석의 보조로만
쓰며, 구성의 우열은 생성 결과의 평가 지표(4절)로 판단한다.

---

<!-- 이후 절(예정). Improving CA-LoRA 보고서의 구조를 따른다.
## 4. Experimental Setup  (4.1 프로토콜, 4.2 계획 대비 이탈, 4.3 평가: CMMD + CLIP-Score, mIoU 제외)
## 5. Results  (5.1 선택 집합 진단, 5.2 주 결과와 트레이드오프, 5.3 통제 C1/C2, 5.4 라우터 분석, 5.5 paired L_Diff, 5.6 정성)
## 6. Discussion
## 7. Limitations  (생성 seed 1개, mIoU 없음, 강건성 sweep 미실시, 단일 백본)
## 8. Conclusion
## Appendix
-->

## References

[1] Y. Balaji et al. "eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert
    Denoisers." arXiv:2211.01324, 2022.
[2] S. Basu, K. Rezaei, P. Kattakinda, V. I. Morariu, N. Zhao, R. A. Rossi, V. Manjunatha,
    and S. Feizi. "On Mechanistic Knowledge Localization in Text-to-Image Generative
    Models." ICML 2024. arXiv:2405.01008.
[3] J. Choi, J. Lee, C. Shin, S. Kim, H. Kim, and S. Yoon. "Perception Prioritized Training
    of Diffusion Models." CVPR 2022. arXiv:2204.00227.
[4] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke,
    S. Roth, and B. Schiele. "The Cityscapes Dataset for Semantic Urban Scene
    Understanding." CVPR 2016. arXiv:1604.01685.
[5] P. Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image
    Synthesis." ICML 2024. arXiv:2403.03206.
[6] Z. Feng et al. "ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with
    Knowledge-Enhanced Mixture-of-Denoising-Experts." CVPR 2023. arXiv:2210.15257.
[7] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and
    D. Cohen-Or. "An Image is Worth One Word: Personalizing Text-to-Image Generation
    using Textual Inversion." ICLR 2023. arXiv:2208.01618.
[8] H. Go, J. Kim, Y. Lee, S. Lee, S. Oh, H. Moon, and S. Choi. "Addressing Negative
    Transfer in Diffusion Models." NeurIPS 2023. arXiv:2306.00354.
[9] T. Hang, S. Gu, C. Li, J. Bao, D. Chen, H. Hu, X. Geng, and B. Guo. "Efficient
    Diffusion Training via Min-SNR Weighting Strategy." ICCV 2023. arXiv:2303.09556.
[10] J. Hessel, A. Holtzman, M. Forbes, R. Le Bras, and Y. Choi. "CLIPScore: A
    Reference-free Evaluation Metric for Image Captioning." EMNLP 2021. arXiv:2104.08718.
[11] J. Ho, A. Jain, and P. Abbeel. "Denoising Diffusion Probabilistic Models."
    NeurIPS 2020. arXiv:2006.11239.
[12] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen.
    "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.
[13] S. Jayasumana, S. Ramalingam, A. Veit, D. Glasner, A. Chakrabarti, and S. Kumar.
    "Rethinking FID: Towards a Better Evaluation Metric for Image Generation." CVPR 2024.
    arXiv:2401.09603.
[14] T. Karras, M. Aittala, T. Aila, and S. Laine. "Elucidating the Design Space of
    Diffusion-Based Generative Models." NeurIPS 2022. arXiv:2206.00364.
[15] D. P. Kingma and R. Gao. "Understanding Diffusion Objectives as the ELBO with Simple
    Data Augmentation." NeurIPS 2023. arXiv:2303.00848.
[16] Y. Lee, A. S. Chen, F. Tajwar, A. Kumar, H. Yao, P. Liang, and C. Finn. "Surgical
    Fine-Tuning Improves Adaptation to Distribution Shifts." ICLR 2023. arXiv:2210.11466.
[17] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. "Flow Matching for
    Generative Modeling." ICLR 2023. arXiv:2210.02747.
[18] X. Liu, C. Gong, and Q. Liu. "Flow Straight and Fast: Learning to Generate and
    Transfer Data with Rectified Flow." ICLR 2023. arXiv:2209.03003.
[19] M. Park, S. Park, J. Lee, H. Park, K. Hwang, F. Porikli, J. Choo, and S. Choi.
    "Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation." CVPR 2025.
    arXiv:2503.22172.
[20] B. Park, S. Woo, H. Go, J. Kim, and C. Kim. "Denoising Task Routing for Diffusion
    Models." ICLR 2024. arXiv:2310.07138.
[21] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna,
    and R. Rombach. "SDXL: Improving Latent Diffusion Models for High-Resolution Image
    Synthesis." ICLR 2024. arXiv:2307.01952.
[22] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. "High-Resolution
    Image Synthesis with Latent Diffusion Models." CVPR 2022. arXiv:2112.10752.
[23] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. "DreamBooth:
    Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR 2023.
    arXiv:2208.12242.
[24] Q. Zhang, M. Chen, A. Bukharin, N. Karampatziakis, P. He, Y. Cheng, W. Chen, and
    T. Zhao. "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning."
    ICLR 2023. arXiv:2303.10512.
[25] S. Zhuang et al. "TimeStep Master: Asymmetrical Mixture of Timestep LoRA Experts for
    Versatile and Efficient Diffusion Models in Vision." ICML 2025. arXiv:2503.07416.
[26] H. Chung. "Reproducing and Probing Concept-Aware LoRA: Timestep Selection,
    Multi-Timestep Aggregation, and Selection Drift." Technical report, August 2026.
    https://github.com/chs35/Improving-CA-LoRA
[27] Y. Zhang et al. "ProSpect: Prompt Spectrum for Attribute-Aware Personalization of
    Diffusion Models." ACM Transactions on Graphics (SIGGRAPH Asia) 2023. arXiv:2305.16225.
