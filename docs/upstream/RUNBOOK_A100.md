# A100 GPU-Day Runbook (RunPod pod)

> 작성: 2026-07-28, Fable 5. GPU 붙이기 전 CPU 단계에서 준비된 것과, GPU가 붙은 직후 실행할 순서.
> 이 pod에서 **/workspace(네트워크 볼륨)만 재배포에서 살아남는다.** 환경·캐시·데이터 전부 그 안에 있다.

## 환경 (준비 완료)

| 항목 | 위치 | 비고 |
|---|---|---|
| Python venv | `/workspace/venv-calora` | `source /workspace/venv-calora/bin/activate`. torch 2.9.1(CUDA), torchvision 0.24.1, diffusers/transformers/peft/accelerate/scipy/clean-fid/pytest |
| HF 캐시 | `/workspace/.cache/huggingface` | `HF_HOME` 환경변수로 이미 지정됨. SDXL base + CLIP ViT-L/14 프리다운로드됨 |
| 데이터셋 | `/workspace/data/cityscapes` | leftImg8bit(2975/500/1525) + gtFine(×4/이미지) 추출·검증 완료. config `data.root`에 반영됨 |
| 저장소 | `/workspace/Optimizing-CA-LoRA` | origin = github.com/chs35/Optimizing-CA-LoRA |

### 이 pod의 함정 (겪은 것만 기록)

- **mfs(FUSE) 마운트에서 `unzip`·`pip`이 몇 분씩 멈춘다** (FUSE 요청 대기). 설치는 `uv`(캐시는 로컬 디스크 `UV_CACHE_DIR=/root/.cache/uv`), 대량 파일 추출은 스레드 병렬 Python(zipfile)으로 할 것. 순수 대역폭은 550–650MB/s로 문제없음.
- **⚠️ mfs가 쓰기를 조용히 손상시킨 사례 2건**: (a) gtFine 추출 중 EIO 후 파일 2개 손상, (b) venv 설치본 22,960개 중 6개 손상(`libtorch_cuda.so` 포함 → import segfault). **대량 쓰기 후에는 반드시 해시/CRC 검증**할 것. venv는 RECORD sha256 전수 검증으로 복구 완료(2026-07-29). 체크포인트 저장 후에도 `safetensors` 재로드 검증 권장.
- **uv가 `/workspace`의 `.lock`(flock)에서 영원히 대기할 수 있다.** 프로세스를 죽였으면 `rm -f <venv>/.lock` 후 재시도.
- **컨테이너 cgroup 메모리 한도 8GB** (호스트 표기 503GB와 무관; HF 다운로드가 OOM kill 당했음 → xet 비활성 + worker 축소로 우회). GPU pod에서 한도를 먼저 확인할 것: `cat /sys/fs/cgroup/memory.max`. **config의 `dtype: float32`로 SDXL을 로드하면 GPU 전송 전 CPU RAM ~13GB가 필요** — 한도가 8GB면 즉사하므로 반드시 더 큰 한도로 배포해야 함 (fp16 로드 검증은 이 pod에서 완료: 33s, UNet 2.57B params).
- ~~CPU가 2코어뿐이다.~~ GPU pod은 128코어 (2026-07-29 A100 부착 후). **torch가 스레드를 128개로 잡아 작은 텐서 CPU 연산이 수백 배 느려진다** (pytest 34s→2h17m 실측). 모든 실행에 `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16` (테스트는 8)을 붙일 것.
- **GPU pod 재배포 후 `HF_HOME`이 사라져 있었다.** 모든 실행에 `HF_HOME=/workspace/.cache/huggingface`를 명시할 것 — 없으면 로컬 디스크(~9GB 여유)로 SDXL 재다운로드를 시도하다 실패한다.
- GPU pod cgroup 메모리 한도는 250GB로 배포됨 (fp32 SDXL 로드 문제 없음, 2026-07-29 확인).

## 차단 항목 (GPU 전 반드시 해소)

- [ ] **`src/calora/data/` 패키지가 저장소에 없다** — `.gitignore`의 `data/` 패턴이 삼켜서 initial commit에서 누락. 로컬 Windows 저장소에서 push 필요 (아래 지시 참조). 해소 전에는 어떤 스크립트도 실행 불가.
- [ ] push 후 이 pod에서: `git pull` → `pytest tests/ -q` (194개 기대) → `python scripts/smoke_test.py` (toy CPU end-to-end).

## GPU가 붙은 직후 (순서대로)

```bash
source /workspace/venv-calora/bin/activate
cd /workspace/Optimizing-CA-LoRA
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
python -m pytest tests/ -q                 # CPU 테스트가 GPU 환경에서도 통과하는지
python scripts/smoke_test.py               # toy 파이프라인 회귀 확인
```

**1. A100 스모크 (수 분, dummy 데이터·Cityscapes 불필요):**
```bash
python scripts/measure_ca.py --config configs/smoke_a100.yaml
python scripts/score_timesteps.py --config configs/smoke_a100.yaml
python scripts/select_projections.py --config configs/smoke_a100.yaml
python scripts/train_lora.py --config configs/smoke_a100.yaml
python scripts/generate_images.py --config configs/smoke_a100.yaml --run-name smoke
python scripts/evaluate.py --config configs/smoke_a100.yaml --run-name smoke
```
이때 확인 (PROJECT_LOG §5 미해결 항목): gradient checkpointing×retain_graph 동작, 캡션 캐시 인코딩 경로 동등성, 측정 dtype 정책(bf16 forward+fp32 누적 vs 순수 fp32) — VRAM 여유를 보고 결정.

**2. 본 측정 (Phase 2–3):**
```bash
python scripts/measure_ca.py --config configs/tsweep_sdxl.yaml     # 13개 t, 256 샘플, m=4
python scripts/score_timesteps.py --config configs/tsweep_sdxl.yaml
```
직후 최우선 확인 2가지 (PROJECT_LOG §3):
- `direction_concentration_mean`이 1/√n 랜덤 바닥보다 유의하게 높은가 (특히 viewpoint 축 — eye-level 템플릿 상쇄 우려).
- conflict quantile sweep {0.1..0.5}에서 t 순위가 뒤집히지 않는가 (뒤집히면 conflict_count 주 지표 자격 상실).
- split-half agreement (Spearman, top-5 overlap)로 t 순위의 통계적 신뢰성.

**3. 선택 + 학습 (Phase 4, arm별):**
```bash
python scripts/select_projections.py --config configs/sdxl_cityscapes.yaml   # arm별 selection JSON
python scripts/train_lora.py --config configs/sdxl_cityscapes.yaml           # 10k steps, drift 콜백 내장
python scripts/generate_images.py --config configs/sdxl_cityscapes.yaml --run-name <arm>
python scripts/evaluate.py --config configs/sdxl_cityscapes.yaml --run-name <arm>
```
arm 매트릭스는 PROJECT_LOG §4. 미결: arm당 seed 수 (기본 1, 여유 시 A/B/E만 3).

## 불변식 리마인더 (CLAUDE.md와 동일)

- CA 측정 batch_size=1, 측정은 crop=center + no-flip (호출부 강제).
- seed 파생에 t를 넣지 말 것 (paired_across_timesteps).
- cos_of_means(주) + mean_of_cos(대조) 둘 다 로깅.
