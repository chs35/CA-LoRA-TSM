# CA-LoRA-TSM

[English](README.md) | **한국어**

timestep 구간마다 head를 따로 고르고 TimeStep Master 방식의 라우터로 섞는 CA-LoRA (SDXL, Cityscapes).

[CA-LoRA](https://arxiv.org/abs/2503.22172)(CVPR 2026)는 concept-awareness(CA)가 높은 attention head에만 LoRA를 붙이고, CA를 timestep 하나(t = 81)에서 잰다. t를 바꾸면 선택되는 head가 바뀌고 학습된 어댑터도 달라진다. 이 레포는 noise 구간마다 head를 따로 골라 구간별 LoRA expert를 학습하고, [TimeStep Master](https://arxiv.org/abs/2503.07416)(TSM)의 라우터로 expert를 섞는다.

Cityscapes 이미지 10장으로 학습했을 때, 구간별 선택은 도메인 정합(CMMD)을 내주고 날씨 프롬프트 제어성(CLIP-Score)을 얻는다. 라우터 혼합은 CMMD를 CA-LoRA 수준으로 유지하면서 CA-LoRA가 잃은 weather CLIP-Score 11.67점 가운데 2.29점을 되찾지만, 혼합 없는 두 구간 expert 구성보다 낫지는 않다. 혼합이 t 선택에 대한 민감성을 줄였는지는 재지 않았다.

자세한 내용은 **[REPORT.md](REPORT.md)** 에 있다.

## 실행

```bash
pip install -e ".[diffusion,dev]"
python -m pytest tests/ -q                     # CPU, toy 모델, 다운로드 없음

CFG=configs/sdxl_cityscapes_selfgen_fewshot10.yaml
SEL=runs/sdxl_cityscapes_selfgen/selections
python -m catsm.cli select   --config configs/sdxl_cityscapes_selfgen.yaml --scheme n4          # 구간별 head 선택
python -m catsm.cli foster   --config $CFG --expert n4_i0 --selection $SEL/n4/n4_i0.json    # 구간 expert 학습
python -m catsm.cli assemble --config $CFG --sets n1 n4 --core n4                           # 라우터 학습
python scripts/generate_images.py --config $CFG --run-name no3 \
    --bundle runs/sdxl_cityscapes_selfgen_fewshot10/bundles/n1+n4-core-n4/bundle.safetensors
python scripts/evaluate.py        --config $CFG --run-name no3                           # CMMD, CLIP-Score
```

CA 측정은 Improving-CA-LoRA에서 한 번 하고 `scripts/ops/import_measurements.sh`로 가져온다. `scripts/ops/`에는 GPU 서버(A100 한 대)에서 실험을 돌린 스크립트가 있다. 기본값은 이 레포에 결과가 없는 이전 실험의 것이고, 보고한 실험은 `scripts/ops/run_selfgen_chain.sh`가 설정과 run 이름을 지정해 돌렸다. Cityscapes는 포함하지 않으며 [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)에서 따로 받아야 한다.

## 폴더 구조

```
configs/      실험 설정 (보고한 실험은 sdxl_cityscapes_selfgen_fewshot10.yaml)
scripts/      CA 측정, 진단, 생성, 평가, 결과 표와 그림
src/catsm/    구간, 구간별 선택, 다중 expert sliced LoRA, 라우터, 학습
src/calora/   CA-LoRA 재구현, Improving-CA-LoRA에서 복사 (CA-LoRA는 공식 코드가 없다)
tests/        toy 모델 단위 테스트
results/      보고서의 근거가 되는 평가 결과(selfgen_fewshot10/)와 그림(figures/)
```

## 관련 레포

[Improving-CA-LoRA](https://github.com/chs35/Improving-CA-LoRA)(CA 측정 timestep을 정량 기준으로 고르기) → **CA-LoRA-TSM**(timestep 하나에 의존하지 않기) → [Hierarchical-CFG](https://github.com/chs35/Hierarchical-CFG)(샘플링 단계에서 제어성 되찾기).

## 라이선스

[MIT](LICENSE). SDXL 가중치와 Cityscapes 이미지는 포함하지 않으며 각자의 라이선스를 따른다.
