# 아키텍처별 자료

구조를 기술한 YAML과 그 입력으로 생성한 복원 코드를 정리합니다. 계보 기반 연구가 진행됨에 따라 아키텍처와 실험을 추가합니다.

| 아키텍처 | 표현 | 복원 코드 | 안내 |
|---|---|---|---|
| ResNet | [YAML](resnet/generator/input.yaml) | [코드](resnet/response_review/generated_model.py) | [자료 안내](resnet/README.md) |
| Transformer | [YAML](transformer/representation.yaml) | [코드](transformer/generated_model.py) | [자료 안내](transformer/README.md) |
| LSTM | [YAML](lstm/representation.yaml) | [코드](lstm/generated_model.py) | [자료 안내](lstm/README.md) |
| Seq2Seq | [YAML](seq2seq/representation.yaml) | [코드](seq2seq/generated_model.py) | [자료 안내](seq2seq/README.md) |
| Linear Transformer | [YAML](linear-transformer/representation.yaml) | — | [자료 안내](linear-transformer/README.md) |
| S4 | [YAML](s4/representation.yaml) | — | [자료 안내](s4/README.md) |
| Mamba | [YAML](mamba/representation.yaml) | [코드](mamba/generated_model.py) | [자료 안내](mamba/README.md) |
| ResNet · 마지막 활성화 편집 | [YAML](resnet-gelu/generator/input.yaml) | [코드](resnet-gelu/response_review/generated_model.py) | [자료 안내](resnet-gelu/README.md) |
`—`는 아직 복원 코드가 없는 자료입니다. 각 자료는 선택된 블록·핵심 연산을 대상으로 하며, 이 목록은 검증 판정표가 아닙니다.

[파일 출처](manifest.json) · [프로젝트 소개](../README.md)
