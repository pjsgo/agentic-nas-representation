# 아키텍처별 자료

7종의 지정 블록·core 복원 검증을 완료했습니다. 각 평가 범위·기준과 생성 코드 수정 여부는 자료 안내에 명시합니다.

| 아키텍처 | 표현 | 복원 코드 | 안내 |
|---|---|---|---|
| ResNet | [YAML](resnet/generator/input.yaml) | [코드](resnet/response_review/generated_model.py) | [범위·판정](resnet/README.md) |
| LSTM | [YAML](lstm/representation.yaml) | [코드](lstm/generated_model.py) | [범위·판정](lstm/README.md) |
| Seq2Seq | [YAML](seq2seq/representation.yaml) | [코드](seq2seq/generated_model.py) | [범위·판정](seq2seq/README.md) |
| Transformer | [YAML](transformer/20261113-core-completion/representation.yaml) | [코드](transformer/20261113-core-completion/generated_model.py) | [범위·판정](transformer/README.md) |
| Linear Transformer | [YAML](linear-transformer/20261113-core-completion/representation.yaml) | [코드](linear-transformer/20261113-core-completion/generated_model.py) | [범위·판정](linear-transformer/README.md) |
| S4 | [YAML](s4/20261113-core-completion/representation.yaml) | [코드](s4/20261113-core-completion/repaired_model.py) | [범위·판정](s4/README.md) |
| Mamba | [YAML](mamba/representation.yaml) | [코드](mamba/generated_model.py) | [범위·판정](mamba/README.md) |

[ResNet 마지막 활성화 편집](resnet-gelu/README.md)은 별도의 편집 사례이며 7종 집계에 추가하지 않습니다. 기존 파일은 이전 실험 자료로 보존합니다.

[파일 출처](manifest.json) · [프로젝트 소개](../README.md)
