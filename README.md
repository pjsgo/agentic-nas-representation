# Agentic NAS · 신경망 구조 표현과 복원

신경망 구조를 YAML로 표현하고, LLM이 복원한 코드가 연산·객체 공유·상태 전달을 보존하는지 연구하고 있습니다.

현재는 선택된 블록과 핵심 연산을 대상으로 실험하고 있습니다. YAML의 기본 구조를 기준으로 유지하면서, 연구에서 다루는 **아키텍처 계보 전체로 표현과 복원을 확장**하고 필요한 정보를 구체화할 계획입니다. 장기적으로는 에이전트의 구조 편집·탐색으로 연결하려 합니다.

진행 중인 팀 연구 · 이 레포는 제가 담당하는 표현·재구성·검증 작업의 자료를 정리합니다.

## 실제 자료

| 사례 | YAML · 복원 코드 · 결과 |
|---|---|
| ResNet BasicBlock | [YAML](experiments/resnet/generator/input.yaml) · [코드](experiments/resnet/response_review/generated_model.py) · [7개 입력과 구조 검사 통과](experiments/resnet/README.md) |
| 마지막 ReLU → GELU 편집 | [YAML](experiments/resnet-gelu/generator/input.yaml) · [코드](experiments/resnet-gelu/response_review/generated_model.py) · [편집 결과](experiments/resnet-gelu/README.md) |
| Transformer 인코더 핵심부 | [YAML](experiments/transformer/representation.yaml) · [코드](experiments/transformer/generated_model.py) · [Parameter 요건 위반 사례](experiments/transformer/README.md) |

[전체 실험 자료](experiments/README.md) · [범위별 결과](results/summary.md) · [팀 공유 자료의 기준 버전](https://github.com/UVR-WJCHO/AI4Science_archive/tree/3a6b02d51b05c545f4d742b7341afd5abe05f81b/JunseoPark)

실험은 서로 다른 제한된 범위를 다룹니다. 생성 완료와 검증 통과를 구분하며, 전체 모델이나 자율 NAS 성능의 검증을 뜻하지 않습니다.

## 실행 예제

저장된 ResNet 후보를 API 호출 없이 검증할 수 있습니다.

```bash
python tools/reproduce_resnet.py --output outputs/resnet-run-01
```

[환경과 실행 안내](experiments/resnet/README.md) · [확인된 실행 결과](results/reproduction/resnet-20261109/summary.json). 나머지 사례는 현재 자료 열람을 지원합니다.

## 참고

[표현 설계](docs/representation-design.md) · [검증 방법](docs/validation-methodology.md) · [담당 범위](docs/contributions.md) · [업데이트 방식](docs/updating.md) · [자료 출처](docs/artifact-provenance.md)
