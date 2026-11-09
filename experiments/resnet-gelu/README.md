# ResNet · 지정된 마지막 활성화 편집

[YAML](generator/input.yaml) → [생성 입력](generation-input.json) → [복원 코드](response_review/generated_model.py) → [검증 코드](validate_edit.py) → [기록된 결과](recorded/results.json)

사람이 지정한 마지막 ReLU → GELU 변경입니다. 첫 ReLU를 유지하고 마지막 활성화에 별도 자원을 부여했습니다. [YAML diff](yaml_changes.diff)와 [편집 조건](edit_specification.md)을 함께 볼 수 있습니다. 편집 기준 구현과 8개 입력에서 일치했고 원래 블록과는 모두 출력이 달랐습니다.

이 폴더는 원본 산출물 스냅샷입니다. 검증기에 필요한 기존 바이너리 fixture와 편집 기준 구현은 포함하지 않아 단독 실행은 지원하지 않습니다. [기준 ResNet의 오프라인 검증](../resnet/README.md)은 실행할 수 있습니다.

[사례 해설](../../case-studies/resnet.md) · [전체 실험](../README.md)
