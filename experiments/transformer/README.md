# Transformer 실험 자료

연산 단위로 표현한 인코더 블록 하나, CPU float32 평가 조건 9개 · **FAIL**

[YAML](representation.yaml) → [복원 코드](generated_model.py) → [검증 코드](validator.py) → [기록된 결과](recorded/results.json)

동일한 검증용 값을 주입한 조건부 연산 비교에서 9개 조건 모두 최대 절대·상대 오차 0

필수 Parameter 객체 12개가 모두 requires_grad=False인 일반 텐서로 생성됐다. 올바른 Parameter 대응을 전제로 하는 엄격한 수치 동등성은 INCONCLUSIVE로 남았다.

[생성 입력과 모델 설정](generation-input.json)을 함께 볼 수 있습니다. 후보 코드는 원본 그대로입니다.

## 재현 범위

현재 폴더는 실제 실험 자료를 읽고 비교하는 스냅샷입니다. 검증기는 원래 작업 공간의 참조 모듈·fixture·정책 파일에 의존하므로 이 폴더에서 직접 실행하지 않습니다. 전체 평가를 독립 실행 가능한 묶음으로 만드는 작업은 후속 단계입니다. [실행 가능한 ResNet 예제](../resnet/README.md)는 별도로 제공합니다.

[사례 해설](../../case-studies/transformer.md) · [실험 목록](../README.md)
