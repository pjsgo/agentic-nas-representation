# LSTM 실험 자료

셀 내부 연산을 분해해 표현한 단방향 시퀀스 처리, 입력 차원 I=4·은닉 차원 H=5 고정 · **FAIL**

[YAML](representation.yaml) → [복원 코드](generated_model.py) → [검증 코드](validator.py) → [기록된 결과](recorded/results.json)

T=1/3/5에서 출력 시퀀스와 hidden/cell state 오차 0, T=0에서는 출력 생성 실패

원래 전체 범위 평가는 빈 시퀀스 처리 실패로 FAIL이다. 원본 후보와 판정을 그대로 보존한 별도 T≥1 평가는 PASS다.

[생성 입력과 모델 설정](generation-input.json)을 함께 볼 수 있습니다. 후보 코드는 원본 그대로입니다.

## 재현 범위

현재 폴더는 실제 실험 자료를 읽고 비교하는 스냅샷입니다. 검증기는 원래 작업 공간의 참조 모듈·fixture·정책 파일에 의존하므로 이 폴더에서 직접 실행하지 않습니다. 전체 평가를 독립 실행 가능한 묶음으로 만드는 작업은 후속 단계입니다. [실행 가능한 ResNet 예제](../resnet/README.md)는 별도로 제공합니다.

[T≥1 후속 평가](evaluations/t_ge_1_v1/results.json)는 원래 T=0 실패를 대체하지 않습니다.

[사례 해설](../../case-studies/lstm.md) · [실험 목록](../README.md)
