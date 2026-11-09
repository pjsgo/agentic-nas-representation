# Seq2Seq 실험 자료

임베딩된 입력을 받는 단층 LSTM 인코더·디코더 핵심부, teacher forcing 비율 1, 검증 조건 3개 · **INCONCLUSIVE**

[YAML](representation.yaml) → [복원 코드](generated_model.py) → [검증 코드](validator.py) → [기록된 결과](recorded/results.json)

원본 구현 대비 최대 절대 오차 5.960464477539063e-08, 독립 기준 구현 대비 0

상태 전달, 반복 계산, teacher forcing, 등록된 Parameter 10개 검사는 통과했다. 사전 고정한 실행 그래프 검사는 실패했으며, 후속 분석에서 0 상수 생성 시 backend가 추가한 detach 차이를 분리했다.

[생성 입력과 모델 설정](generation-input.json)을 함께 볼 수 있습니다. 후보 코드는 원본 그대로입니다.

## 별도 후속 평가

2026-11-09의 별도 구조 평가 v1.1에서는 동일 후보가 고정 CPU float32 조건 3개에서 PASS를 받았습니다. 새로 생성한 0 상수 내부의 backend detach만 제한적으로 다루며, 원래 INCONCLUSIVE 판정은 유지합니다. [결과 보고서](followups/20261109_structure_v1_1/results.json)

## 재현 범위

현재 폴더는 실제 실험 자료를 읽고 비교하는 스냅샷입니다. 검증기는 원래 작업 공간의 참조 모듈·fixture·정책 파일에 의존하므로 이 폴더에서 직접 실행하지 않습니다. 전체 평가를 독립 실행 가능한 묶음으로 만드는 작업은 후속 단계입니다. [실행 가능한 ResNet 예제](../resnet/README.md)는 별도로 제공합니다.

[사례 해설](../../case-studies/seq2seq.md) · [실험 목록](../README.md)
