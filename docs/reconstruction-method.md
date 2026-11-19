# YAML 기반 아키텍처 복원

논문과 원본 구현을 바탕으로 연산, 연결, 파라미터와 상태 전달을 YAML에 적습니다. 코드 생성에는 이 YAML과 연산의 해석 규칙, 입출력 형식을 제공합니다. 생성한 코드는 각 모델 폴더의 `reconstructed.py`에서 볼 수 있습니다.

예를 들어 LSTM의 YAML에는 한 시점의 입력과 이전 상태로 게이트를 계산하는 순서가 담겨 있습니다. 구현에서는 같은 가중치로 이 계산을 반복하고 출력과 마지막 상태를 반환합니다.

[LSTM 명세](../architectures/lstm/representation.yaml) / [LSTM 구현](../architectures/lstm/reconstructed.py) / [모델 목록](../architectures/README.md)
