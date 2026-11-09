# S4 실험 자료

rank-1 DPLR의 제한된 핵심부, Nfull=4·H=channels=1, 순환 계산을 이용한 임펄스 커널 표현 · **NOT_RUN**

[YAML](representation.yaml) → 후보 생성 미실행 → [검증 코드](validator.py) → [기록된 결과](recorded/results.json)

원본·독립 기준 구현·YAML 사전 검증은 PASS, 생성 후보 평가는 NOT_RUN

선택한 표현은 사전 검증을 통과했다. 실행·객체 동일성 오류 대조 예제 13개와 잘못된 스키마 예제 5개를 검출했다. 생성 후보는 아직 없다.

[생성 입력과 모델 설정](generation-input.json)은 준비된 요청이며 아직 전송·생성을 실행하지 않았습니다.

## 재현 범위

현재 폴더는 실제 실험 자료를 읽고 비교하는 스냅샷입니다. 검증기는 원래 작업 공간의 참조 모듈·fixture·정책 파일에 의존하므로 이 폴더에서 직접 실행하지 않습니다. 전체 평가를 독립 실행 가능한 묶음으로 만드는 작업은 후속 단계입니다. [실행 가능한 ResNet 예제](../resnet/README.md)는 별도로 제공합니다.

[사례 해설](../../case-studies/s4.md) · [실험 목록](../README.md)
