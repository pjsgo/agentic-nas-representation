# Mamba 실험 자료

Mamba-1 selective-scan 핵심부, 검증 조건 4개·비교 텐서 32개 · **INCONCLUSIVE**

[YAML](representation.yaml) → [복원 코드](generated_model.py) → [검증 코드](validator.py) → [기록된 결과](recorded/results.json)

진단용 수치 평가에서 최대 절대·상대 오차 0

Parameter 객체 동일성 검사는 통과했지만, 사전 고정한 구조 특징 비교는 실패했다. softplus 기본 인자의 명시 여부와 미리 할당한 출력 버퍼에 복사하는 구현을 동등하게 취급할지는 미해결이다.

[생성 입력과 모델 설정](generation-input.json)을 함께 볼 수 있습니다. 후보 코드는 원본 그대로입니다.

## 재현 범위

현재 폴더는 실제 실험 자료를 읽고 비교하는 스냅샷입니다. 검증기는 원래 작업 공간의 참조 모듈·fixture·정책 파일에 의존하므로 이 폴더에서 직접 실행하지 않습니다. 전체 평가를 독립 실행 가능한 묶음으로 만드는 작업은 후속 단계입니다. [실행 가능한 ResNet 예제](../resnet/README.md)는 별도로 제공합니다.

[사례 해설](../../case-studies/mamba.md) · [실험 목록](../README.md)
