# Agentic NAS · Architecture Representation

## 연구 방향

AI Agent가 신경망 아키텍처를 탐색해 더 나은 시퀀스 모델을 찾도록 하는 연구입니다. 팀에서는 아키텍처 계보(Genealogy)를 따라 모델 사이에 **무엇이 어떻게 변했는지와 그 변화의 배경을 구조화해 정리**했습니다. 이 맥락을 에이전트에 제공해 구조 선택과 편집에 활용하고자 합니다.

## 맡은 부분 — Factorization

저는 신경망 구조를 연산과 연결 단위로 분해하고, 계산 흐름·반복·상태 전달·객체 공유를 YAML로 표현하는 **Factorization과 코드 복원·검증**을 맡고 있습니다. 구조화된 계보가 변화의 맥락을 제공한다면, Factorization은 그 맥락에서 다루는 모델 구조를 구체적으로 표현하는 역할을 합니다. 표현에서 LLM으로 코드를 복원하고, 원본과 비교해 계산과 구조가 보존되는지 확인합니다.

## 현재 작업

선택된 블록과 핵심 연산을 대상으로 표현 설계, 코드 복원, 수치·구조 비교 실험을 진행했습니다. 이 레포에는 실제 YAML과 복원 코드를 아키텍처별로 정리하고 있습니다. 현재는 에이전트의 자율 탐색에 사용할 표현과 검증 기반을 마련하는 단계입니다.

[아키텍처별 YAML·복원 코드](experiments/README.md) · [작업 요약](results/summary.md)

## 앞으로의 방향

연구에서 다루는 계보 전체로 표현과 복원 범위를 넓히고, **계보의 변화 맥락과 Factorization을 함께 에이전트에 제공**할 계획입니다. 이후 EvoScientist를 활용해 구조 선택·편집, 코드 생성, 실험·평가를 반복하는 흐름으로 연결하고자 합니다. 팀 공유 자료에 아키텍처와 실험이 추가될 때 이 레포도 함께 갱신합니다.

[팀 공유 자료](https://github.com/UVR-WJCHO/AI4Science_archive/tree/junseo/nas-artifacts/JunseoPark) · [자료 출처](docs/artifact-provenance.md)
