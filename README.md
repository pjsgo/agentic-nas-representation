# Agentic NAS · Architecture Representation

## 연구 방향

팀에서는 아키텍처 계보(Genealogy)를 바탕으로 신경망 구조를 이해하고, 에이전트가 구조를 편집·탐색할 수 있도록 하는 Agentic NAS를 연구하고 있습니다. 계보에 담긴 구조적 관계를 실제 모델의 표현과 구현으로 연결하는 것이 연구 방향입니다.

## 맡은 부분

저는 이 과정에서 **신경망 구조의 표현과 코드 복원·검증**을 맡고 있습니다. 모델의 연산과 연결, 객체 공유, 상태 전달을 YAML로 표현하고, 이를 바탕으로 LLM이 생성한 코드가 해당 구조를 유지하는지 확인하는 작업입니다. 계보 분석은 팀의 다른 연구원이 담당합니다.

## 현재 작업

선택된 블록과 핵심 연산을 대상으로 YAML 표현, 복원 코드, 검증 도구를 구성했습니다. 이 레포에는 실제 산출물을 아키텍처별로 정리하고 있으며, 연구 진행에 맞춰 자료를 추가합니다.

[아키텍처별 YAML·복원 코드](experiments/README.md) · [작업 요약](results/summary.md) · [실행 예제](experiments/resnet/README.md)

## 앞으로의 방향

현재 표현의 기본 구조를 바탕으로 필요한 정보를 구체화하며, 연구에서 다루는 **아키텍처 계보 전체로 표현과 복원 범위를 확장**할 계획입니다. 팀 공유 자료에 아키텍처와 실험이 추가될 때 이 레포도 함께 갱신합니다. 이후에는 이 표현·검증 작업을 에이전트의 구조 편집과 탐색으로 연결하고자 합니다.

[팀 공유 자료](https://github.com/UVR-WJCHO/AI4Science_archive/tree/junseo/nas-artifacts/JunseoPark) · [자료 출처](docs/artifact-provenance.md)
