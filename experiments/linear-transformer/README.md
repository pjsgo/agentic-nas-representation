# Linear Transformer

**지정 core 복원 검증 완료.** ELU+1 recurrent linear-attention core; external q/k/v and in-place S/Z state, four CPU float32 fixtures.

[YAML](20261113-core-completion/representation.yaml) · [복원 코드](20261113-core-completion/generated_model.py)

실용 core PASS. 원본 Common Schema의 stateless Module identity BLOCKED 유지.

평가 기준: practical core and current-candidate recheck. 상세 실행 증거는 연구 작업 공간에서 관리하며 [요약](../../results/summary.json)에 출처와 해시를 기록합니다. 기존 자료는 보존합니다.

[전체 자료](../README.md) · [프로젝트 소개](../../README.md)
