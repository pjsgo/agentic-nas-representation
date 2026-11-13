# LSTM

**지정 core 복원 검증 완료.** Single-layer unidirectional core, T>=1; tested T=1,3,5, B=2,I=4,H=5.

[YAML](representation.yaml) · [복원 코드](generated_model.py)

T≥1·어댑터 기반 검증 PASS. T=0 포함 원래 FAIL 및 CLI 연결 INCONCLUSIVE 유지.

평가 기준: restricted positive-length follow-up. 상세 실행 증거는 연구 작업 공간에서 관리하며 [요약](../../results/summary.json)에 출처와 해시를 기록합니다. 기존 자료는 보존합니다.

[전체 자료](../README.md) · [프로젝트 소개](../../README.md)
