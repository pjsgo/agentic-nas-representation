# RoPE

명시 위치의 인접 feature 쌍 회전과 attention core. 연구용 실현기가 독립 참조 대표 사례 3개를 통과했습니다. 외부 LLM 생성 원본을 수정하지 않고 독립 참조 대표 사례 3개를 통과했습니다.

[복원 YAML](representation.yaml), [최종 생성 코드](reconstructed.py), [출처](sources.yaml), [provenance](provenance.json)를 함께 읽습니다.

모든 짝수 feature를 인접 쌍으로 회전하며 value는 회전하지 않습니다. 서로 다른 query/key 길이와 위치는 수식 범위의 확장으로, 공식 cross-attention API 실행을 뜻하지 않습니다.

전체 모델 학습과 성능 재현은 제외합니다. 입력, 파라미터와 상태 계약 및 제외 범위는 YAML에 명시했습니다.

[아키텍처 목록](../README.md)
