# Relative Position

부호 있는 거리 clipping과 상대 key/value를 포함한 projected self-attention core. 연구용 실현기가 독립 참조 대표 사례 3개를 통과했습니다. 외부 LLM 생성 원본을 수정하지 않고 독립 참조 대표 사례 3개를 통과했습니다.

[복원 YAML](representation.yaml), [최종 생성 코드](reconstructed.py), [출처](sources.yaml), [provenance](provenance.json)를 함께 읽습니다.

상대 key와 상대 value는 서로 다른 외부 테이블이며 batch와 head가 공유합니다. 모든 key가 차단된 행은 범위 밖입니다.

전체 모델 학습과 성능 재현은 제외합니다. 입력, 파라미터와 상태 계약 및 제외 범위는 YAML에 명시했습니다.

[아키텍처 목록](../README.md)
