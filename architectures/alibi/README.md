# ALiBi

고정 head slope와 causal 거리 bias attention core. 연구용 실현기가 독립 참조 대표 사례 3개를 통과했습니다. 외부 LLM 생성 원본을 수정하지 않고 독립 참조 대표 사례 3개를 통과했습니다.

[복원 YAML](representation.yaml), [최종 생성 코드](reconstructed.py), [출처](sources.yaml), [provenance](provenance.json)를 함께 읽습니다.

2의 거듭제곱이 아닌 head 수의 slope 순서와 persistent slopes buffer를 확인했습니다. 공식 key-column bias와 정규화 전 score는 행별 상수만큼 다르며 softmax와 출력의 동등성만 비교합니다.

전체 모델 학습과 성능 재현은 제외합니다. 입력, 파라미터와 상태 계약 및 제외 범위는 YAML에 명시했습니다.

[아키텍처 목록](../README.md)
