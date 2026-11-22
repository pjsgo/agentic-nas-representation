# ALiBi

고정 head slope와 causal 거리 bias attention core.

[복원 YAML](representation.yaml), [최종 생성 코드](reconstructed.py), [출처](sources.yaml), [provenance](provenance.json)를 함께 읽습니다.

헤드별 기울기는 persistent buffer에 저장합니다. 공식 key-column bias와 정규화 전 score는 행별 상수만큼 다르며 softmax를 적용하면 행별 상수 차이가 사라집니다.

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → ALiBi](../../relations/tr_047.yaml) — A, tr_047
