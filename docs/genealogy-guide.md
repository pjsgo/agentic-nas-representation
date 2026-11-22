# 계보 관계 YAML

[관계 목록](../relations/README.md)의 항목 하나가 [계보도](genealogy-map.md)의 화살표 하나를 설명합니다. 같은 두 모델을 연결하더라도 비교하거나 이어받은 대상이 다르면 서로 다른 ID를 사용합니다.

| 필드 | 의미 |
|---|---|
| `transition_id` | 원본 계보의 관계 ID |
| `source_architecture` | 원본 화살표의 출발 모델 |
| `target_architecture` | 원본 화살표의 도착 모델 |
| `relation_type` | A는 구조 계승, C는 부품 계승, I는 아이디어 계승, P는 병목 대응, X는 대안 비교 |
| `inferred` | 원본이 관계에 물음표를 붙였으면 true |
| `summary` | 이 모델 쌍에서 계승한 부분, 대응한 문제 또는 비교한 해법을 설명하는 영어 한 문장 |
| `evidence` | 원본 자료의 파일명과 해당 쪽 |

예를 들어 ConvS2S와 Transformer를 연결한 [tr_025](../relations/tr_025.yaml)는 합성곱과 self-attention을 비교하고, [tr_072](../relations/tr_072.yaml)는 토큰에 위치 표현을 더하는 부품을 연결합니다. 모델 내부의 연산과 상태는 [아키텍처 YAML 안내](representation-guide.md)에서 확인할 수 있습니다.
