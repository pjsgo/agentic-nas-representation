# GQA

외부 QKV의 query-to-KV 그룹 대응, causal prefix와 G-head cache.

[복원 YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)을 함께 읽습니다.

입력, dtype, shape, 파라미터와 상태의 계약은 YAML의 `interface`, `parameters`, `state`를 따릅니다. 제외 범위는 `scope.excluded`에 명시했습니다.

캐시는 G개의 KV head만 반환하며 새 저장 공간을 사용합니다. 원본의 사전 할당 제자리 캐시 API, projection, RoPE와 checkpoint 평균화는 포함하지 않습니다. 공식 Meta 구현은 GQA 논문 저자의 구현과 구분합니다.

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → GQA](../../relations/tr_056.yaml) — A, tr_056
