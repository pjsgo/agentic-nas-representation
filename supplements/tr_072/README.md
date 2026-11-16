# 위치 표현 보충 부품

[tr_072 관계](../../relations/tr_072.yaml)의 선택 위치 표현 대응을 위한 I 작성 코드입니다. 기존 ConvS2S와 Transformer YAML은 유지합니다.

[ConvS2S YAML](convs2s.yaml), [Transformer YAML](transformer.yaml), [구현](position.py), [출처](sources.yaml), [검증 요약](evaluation.yaml)

ConvS2S는 caller의 token 벡터와 명시적 위치 table lookup을 합산합니다. source와 이미 shifted된 target에 독립 table을 적용합니다. Transformer는 unscaled token 벡터를 sqrt(D)로 확대하고 선택한 sinusoidal 위치 벡터를 더합니다. transformer_forward는 기존 UnitA의 D=8, float32 입력에 연결합니다.

CPU eval, 명시적 위치 ID만 지원합니다. padding을 자동으로 번호 매기거나 합산 결과에서 제거하지 않으며, mask를 기존 블록에 전달합니다. Transformer의 기존 mask는 공통 [T,T]입니다. 원본 전체 모델, 학습, 체크포인트 이식, 두 위치 표현의 값 동등성은 범위 밖입니다. 이 코드는 외부 생성 성공 개수에 포함되지 않습니다.
