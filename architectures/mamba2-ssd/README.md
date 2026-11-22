# Mamba-2 / SSD

scalar-identity multihead SSD chunk core.

[복원 YAML](representation.yaml) / [최종 구현](reconstructed.py) / [논문과 공식 코드](sources.yaml) / [범위, 코드 출처](provenance.json)

## 계산 흐름

selective 이산화 → chunk 내부 계산, 경계 상태 전달 → chunk 간 계산 → D skip.

## 사용 계약

`build(values)`

`forward(self, x, dt_logits, B, C, initial_state)`

CPU float64, YAML에 선언된 크기, dtype, 파라미터, 외부 상태 계약을 따릅니다. 생성자에 필요한 파라미터/자원은 호출자가 준비하며, `spec`이 필요한 경우 이 YAML을 파싱해 전달합니다.

## 범위

제외: input_projection, causal_convolution, gating_RMSNorm, output_projection, grouped_BC, fused_GPU_kernels, full_network, training. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Mamba → Mamba-2 / SSD](../../relations/tr_063.yaml) — A, tr_063
