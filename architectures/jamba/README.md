# Jamba

축소 8층 주기, GQA와 Mamba mixer, dense 및 top-2 SwiGLU expert, 상태 이어 계산.

범위: one_original_eight_layer_cycle; pre_norm_residuals; causal_grouped_query_attention; full_mamba_mixer; normalized_dt_B_C; dense_swiglu; full_capacity_top2_swiglu_experts; explicit_continuation_state

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처와 한계](sources.yaml), [구현 기록](provenance.json)

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Mamba → Jamba](../../relations/tr_064.yaml) — C, tr_064
- [Transformer → Jamba](../../relations/tr_065.yaml) — A, tr_065
- [Switch Transformer → Jamba](../../relations/tr_066.yaml) — C?, tr_066
