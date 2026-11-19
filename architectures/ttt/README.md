# TTT

단일 head TTT-Linear 내부 학습기, chunk 내 causal 갱신과 경계 상태 이어 계산.

범위: single_head_TTT_Linear_inner_learner; learned_view_projections; residual_LN_learner; input_dependent_learning_rate; frozen_chunk_gradients; causal_prefix_updates; learned_initial_state; learned_token_scaling; chunk_boundary_continuation

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처와 한계](sources.yaml), [구현 기록](provenance.json)

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Mamba → TTT](../../relations/tr_067.yaml) — P, tr_067
- [Transformer → TTT](../../relations/tr_068.yaml) — X, tr_068
