# Reformer

제공된 회전의 LSH attention, chunk와 이전 chunk 참조, multi-round 결합.

범위: shared_qk; supplied_random_rotations; angular_hash; round_bucket_offsets; bucket_position_sort; chunk_attention; previous_chunk_lookup; finite_self_penalty; optional_causal_penalty; inverse_sort; multi_round_partition_weighted_merge.

제외: projection_layers; output_projection; reversible_residual; ffn_chunking; embeddings; dropout; backward; training; optimized_kernels; padding_mask; incremental_cache; rng_generation_and_state; factorized_hash; automatic_bucket_count; paper_appendix_duplicate_correction; strict_same_bucket_mask.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → Reformer](../../relations/tr_033.yaml) — A, tr_033
