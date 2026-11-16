# reformer

생성 원본은 실패했으며 별도 수정본이 독립 참조 검사를 통과했습니다. 대표 사례 3개의 선언된 범위 내 결과입니다.

범위: shared_qk; supplied_random_rotations; angular_hash; round_bucket_offsets; bucket_position_sort; chunk_attention; previous_chunk_lookup; finite_self_penalty; optional_causal_penalty; inverse_sort; multi_round_partition_weighted_merge.

제외: projection_layers; output_projection; reversible_residual; ffn_chunking; embeddings; dropout; backward; training; optimized_kernels; padding_mask; incremental_cache; rng_generation_and_state; factorized_hash; automatic_bucket_count; paper_appendix_duplicate_correction; strict_same_bucket_mask.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [생성 및 평가 이력](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.
