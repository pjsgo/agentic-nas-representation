# Compressive Transformer

평균 풀링 압축, 두 단계 FIFO memory attention.

[복원 YAML](representation.yaml) / [최종 구현](reconstructed.py) / [논문과 공식 코드](sources.yaml) / [범위, 코드 출처](provenance.json)

## 계산 흐름

compressed, episodic, 현재 memory → joint attention → 오래된 상태 평균 풀링 → memory 갱신.

## 사용 계약

`build(spec, parameters)`

`forward(self, x, episodic, compressed, index)`

CPU float64, YAML에 선언된 크기, dtype, 파라미터, 외부 상태 계약을 따릅니다. 생성자에 필요한 파라미터/자원은 호출자가 준비하며, `spec`이 필요한 경우 이 YAML을 파싱해 전달합니다.

## 범위

제외: learned_conv_compression, dilated_conv, max_pooling, most_used_compression, auxiliary_attention_reconstruction_loss, auxiliary_autoencoding_loss, full_training, residual_norm_ffn_tower, input_embedding_mlp, single_token_sampling_schedule, dropout. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer-XL → Compressive Transformer](../../relations/tr_031.yaml) — A, tr_031
