# Hyena

입력 gating, short filter와 implicit long filter, causal FFT convolution core.

범위: input_projection; causal_depthwise_short_filter; position_generated_long_filter; shared_frequency_sine_MLP; exponential_modulation; input_gating; causal_FFT_convolution; diagonal_skip; output_projection

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처와 한계](sources.yaml), [구현 기록](provenance.json)

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [WaveNet → Hyena](../../relations/tr_052.yaml) — I?, tr_052
- [S4 → Hyena](../../relations/tr_053.yaml) — X?, tr_053
- [H3 → Hyena](../../relations/tr_075.yaml) — I, tr_075
- [Transformer → Hyena](../../relations/tr_076.yaml) — A, tr_076
