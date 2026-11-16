# 복원 범위와 검증

## 현재 범위

53개 아키텍처의 선언된 계산 범위에 최종 성공 근거가 있습니다. 전체 모델 인증은 0개입니다. 아키텍처별 포함 및 제외 범위, 입력 계약, 생성 결과와 수정 이력은 [아키텍처 목록](../architectures/README.md)에서 각 문서와 `provenance.json`을 통해 확인합니다.

외부 생성 후보의 과거 채택 기록은 원본 26개, 수정본 27개입니다. 서로 다른 평가 기준과 후속 실험이 포함되어 있으므로 공통 기준의 최초 시도 성공률로 해석하지 않습니다. 현재 ConvS2S와 Transformer에는 로컬 작성 위치 계산이 통합되어 있습니다. 이 확장을 외부 생성 성공으로 추가하거나 현재 파일 전체를 무수정 생성 원본으로 간주하지 않습니다.

## 검증 근거

다음은 확인 범위와 날짜가 명시된 검증 기록입니다. 사례 수와 검사 수는 서로 다른 단위이며 합산하지 않습니다. 저장된 결과를 검토한 날짜와 수치 실행 여부를 구분합니다. 전체 대상의 세부 근거는 각 아키텍처 provenance를 따릅니다.

| 기록 날짜 | 대상과 확인 범위 | 근거와 결과 |
|---|---|---|
| 2026-11-16 | HiPPO, Hyena, RetNet, RWKV, Memory Network, E2E Memory Network, LSTMN, RETRO, Jamba, TTT의 배포 구현 | 독립 참조와 계약 검사 실행, 각 3개 총 30개 대표 사례가 선언 범위 내 통과 |
| 2026-11-17 | ConvS2S, QRNN, Decomposable Attention, Structured Self-Attention, WaveNet, ByteNet, Neural LM, RNN LM의 생성 후보 | 각 3개 총 24개 대표 사례의 저장된 독립 참조 결과를 검토, 수치 재실행 없이 근거 재사용 |
| 2026-11-17 | ConvS2S와 Transformer 내부의 위치 계산 및 블록 연결 | 독립 스칼라 수식, YAML 연결 실행, 입력 계약 등을 포함한 [36개 검사 통과](position-integration-validation.json) |

2026-11-16의 10개 대상 중 RWKV와 Memory Network는 생성 원문을 채택했고, 나머지 8개는 원문 실패 후 수정본을 채택했습니다. 계산 코드, 참조, 입력 사례, 허용 오차를 변경하지 않고 배포 구현을 확인했습니다. Memory Network 검증 helper의 경로 누락을 수정한 뒤 관련 검사를 완료했으며, 해당 수정은 모델 계산 변경과 구분합니다.

2026-11-17에 검토한 8개 후보 중 QRNN, Structured Self-Attention, Neural LM, RNN LM은 계산 수정 없는 원본이고, ConvS2S, Decomposable Attention, WaveNet, ByteNet은 수정본입니다. 이 판정은 당시 생성 후보에 대한 것이며 이후 위치 계산 확장과 구분합니다. 두 날짜의 검증은 기존 생성 시도와 근거를 사용했고, 통합 과정에서 외부 재생성을 하지 않았습니다.

위치 계산의 이전 25개 검사는 별도 구현 당시의 기록입니다. 현재 사용 범위는 아키텍처 내부에 통합한 경로의 36개 검사와 해당 provenance를 기준으로 판단합니다. 이 연결 검사는 기존 블록 전체나 공식 runtime의 동등성을 새로 인증하지 않습니다.

## 판정과 출처를 읽는 방법

응답 형식 준수, 계산 결과, 수정 여부는 별도로 판단합니다. [Decomposable Attention](../architectures/decomposable-attention/provenance.json)과 [Structured Self-Attention](../architectures/structured-self-attention/provenance.json)은 응답 형식 실패 및 오래된 `generation=NOT_ATTEMPTED` 메타데이터를 실제 생성 시도 기록과 구분합니다. [Neural LM](../architectures/neural-lm/provenance.json)과 [RNN LM](../architectures/rnn-lm/provenance.json)은 JSON 바깥 Markdown fence 제거를 계산 수정으로 집계하지 않습니다.

Longformer의 중복 생성 요청, H3의 후속 생성, LSTM의 양의 길이 제한, Seq2Seq의 평가 기준 변경 등 결과 해석에 영향을 주는 이력은 각 provenance에 보존합니다. 최종 통과가 과거 실패나 시도 이력을 지우지는 않습니다.

## 관계의 완료 의미

76개 관계 중 전체 대응은 16개 `complete_scoped`, 60개 `partial`입니다. 선택 core의 상태는 [관계 목록](../relations/README.md)에서 별도로 확인합니다. `tr_052`는 복원된 WaveNet 부품을 참조하며, `tr_072`는 ConvS2S와 Transformer 내부의 위치 부품을 참조합니다. `tr_072`의 core는 `complete_scoped`이고 전체 대응은 `partial`입니다.

모든 공개 관계는 `executable=false`입니다. 부품 참조가 존재한다는 검사는 수치 동등성이나 자동 변환 인증이 아닙니다. `tr_037`과 `tr_061`의 과거 범위 제한 실행 근거도 전체 자동 변환 성공으로 확대하지 않습니다.

과거 정적 검토에서는 2026-11-16의 45개 아키텍처와 64개 관계에 대해 678개 부품 참조, 2026-11-17의 53개 아키텍처와 76개 관계에 대해 764개 부품 참조를 확인했습니다. 이는 당시 표현에 대한 기록이며 위치 부품 통합 이후의 참조 개수로 재사용하지 않습니다. 부품 ID 오류와 카탈로그의 추론 여부 불일치를 거부하는 검토도 포함했습니다.

## 사용 범위와 한계

전체 모델, 학습, 성능, 공식 runtime 동등성과 공통 단일 스키마 인증을 주장하지 않습니다. 공식 소스 확인 공백과 모델별 제외 범위는 [잔여 범위](../relations/remaining-scope.yaml) 및 각 provenance를 따릅니다. 위치 계산은 명시적 위치 ID를 사용하며 자동 번호 부여, 공식 padding 규칙, 가중치 이식은 검증 범위 밖입니다. 계보의 혁신과 한계를 종합한 역사적 해석 역시 이 검증만으로 확정하지 않습니다.

[전체 관계도](genealogy-map.md) / [관계별 상태](../relations/README.md) / [복원 방법](reconstruction-method.md)
