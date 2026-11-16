# 복원 범위와 검증

원본 대상 53개 모두 선언된 계산 범위의 최종 성공 근거를 확보했습니다. 외부 생성 후보의 과거 채택 기록은 원본 26개, 수정본 27개이며 전체 모델 인증은 0개입니다. 서로 다른 과거 평가 기준과 후속 실험을 포함하므로 26개를 공통 기준의 최초 시도 성공으로 해석하지 않습니다.

## 검증 기록 (2026-11-17)

신규 ConvS2S, QRNN, Decomposable Attention, Structured Self-Attention, WaveNet, ByteNet, Neural LM, RNN LM의 기존 독립 참조 결과를 검토하고 재사용했습니다. 각 3개, 총 24개 대표 사례가 선언 범위 내 통과했습니다. QRNN, Structured Self-Attention, Neural LM, RNN LM은 계산 수정 없는 원본, 나머지 4개는 별도 수정본입니다. 원본 실패와 수정 내용을 각 provenance에 보존했습니다. 기존 45개를 이유 없이 재실행하지 않았습니다.

R2의 응답 형식 실패와 계산 판정은 별개이며, 결과 파일의 NOT_ATTEMPTED는 validator의 오래된 메타데이터입니다. 실제 시도 기록을 별도로 확인했습니다. R4는 JSON 바깥 Markdown fence만 제거했습니다. I의 신규 외부 생성 호출은 없습니다.

tr_072의 위치 계산은 ConvS2S와 Transformer의 기존 YAML과 구현에 통합되어 있습니다. 각 아키텍처의 출처와 provenance에 로컬 작성 이력을 기록합니다. 기존 25개 위치 검증은 과거 근거이며, 현재 통합 경로와 표현 실행은 [통합 검증](position-integration-validation.json)을 따릅니다. 외부 생성 후보의 26개 원본/27개 수정본은 과거 채택 집계이며 현재 확장을 무수정 생성 결과로 뜻하지 않습니다.

## 관계의 완료 의미

76개 관계 중 전체 대응은 16개 complete_scoped, 60개 partial입니다. 신규 12개는 선택 core 대응을 기록했습니다. tr_052의 WaveNet 끝점 공백은 해소됐고, tr_072는 아키텍처 내부 위치 부품을 통해 core complete_scoped, 전체 partial입니다. 기존 관계에는 core partial도 남아 있습니다. 모든 공개 관계는 executable=false이며, 참조 존재 검사는 수치 동등성이나 자동 변환 인증이 아닙니다.

## 사용 범위와 남은 한계

전체 모델, 학습, 성능, 공식 runtime 동등성과 공통 단일 스키마 인증을 주장하지 않습니다. 공식 소스 확인 공백과 모델별 제외 범위는 [잔여 범위](../relations/remaining-scope.yaml) 및 각 provenance를 따릅니다. tr_037과 tr_061의 과거 실행 근거는 전체 자동 변환 성공으로 확대하지 않습니다. Longformer 중복 생성, H3 후속 생성, LSTM 양의 길이 제한, Seq2Seq 후속 평가 기준 등 과거 이력을 보존합니다. innovation과 한계의 종합 해석은 후속 연구 범위입니다.

[전체 관계도](genealogy-map.md), [관계별 상태](../relations/README.md)

## 이전 검증 기록 (2026-11-16)

HiPPO, Hyena, RetNet, RWKV, Memory Network, E2E Memory Network, LSTMN, RETRO, Jamba, TTT의 최종 후보를 배포 디렉터리에서 불러 기존 독립 참조 및 계약 검사로 실행했습니다. 각각 대표 사례 3개, 총 30개가 선언 범위 내 통과했습니다. 계산 코드, 참조, 입력 사례, 허용오차는 바꾸지 않았습니다. 기존 35개 수치 검사는 반복하지 않았습니다. Memory Network 검증 helper의 배포 검증 폴더 누락을 고친 후 세 계열을 정상 실행했습니다.

| 대상 | 생성 결과 | 최종 후보 평가 |
|---|---|---|
| RWKV, Memory Network | 원문 통과 | 각 3개 통과 |
| HiPPO, Hyena, RetNet, E2E Memory Network, LSTMN, RETRO, Jamba, TTT | 원문 실패 후 별도 수정본 | 각 3개 통과 |

기존 생성 시도 10건을 재사용했습니다. 통합에서 새 API 호출이나 재생성을 하지 않았습니다. 원문 실패 이력은 성공으로 변경하지 않았습니다. 최종 코드의 출처와 제외 범위는 각 provenance와 YAML에 있습니다.
