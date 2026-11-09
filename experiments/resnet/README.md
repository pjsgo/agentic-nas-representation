# ResNet 실험 자료

입력을 그대로 더하는 지름길 연결을 가진 BasicBlock, 채널 수 64, CPU float32 평가 모드 · **PASS**

[YAML](generator/input.yaml) → [복원 코드](response_review/generated_model.py) → [검증 코드](evaluate_forward.py) → [기록된 결과](recorded/results.json)

기준 모델 입력 7/7개 통과, 최대 절대 오차 0

사람이 지정한 마지막 활성화 함수 ReLU → GELU 변경도 별도로 검증했다. 입력 8/8개에서 편집된 기준 구현과 일치했고, 모두 원래 모델의 출력과 달랐다.

[생성 입력과 모델 설정](generation-input.json)을 함께 볼 수 있습니다. 후보 코드는 원본 그대로입니다.

## 실행하기

Python 3.10 환경에서 `requirements.txt`의 패키지를 준비한 뒤 레포 루트에서 실행합니다. CPU 실행이며 가중치를 다운로드하거나 API를 호출하지 않습니다.

```bash
python -m pip install -r requirements.txt
python tools/reproduce_resnet.py --output outputs/resnet-run-01
```

Torch와 Torchvision은 플랫폼에 맞는 CPU 배포판을 사용해도 됩니다. 이미 호환 환경이 있으면 설치 단계는 생략합니다. 다른 버전은 별도 검증이 필요합니다.

당시 [평가 계획](evaluation_plan.json)에 따라 seed로 입력과 비기본 BN 상태를 다시 만들고, 설치된 Torchvision BasicBlock과 비교합니다. 원본 검증기·생성 코드를 임시 디렉터리에서 실행하며 고정 artifact의 SHA256을 먼저 확인합니다. 검사에는 7개 입력의 수치 비교, 6개 Parameter와 6개 Buffer 대응, 공유 ReLU 두 번 호출, 데이터 흐름, in-place 연산과 입력 보존이 포함됩니다.

출력 폴더에는 `summary.json`, `validation/results.json`, 실행 로그와 재생성한 텐서가 남습니다. 실패 시 종료 코드 1을 반환합니다. 동일 폴더를 덮어쓰지 않으므로 다시 실행할 때는 새 이름을 지정합니다. `python -O`는 사용할 수 없습니다.

이 실행은 저장된 후보의 검증을 반복합니다. LLM을 다시 호출해 같은 코드를 생성하는 실험이나 YAML 파서 검증은 아닙니다. [GELU 편집 실험](../resnet-gelu/README.md)은 별도 기록입니다.

[사례 해설](../../case-studies/resnet.md) · [실험 목록](../README.md)

2026-11-09 공개 예제 확인: **7/7 통과**. [재실행 요약](../../results/reproduction/resnet-20261109/summary.json) · [상세 검사 결과](../../results/reproduction/resnet-20261109/results.json). GELU 편집은 이번 재실행에 포함하지 않았습니다.
