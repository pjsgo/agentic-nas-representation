# RoPE

명시 위치의 인접 feature 쌍 회전과 attention core.

[복원 YAML](representation.yaml), [최종 생성 코드](reconstructed.py), [출처](sources.yaml), [provenance](provenance.json)를 함께 읽습니다.

모든 짝수 feature를 인접 쌍으로 회전하며 value는 회전하지 않습니다. 서로 다른 query/key 길이와 위치는 수식 범위의 확장으로, 공식 cross-attention API 실행을 뜻하지 않습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Relative Position → RoPE](../../relations/tr_046.yaml) — I?, tr_046
