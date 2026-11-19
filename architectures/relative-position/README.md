# Relative Position

부호 있는 거리 clipping과 상대 key/value를 포함한 projected self-attention core.

[복원 YAML](representation.yaml), [최종 생성 코드](reconstructed.py), [출처](sources.yaml), [provenance](provenance.json)를 함께 읽습니다.

상대 key와 상대 value는 서로 다른 외부 테이블이며 batch와 head가 공유합니다. 모든 key가 차단된 행은 범위 밖입니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → Relative Position](../../relations/tr_028.yaml) — A, tr_028
- [Relative Position → Transformer-XL](../../relations/tr_030.yaml) — C, tr_030
- [Relative Position → RoPE](../../relations/tr_046.yaml) — I?, tr_046
