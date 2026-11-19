# Agentic NAS: Architecture Reconstruction

신경망의 구조를 YAML로 기술하고, 그 명세를 바탕으로 핵심 계산을 PyTorch 코드로 구현했습니다. 아키텍처를 나란히 보면 한 모델의 어떤 계산이 다음 모델에서 이어지고 달라지는지 살펴볼 수 있습니다.

**아키텍처 53개와 그 사이의 관계 76개**를 담았습니다.

## 세 모델로 보는 메모리의 변화

```mermaid
flowchart LR
    A["Transformer<br/>현재 구간 안에서 참조"] --> B["Transformer-XL<br/>이전 구간의 상태도 참조"]
    B --> C["Compressive Transformer<br/>오래된 상태를 압축해 보관"]
```

| 모델 | 코드에서 볼 수 있는 계산 | 구현 |
|---|---|---|
| Transformer | 현재 구간의 각 토큰이 같은 구간의 다른 토큰을 참조합니다. | [YAML](architectures/transformer/representation.yaml) / [PyTorch](architectures/transformer/reconstructed.py) |
| Transformer-XL | 이전 구간의 은닉 상태를 메모리에 보관해 현재 구간의 어텐션에 사용합니다. | [YAML](architectures/transformer-xl/representation.yaml) / [PyTorch](architectures/transformer-xl/reconstructed.py) |
| Compressive Transformer | 메모리가 차면 밀려나는 상태를 압축해 별도로 보관합니다. 어텐션은 최근 상태와 압축된 상태를 함께 읽습니다. | [YAML](architectures/compressive-transformer/representation.yaml) / [PyTorch](architectures/compressive-transformer/reconstructed.py) |

[전체 아키텍처와 구현](architectures/README.md) / [전체 계보도](docs/genealogy-map.md)
