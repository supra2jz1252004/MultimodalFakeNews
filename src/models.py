"""Small matched classifiers on frozen features."""
import torch
from torch import nn

NAMES = {'text': 'Text Only', 'image': 'Image Only', 'baseline': 'Text + Image',
         'proposed': 'Text + Image + Similarity'}


class Classifier(nn.Module):
    def __init__(self, kind: str, dimension: int = 128, dropout: float = .3):
        super().__init__()
        if kind not in NAMES:
            raise ValueError(kind)
        self.kind = kind
        self.text = nn.Linear(768,dimension) if kind!='image' else None
        self.image = nn.Linear(2048,dimension) if kind!='text' else None
        inputs = dimension * (2 if kind in ['baseline','proposed'] else 1)
        self.fusion = nn.Linear(inputs,dimension)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dimension,2)
        if kind=='proposed':
            # Identical C/D initialization and predictions; only the new scalar weights start at zero.
            self.fusion.weight = nn.Parameter(torch.cat([self.fusion.weight,torch.zeros(dimension,1)],dim=1))
            self.fusion.in_features += 1

    def forward(self, text, image, similarity):
        parts = []
        if self.text is not None:
            parts.append(torch.relu(self.text(text)))
        if self.image is not None:
            parts.append(torch.relu(self.image(image)))
        if self.kind=='proposed':
            parts.append(similarity)
        fused = torch.cat(parts,dim=1)
        return self.output(self.dropout(torch.relu(self.fusion(fused))))


def verify_matched_initialization(dimension=128,dropout=.3):
    torch.manual_seed(42)
    baseline = Classifier('baseline',dimension,dropout).eval()
    torch.manual_seed(42)
    proposed = Classifier('proposed',dimension,dropout).eval()
    for name,value in baseline.state_dict().items():
        other = proposed.state_dict()[name]
        assert torch.equal(value,other[:,:-1] if name=='fusion.weight' else other),name
    inputs = [torch.randn(5,n) for n in [768,2048,1]]
    assert torch.allclose(baseline(*inputs),proposed(*inputs),atol=1e-7)
    assert sum(p.numel() for p in proposed.parameters())-sum(p.numel() for p in baseline.parameters())==dimension
