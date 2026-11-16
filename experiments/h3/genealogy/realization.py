"""Agent-authored realization of the explicit local YAML graph; no reference import.
Not a frozen common-schema compiler. Float32 parameters / complex64 external carry.
"""
from pathlib import Path
import torch
from torch import nn
import yaml

class YAMLCore(nn.Module):
    def __init__(self, architecture, values):
        super().__init__()
        self.spec = yaml.safe_load(Path(architecture).read_text())
        if self.spec['format_version'] != 'local_h3_core_v1':
            raise ValueError('unsupported dialect')
        c = self.spec['configuration']
        if c != dict(d_model=2, head_dim=1, d_state=4, stored_pairs=2):
            raise ValueError('only declared scalar-head configuration supported')
        if set(values) != set(self.spec['parameters']):
            raise ValueError('parameter keys differ')
        for name, shape in self.spec['parameters'].items():
            value = values[name]
            if list(value.shape) != shape or value.dtype != torch.float32:
                raise ValueError(name)
        self.raw = nn.ParameterDict({k: nn.Parameter(v.detach().clone()) for k,v in values.items()})

    def forward(self, u, shift0, diagonal0):
        if u.ndim != 3 or u.shape[1] < 1 or u.shape[2] != 2 or u.dtype != torch.float32:
            raise ValueError('u must be float32 B,T,2 with T>=1')
        if shift0.shape != (u.shape[0],2,4) or shift0.dtype != torch.float32:
            raise ValueError('shift state contract')
        if diagonal0.shape != (u.shape[0],2,2) or diagonal0.dtype != torch.complex64:
            raise ValueError('diagonal state contract')
        env = dict(self.raw.items(), u=u, shift0=shift0, diagonal0=diagonal0)
        for node in self.spec['graph']:
            a = [env[key] for key in node['inputs']]
            op = node['op']
            if op == 'affine': r = a[0] @ a[1].T + a[2]
            elif op in ('multiply','scale'): r = a[0]*a[1]
            elif op == 'add': r = a[0]+a[1]
            elif op == 'exp': r = a[0].exp()
            elif op == 'negative_exp': r = -a[0].exp()
            elif op == 'complex': r = torch.complex(a[0],a[1])
            elif op == 'complex_pairs': r = torch.view_as_complex(a[0].contiguous())
            elif op == 'bilinear':
                A,B,dt = a; z = dt[:,None]*A/2
                r = ((1+z)/(1-z), dt[:,None]*B/(1-z))
            elif op == 'shift_ssm':
                x,s,B,C = a; ys=[]
                for xt in x.unbind(1):
                    s = torch.cat((torch.zeros_like(s[...,:1]),s[...,:-1]),-1)+B*xt[...,None]
                    ys.append((C*s).sum(-1))
                r = (torch.stack(ys,1),s)
            elif op == 'diagonal_ssm':
                x,s,A,B,C = a; ys=[]
                for xt in x.unbind(1):
                    s = A*s+B*xt[...,None]
                    ys.append(2*(C*s).sum(-1).real)
                r = (torch.stack(ys,1),s)
            else: raise ValueError(f'unsupported operation {op}')
            values = r if isinstance(r,tuple) else (r,)
            if len(values) != len(node['outputs']): raise ValueError('output arity')
            for key,value in zip(node['outputs'],values):
                if key in env: raise ValueError('duplicate binding')
                env[key]=value
        return tuple(env[key] for key in self.spec['interface']['outputs'])
