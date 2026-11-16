"""Relation-specific executable graph realization. No upstream code imported.
Only operators of the selected edited block are supported. Not a general compiler.
"""
import torch
from torch import nn
from torch.nn import functional as F

def scan(q,k,v,S,Z,eps):
 out=[]
 for t in range(q.shape[2]):
  qt=F.elu(q[:,:,t])+1; kt=F.elu(k[:,:,t])+1
  S.add_(kt.unsqueeze(-1)*v[:,:,t].unsqueeze(-2)); Z.add_(kt)
  out.append((qt.unsqueeze(-2)@S).squeeze(-2)/((qt*Z).sum(-1,keepdim=True)+eps))
 return torch.stack(out,dim=2),S,Z

class EditedBlock(nn.Module):
 def __init__(self,architecture):
  super().__init__(); self.architecture=architecture
  self.weights=nn.ParameterDict({n:nn.Parameter(torch.empty(s['shape']),requires_grad=s['requires_grad']) for n,s in architecture['parameters'].items()})
 @torch.no_grad()
 def forward(self,x,state=None):
  if x.ndim!=3 or x.shape[-1]!=8 or min(x.shape[:2])<1 or x.dtype!=torch.float32 or x.device.type!='cpu': raise ValueError('Expected CPU float32 [B,T,8], B,T>=1')
  B,T,_=x.shape
  S,Z=state if state is not None else (x.new_zeros(B,2,4,4),x.new_zeros(B,2,4))
  if S.shape!=(B,2,4,4) or Z.shape!=(B,2,4): raise ValueError('carry shape')
  vals={'unit.x':x,'unit.S_in':S,'unit.Z_in':Z,**{'parameters.'+n:p for n,p in self.weights.items()}}
  todo={n:s for n,s in self.architecture['factors'].items() if 'operator' in s}
  edges=self.architecture['dataflow']
  while todo:
   progress=False
   for src,dst in edges:
    if src in vals and dst not in vals: vals[dst]=vals[src]; progress=True
   for name,spec in list(todo.items()):
    ports=self.architecture['interface'][name]['inputs']
    if not all(name+'.'+k in vals for k in ports): continue
    a={k:vals[name+'.'+k] for k in ports}; at=spec.get('attributes',{}); op=spec['operator']
    if op=='linear': y=F.linear(a['input'],a['weight'],a['bias'])
    elif op=='split':
     for i,y in enumerate(torch.split(a['input'],at['sections'],at['dim'])): vals[f'{name}.output{i}']=y
     del todo[name]; progress=True; continue
    elif op=='reshape': y=a['input'].reshape(*[{'B':B,'T':T}.get(v,v) for v in at['shape']])
    elif op=='permute': y=a['input'].permute(*at['dims'])
    elif op=='add': y=a['main']+a['shortcut']
    elif op=='layer_norm': y=F.layer_norm(a['input'],at['normalized_shape'],a['weight'],a['bias'],at['eps'])
    elif op=='relu_functional': y=F.relu(a['input'])
    elif op=='local_causal_elu_scan':
     assert at['feature_map']=='elu_plus_one' and at['time_axis']==2 and at['update_before_read']
     y,so,zo=scan(a['q'],a['k'],a['v'],a['S'],a['Z'],at['eps'])
     vals[name+'.S_out']=so; vals[name+'.Z_out']=zo
    else: raise ValueError(op)
    vals[name+'.output']=y; del todo[name]; progress=True
   if not progress: raise RuntimeError('Unresolved graph: '+str(todo))
  for _ in range(3):
   for src,dst in edges:
    if src in vals and dst not in vals: vals[dst]=vals[src]
  return vals['unit.y'],(vals['unit.S_out'],vals['unit.Z_out']),vals
