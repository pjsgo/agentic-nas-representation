"""Small executor for this modified YAML only; agent authored, not S4 compiler output."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import yaml

class ChangedStateLayer(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        path = path or Path(__file__).with_name('representation.yaml')
        self.spec = yaml.safe_load(Path(path).read_text())
        if self.spec['operator_profile'] != 'local_tr061_v1_not_common_schema':
            raise ValueError('This executor supports only local_tr061_v1')
        self.resources = nn.ParameterDict({name:nn.Parameter(torch.zeros(p['shape'],dtype=torch.float32),requires_grad=p['requires_grad']) for name,p in self.spec['parameters'].items()})
        self.connections = {}
        for src,dst in self.spec['dataflow']:
            if dst in self.connections: raise ValueError('Duplicate destination: '+dst)
            self.connections[dst] = src
        self.order = self.spec['factors']['unit']['contains']

    def forward(self,u,h0,diagnostics=False):
        if u.dtype != torch.float32 or h0.dtype != torch.float32 or u.device.type!='cpu' or h0.device.type!='cpu':
            raise ValueError('CPU float32 only')
        if u.ndim!=3 or u.shape[1]!=1 or u.shape[2]<1 or tuple(h0.shape)!=(u.shape[0],1,2):
            raise ValueError('Expected u[B,1,T>=1], h0[B,1,2]')
        values={'unit.u':u,'unit.h0':h0,**{'parameters.'+k:v for k,v in self.resources.items()}}
        for node in self.order:
            factor=self.spec['factors'][node]; op=factor['operator']; a=factor.get('attributes',{})
            ports={p:values[self.connections[node+'.'+p]] for p in self.spec['interface'][node]['inputs']}
            if op=='local_select':
                if a['split_order']!=['dt','B','C'] or a['delta_activation']!='softplus_after_bias':
                    raise ValueError('Unsupported selection semantics')
                r,n=a['dt_rank'],a['state_size']
                mixed=F.linear(ports['u'].transpose(1,2),ports['x_weight'])
                dt,B,C=torch.split(mixed,[r,n,n],dim=-1)
                delta=F.softplus(F.linear(dt,ports['dt_weight'],ports['dt_bias'])).transpose(1,2)
                outputs={'delta':delta,'B':B,'C':C}
            elif op=='local_selective_scan':
                if a!={'A':'negative_exp','discrete_A':'exp_delta_A','discrete_B':'delta_B','readout':'updated_state','axis':2}:
                    raise ValueError('Unsupported scan semantics')
                A=-ports['A_log'].exp()
                delta=ports['delta']
                decay=(delta.unsqueeze(-1)*A[None,:,None,:]).exp()
                drive=delta.unsqueeze(-1)*ports['B'][:,None,:,:]*ports['u'].unsqueeze(-1)
                state=ports['h0']; rows=[]
                for t in range(u.shape[2]):
                    state=decay[:,:,t,:]*state+drive[:,:,t,:]
                    rows.append((state*ports['C'][:,None,t,:]).sum(-1))
                outputs={'sequence':torch.stack(rows,dim=2),'sf':state}
            elif op=='unsqueeze': outputs={'output':ports['input'].unsqueeze(a['dim'])}
            elif op=='mul': outputs={'output':ports['left']*ports['right']}
            elif op=='add':
                if a.get('inplace',False): raise ValueError('In-place add excluded')
                outputs={'output':ports['main']+ports['shortcut']}
            else: raise ValueError('Unsupported operator: '+op)
            values.update({node+'.'+p:v for p,v in outputs.items()})
        result=tuple(values[self.connections['unit.'+p]] for p in ['y','h_final'])
        return (*result,values) if diagnostics else result
