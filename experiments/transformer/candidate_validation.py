"""Reviewed candidate adapter with semantic state and FX runtime verification.
Mapping hints are checked against actual tensors, operations and direct edges.
Unsupported tracing/mapping is BLOCKED, never numerical/schema PASS.
"""
import hashlib
import importlib.util
import json
import operator
from pathlib import Path
import torch
from torch import fx
from torch.nn import functional as F
from graph_engine import validate,execute,TRACE_ROLES,require
from reference import Reference,SHAPES

class Capture(fx.Interpreter):
    def __init__(self,module):
        super().__init__(module); self.values={}; self.calls={}
    def run_node(self,node):
        args,kwargs=self.fetch_args_kwargs_from_env(node)
        value=super().run_node(node)
        self.values[node.name]=value; self.calls[node.name]=(node,args,kwargs)
        return value

def same(a,b,rtol,atol):
    return isinstance(a,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.allclose(a,b,rtol=rtol,atol=atol,equal_nan=False)

def runtime_structure(doc,cap,mapping,parameters,x,mask):
    checked=validate(doc); factors=doc['factors']; interface=doc['interface']
    require(set(mapping)==checked['leaves'],'complete leaf traceability mapping required')
    named_nodes=[v for v in mapping.values() if isinstance(v,str)]
    require(len(named_nodes)==len(set(named_nodes)),'distinct leaf invocation mapping')
    outputs={checked['root']+'.x':x,checked['root']+'.mask':mask}
    outputs.update({'parameters.'+k:v for k,v in parameters.items()})
    covered=set()
    for key in checked['order']:
        f=factors[key]; op=f['operator']; attr=f.get('attributes',{}); hint=mapping[key]
        if op=='constant' and isinstance(hint,dict):
            require(set(hint)=={'literal'} and hint['literal']==attr['value'],'constant literal')
            value=hint['literal']; outputs[key+'.output']=value; continue
        require(isinstance(hint,str) and hint in cap.calls,'missing FX node '+key)
        node,args,kwargs=cap.calls[hint]; value=cap.values[hint]; covered.add(hint)
        target=node.target; method=node.op=='call_method'
        def argument(index,name,default=None): return kwargs.get(name,args[index] if len(args)>index else default)
        def expect(port,actual):
            wanted=outputs[checked['resolve'](key+'.'+port)]
            if isinstance(wanted,(int,float)):
                require(isinstance(actual,(int,float)) and actual==wanted,'literal input edge '+key+'.'+port)
            else: require(actual is wanted,'direct tensor input edge '+key+'.'+port)
        if op=='linear':
            require(node.op=='call_function' and target is F.linear,'linear operation')
            for i,p in enumerate(['input','weight','bias']): expect(p,argument(i,p))
        elif op=='split':
            require((method and target=='split') or target in {torch.split,torch.functional.split},'split operation')
            expect('input',argument(0,'tensor')); sections=argument(1,'split_size_or_sections'); dim=argument(2,'dim',0)
            require(list(sections)==attr['sections'] and dim%3==attr['dim'],'split attributes')
        elif op in {'reshape','permute','transpose','unsqueeze'}:
            acceptable={'reshape':{torch.reshape},'permute':{torch.permute},'transpose':{torch.transpose},'unsqueeze':{torch.unsqueeze}}
            require((method and target==op) or target in acceptable[op],op+' operation')
            expect('input',argument(0,'input'))
            if op=='permute':
                raw=args[1:] if method else (argument(1,'dims'),)
                dims=list(raw[0]) if len(raw)==1 and isinstance(raw[0],(list,tuple)) else list(raw)
                require(dims==attr['dims'],'head/layout permutation')
            elif op=='transpose':
                require(argument(1,'dim0')%4==attr['dim0'] and argument(2,'dim1')%4==attr['dim1'],'transpose attributes')
            elif op=='unsqueeze': require(argument(1,'dim')==attr['dim'],'unsqueeze attributes')
            else:
                raw=args[1:] if method else (argument(1,'shape'),)
                shape=list(raw[0]) if len(raw)==1 and isinstance(raw[0],(tuple,list)) else list(raw)
                env={'B':x.shape[0],'T':x.shape[1]}
                expected=[env[d] if isinstance(d,str) else d for d in attr['shape']]
                require(shape==expected,'reshape attributes')
        elif op in {'mul','matmul','add'}:
            allowed={'mul':{operator.mul,torch.mul},'matmul':{operator.matmul,torch.matmul},'add':{operator.add,torch.add}}
            require(target in allowed[op] or (method and target==op),op+' operation')
            names=['main','shortcut'] if op=='add' else ['left','right']
            expect(names[0],argument(0,'input')); expect(names[1],argument(1,'other'))
            require(kwargs.get('alpha',1)==1,'unexpected arithmetic alpha')
        elif op=='masked_fill':
            require((method and target=='masked_fill') or target is torch.masked_fill,'masked_fill operation')
            expect('input',argument(0,'input')); expect('mask',argument(1,'mask'))
            require(argument(2,'value')==float('-inf'),'masked_fill value')
        elif op=='softmax':
            require((method and target=='softmax') or target in {torch.softmax,F.softmax},'softmax operation')
            expect('input',argument(0,'input')); require(argument(1,'dim')%4==attr['dim'],'softmax axis')
            require(kwargs.get('dtype') is None,'unexpected softmax dtype')
        elif op=='relu_functional':
            require(target in {torch.relu,F.relu} or (method and target=='relu'),'activation operation')
            expect('input',argument(0,'input')); require(argument(1,'inplace',False) is False,'relu inplace')
        elif op=='layer_norm':
            require(target is F.layer_norm,'layer_norm operation')
            expect('input',argument(0,'input')); expect('weight',argument(2,'weight')); expect('bias',argument(3,'bias'))
            require(list(argument(1,'normalized_shape'))==attr['normalized_shape'] and argument(4,'eps',1e-5)==attr['eps'],'norm attributes')
        elif op=='constant':
            require(isinstance(value,torch.Tensor) and value.numel()==1 and float(value)==attr['value'] and value.dtype==torch.float32,'constant value/dtype')
        else: raise ValueError('unverified operator '+op)
        vals=value if op=='split' else [value]
        require(len(vals)==len(interface[key]['outputs']),'output arity')
        for port,v in zip(interface[key]['outputs'],vals): outputs[key+'.'+port]=v
    # FX indexing of a split tuple and shape extraction are bookkeeping, not hidden tensor calculations.
    for name,(node,args,kwargs) in cap.calls.items():
        if name in covered or node.op in {'placeholder','output','get_attr'}: continue
        bookkeeping=(node.op=='call_function' and node.target in {operator.getitem,getattr}) or (node.op=='call_method' and node.target=='size')
        require(bookkeeping,'unmapped candidate computation '+name)
        if isinstance(cap.values[name],torch.Tensor):
            require(any(cap.values[name] is v for v in outputs.values()),'unmapped tensor view/index '+name)
    return {role:outputs[endpoint] for role,endpoint in TRACE_ROLES.items()}

def evaluate(path,root,plan,cases,rtol,atol,*,prepared_model=None,prepared_adapter=None):
    result={'status':'BLOCKED','schema_consistency':'BLOCKED','structural_fidelity':'BLOCKED',
            'numerical_fidelity':'BLOCKED','resource_identity':'BLOCKED','evidence':[]}
    try:
        path=Path(path).resolve()
        if prepared_model is None:
            require(path.is_relative_to(root) and path.is_file(),'candidate must be owned local file')
            review=root/'validation/candidate_review.json'; adapter_path=root/'candidate_adapter.json'
            require(review.is_file() and adapter_path.is_file(),'review and candidate_adapter.json required before import')
            record=json.loads(review.read_text()); digest=hashlib.sha256(path.read_bytes()).hexdigest()
            require(record.get('reviewed') is True and record.get('candidate_sha256')==digest,'candidate whole-source review hash missing/stale')
            adapter=json.loads(adapter_path.read_text())
            spec=importlib.util.spec_from_file_location('reviewed_candidate',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            model=getattr(module,adapter['factory'])(**adapter.get('constructor_kwargs',{})).eval()
        else: model=prepared_model.eval(); adapter=prepared_adapter
        require(adapter['forward_inputs']==['x','mask'] and adapter['forward_outputs']==['y'],'entry port adapter')
        # Semantic role bindings, shape, object and storage identity are all independently checked.
        # Imported lazily to avoid a dependency cycle when validator self-tests this function.
        named=dict(model.named_parameters(remove_duplicate=False)); pm=adapter['parameters']
        require(set(pm)==set(SHAPES) and set(pm.values())==set(named) and len(named)==12,'candidate state coverage')
        params={k:named[v] for k,v in pm.items()}
        require(len({id(v) for v in params.values()})==12 and len({v.untyped_storage().data_ptr() for v in params.values()})==12,'candidate Parameter alias')
        require(not list(model.named_buffers(remove_duplicate=False)),'candidate extra Buffer')
        for k,v in params.items(): require(tuple(v.shape)==SHAPES[k] and v.dtype==torch.float32 and v.requires_grad and v.device.type=='cpu','candidate Parameter metadata')
        doc=json.loads((root/'representation.yaml').read_text()); validate(doc)
        graph=fx.symbolic_trace(model)
        for case in cases:
            with torch.no_grad():
                for k,v in params.items(): v.copy_(case['parameters'][k])
            require(all(torch.equal(params[k],v) for k,v in case['parameters'].items()),'candidate state copy')
            expected=Reference().eval()
            with torch.no_grad():
                for k,v in expected.theta.items(): v.copy_(case['parameters'][k])
            x,mask=case['x'].clone(),case['mask'].clone(); before_x=x.clone(); before_mask=mask.clone()
            wanted,wtrace=expected.compute(x,mask)
            cap=Capture(graph); observed=cap.run(x,mask)
            if isinstance(observed,(tuple,list)) and len(observed)==1: observed=observed[0]
            require(same(observed,wanted,rtol,atol),'candidate final numerical mismatch')
            trace=runtime_structure(doc,cap,adapter['fx_nodes'],params,x,mask)
            require(all(same(trace[k],v,rtol,atol) for k,v in wtrace.items()),'candidate intermediate mismatch')
            require(torch.equal(x,before_x) and torch.equal(mask,before_mask),'candidate input mutation')
            require(all(torch.equal(params[k],v) for k,v in case['parameters'].items()),'candidate state mutation')
            result['evidence'].append({'case':case['id'],'final_and_18_intermediates':'PASS','verified_leaf_nodes':len(adapter['fx_nodes'])})
        result.update(status='PASS',schema_consistency='PASS',structural_fidelity='PASS',numerical_fidelity='PASS',resource_identity='PASS',
                      scope='Declared functional eval graph and fixture tests only; symbolic all-length equivalence not proven')
    except (ValueError,KeyError,TypeError,RuntimeError,AttributeError,torch.fx.proxy.TraceError) as exc:
        result['reason']=str(exc)
        # A prerequisite/tracing gap is BLOCKED. An executed numerical/structure violation is FAIL.
        if 'mismatch' in str(exc) or 'unmapped candidate' in str(exc) or 'candidate Parameter alias' in str(exc): result['status']='FAIL'
    return result
