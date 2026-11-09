"""Frozen CPU preflight and reviewed-candidate evaluation. No HTTP capability."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
ENV={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
     'NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
# Check process limits before importing the compute library.
if any(os.environ.get(k)!=v for k,v in ENV.items()):
    raise SystemExit('Run with frozen per-process thread environment; see fixtures/plan.json')
import torch
from torch.nn import functional as F
from reference import Reference,SHAPES
from graph_engine import validate,execute,TRACE_ROLES,require

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
PLAN=json.loads((ROOT/'fixtures/plan.json').read_text())
ROLES=json.loads((ROOT/'source/parameter_roles.json').read_text())
RTOL=PLAN['tolerances']['all_outputs']['rtol']; ATOL=PLAN['tolerances']['all_outputs']['atol']

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(relative,value):
    path=ROOT/relative; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def integrity():
    result=subprocess.run(['/opt/conda/bin/python','-B',str(ROOT.parent/'common/verify_setup.py'),'--worker','transformer'],
                          capture_output=True,text=True,timeout=30)
    require(result.returncode==0,'common/source integrity failed: '+result.stdout)
    return json.loads(result.stdout)
def close(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.allclose(a,b,rtol=RTOL,atol=ATOL,equal_nan=False)
def error(a,b):
    if a.shape!=b.shape: return {'shape_mismatch':True}
    finite=torch.isfinite(a)&torch.isfinite(b)
    da=(a-b).abs()[finite]; db=b.abs()[finite]
    same_nonfinite=torch.equal(torch.isnan(a),torch.isnan(b)) and torch.equal(torch.isinf(a),torch.isinf(b)) and torch.equal(torch.sign(a[~finite]),torch.sign(b[~finite]))
    return {'max_abs':float(da.max()) if da.numel() else 0.,
            'max_rel':float((da/db.clamp_min(1e-12)).max()) if da.numel() else 0.,
            'matching_nonfinite':same_nonfinite,'allclose':bool(close(a,b))}
def compare(traces,expected):
    require(set(traces)==set(expected),'trace role coverage')
    result={key:error(traces[key],expected[key]) for key in expected}
    require(all(v['allclose'] for v in result.values()),'trace mismatch '+str(result))
    return result

def check_state(model,mapping,expected=None):
    named=dict(model.named_parameters(remove_duplicate=False))
    require(set(mapping)==set(SHAPES),'state role coverage')
    require(set(mapping.values())==set(named) and len(mapping)==len(named),'extra/missing/aliased Parameter slot')
    values={role:named[path] for role,path in mapping.items()}
    require(len({id(p) for p in values.values()})==12,'Parameter object identity mismatch')
    require(len({p.untyped_storage().data_ptr() for p in values.values()})==12,'unexpected storage alias')
    require(not list(model.named_buffers(remove_duplicate=False)),'extra Buffer')
    for role,p in values.items():
        require(tuple(p.shape)==SHAPES[role] and p.dtype==torch.float32 and p.requires_grad and p.device.type=='cpu','state metadata '+role)
        if expected is not None: require(torch.equal(p,expected[role]),'state copied values '+role)
    return values

def align(model,mapping,values):
    actual=check_state(model,mapping)
    with torch.no_grad():
        for key,p in actual.items(): p.copy_(values[key])
    check_state(model,mapping,values)
    return actual

REF_MAP={k:'theta.'+k for k in SHAPES}

def fixtures():
    path=ROOT/'fixtures/frozen.pt'
    if path.exists(): return torch.load(path,map_location='cpu',weights_only=True)
    cases=[]
    for item in PLAN['cases']:
        g=torch.Generator(device='cpu').manual_seed(item['seed'])
        params={k:torch.randn(shape,generator=g,dtype=torch.float32)*0.25 for k,shape in SHAPES.items()}
        params['p4']=torch.linspace(0.6,1.4,8); params['p10']=torch.linspace(1.3,0.7,8)
        b,t=item['B'],item['T']
        x=torch.randn((b,t,8),generator=g)*0.8+torch.linspace(-0.4,0.5,8)
        mask=torch.zeros((t,t),dtype=torch.bool)
        if item['mask']=='causal': mask=torch.ones((t,t),dtype=torch.bool).triu(1)
        elif item['mask']=='asymmetric' and t>1:
            for i in range(t): mask[i,(i+1)%t]=True
        cases.append({'id':item['id'],'x':x,'mask':mask,'parameters':params})
    torch.save(cases,path)
    return cases

def source_model():
    return torch.nn.TransformerEncoderLayer(8,2,dim_feedforward=12,dropout=0.1,
          activation='relu',layer_norm_eps=1e-5,batch_first=True,norm_first=False,
          bias=True,device='cpu',dtype=torch.float32).eval()

def source_trace(model,x,mask):
    actual={}; handles=[]; path_counts={'sdpa':0,'fast_encoder':0}
    def pre(role):
        def hook(module,args): actual[role]=args[0].detach().clone()
        return hook
    def post(role):
        def hook(module,args,out): actual[role]=(out[0] if isinstance(out,tuple) else out).detach().clone()
        return hook
    for module,role in [(model.norm1,'r1'),(model.norm2,'r2')]: handles.append(module.register_forward_pre_hook(pre(role)))
    for module,role in [(model.norm1,'n1'),(model.norm2,'y'),(model.self_attn,'projected'),
                        (model.linear1,'ff1'),(model.linear2,'ff2'),(model.dropout,'activation')]:
        handles.append(module.register_forward_hook(post(role)))
    original_sdpa=F.scaled_dot_product_attention
    def observe_sdpa(*args,**kwargs):
        path_counts['sdpa']+=1
        return original_sdpa(*args,**kwargs)
    try:
        with torch.enable_grad(),patch.object(F,'scaled_dot_product_attention',observe_sdpa):
            y=model(x,src_mask=mask,is_causal=False)
    finally:
        for h in handles: h.remove()
    require(path_counts['sdpa']==1,'source slow SDPA path not observed')
    # Actual source explicit attention branch exposes per-head weights and logits.
    ip=F._in_projection_packed; badd=torch.baddbmm; b=x.shape[0]; t=x.shape[1]
    def observe_ip(*args,**kwargs):
        values=ip(*args,**kwargs)
        q,k,v=[a.transpose(0,1) for a in values]
        actual['packed']=torch.cat([q,k,v],dim=2)
        for role,val in zip(['q','k','v'],[q,k,v]): actual[role]=val.reshape(b,t,2,4).permute(0,2,1,3)
        return values
    def observe_badd(mask_arg,q_scaled,kt,*args,**kwargs):
        result=badd(mask_arg,q_scaled,kt,*args,**kwargs)
        actual['q_scaled']=q_scaled.reshape(b,2,t,4)
        actual['logits']=torch.bmm(q_scaled,kt).reshape(b,2,t,t)
        actual['masked_logits']=result.reshape(b,2,t,t)
        return result
    with torch.enable_grad(),patch.object(F,'_in_projection_packed',observe_ip),patch.object(torch,'baddbmm',observe_badd):
        projected,weights=model.self_attn(x,x,x,attn_mask=mask,need_weights=True,average_attn_weights=False,is_causal=False)
    require(close(projected,actual['projected']),'SDPA vs explicit-source projected mismatch')
    actual['weights']=weights
    actual['context']=torch.matmul(weights,actual['v'])
    actual['joined']=actual['context'].permute(0,2,1,3).reshape(b,t,8)
    # These two are observation calculations on source-returned weights, not extra source outputs.
    original_fast=torch._transformer_encoder_layer_fwd
    def observe_fast(*args,**kwargs):
        path_counts['fast_encoder']+=1
        return original_fast(*args,**kwargs)
    with torch.no_grad(),patch.object(torch,'_transformer_encoder_layer_fwd',observe_fast):
        fast=model(x,src_mask=mask,is_causal=False)
    require(path_counts['fast_encoder']==1,'source optimized encoder path not observed')
    return y,actual,fast,path_counts

def malformed(doc,name):
    d=copy.deepcopy(doc)
    if name=='unknown_operator': d['factors']['score_a']['operator']='attention'
    elif name=='unknown_field': d['custom_formula']='invented'
    elif name=='missing_producer': d['dataflow'].remove(['project_a.output','partition_a.input'])
    elif name=='duplicate_producer': d['dataflow'].append(['project_a.output','partition_a.input'])
    elif name=='cross_region_edge':
        d['dataflow'].remove(['unit.x','stage_a.x']); d['dataflow'].append(['unit.x','project_a.input'])
    elif name=='wrong_parameter_shape': d['parameters']['p0']['shape']=[8,24]
    elif name=='duplicate_membership': d['factors']['stage_b']['contains'].append('project_a')
    elif name=='cycle':
        d['dataflow'].remove(['unit.x','merge_a.main']); d['dataflow'].append(['normalize_a.output','merge_a.main'])
    elif name=='wrong_softmax_contract': d['factors']['prob_a']['attributes']['dim']=4
    elif name=='undeclared_symbol': d['interface']['shape_q']['outputs']['output']['shape'][0]='UNBOUND'
    return d

def candidate_selftest(doc,cases):
    from candidate_validation import Capture,evaluate
    from torch import fx
    good=Reference().eval(); first=cases[0]
    align(good,REF_MAP,first['parameters'])
    graph=fx.symbolic_trace(good); cap=Capture(graph); cap.run(first['x'],first['mask'])
    _,_,expected=execute(doc,good.theta,first['x'],first['mask'],return_values=True)
    hints={}
    for key in validate(doc)['order']:
        op=doc['factors'][key]['operator']
        if op=='constant': hints[key]={'literal':doc['factors'][key]['attributes']['value']}; continue
        ports=doc['interface'][key]['outputs']
        wanted=[expected[key+'.'+p] for p in ports]
        matches=[]
        for name,(node,args,kwargs) in cap.calls.items():
            if node.op not in {'call_function','call_method'}: continue
            target_name=getattr(node.target,'__name__',node.target)
            alias={'add':'add','mul':'mul','matmul':'matmul','relu_functional':'relu'}.get(op,op)
            if target_name!=alias: continue
            value=cap.values[name]; values=list(value) if isinstance(value,(tuple,list)) else [value]
            if len(values)==len(wanted) and all(isinstance(v,torch.Tensor) and close(v,w) for v,w in zip(values,wanted)):
                matches.append(name)
        require(len(matches)==1,'reference FX role ambiguous '+key+': '+str(matches))
        hints[key]=matches[0]
    adapter={'factory':'Reference','constructor_kwargs':{},'forward_inputs':['x','mask'],'forward_outputs':['y'],
             'parameters':REF_MAP,'fx_nodes':hints}
    save('fixtures/reference_adapter.json',adapter)
    result=evaluate(ROOT/'reference.py',ROOT,PLAN,cases,RTOL,ATOL,prepared_model=Reference(),prepared_adapter=adapter)
    require(result['status']=='PASS','reference rejected by candidate path: '+str(result))
    tampered=copy.deepcopy(adapter)
    tampered['fx_nodes']['normalize_a'],tampered['fx_nodes']['normalize_b']=tampered['fx_nodes']['normalize_b'],tampered['fx_nodes']['normalize_a']
    rejected=evaluate(ROOT/'reference.py',ROOT,PLAN,cases,RTOL,ATOL,prepared_model=Reference(),prepared_adapter=tampered)
    require(rejected['status']!='PASS','false semantic mapping accepted')
    wrong_state=copy.deepcopy(adapter)
    wrong_state['parameters']['p4'],wrong_state['parameters']['p10']=wrong_state['parameters']['p10'],wrong_state['parameters']['p4']
    state_rejected=evaluate(ROOT/'reference.py',ROOT,PLAN,cases,RTOL,ATOL,prepared_model=Reference(),prepared_adapter=wrong_state)
    require(state_rejected['status']!='PASS','false state mapping accepted')
    save('validation/candidate_interface.json',{'status':'PASS','reference_acceptance':result,
         'false_semantic_mapping':rejected,'false_state_mapping':state_rejected,
         'review_gate':'External candidate requires exact whole-source review hash before import; self-test uses reviewed source-authored Reference only'})

def preflight():
    start=time.monotonic(); setup=integrity(); doc=json.loads((ROOT/'representation.yaml').read_text())
    provenance=json.loads((ROOT/'source/provenance.json').read_text())
    require(torch.__version__=='2.2.1','runtime torch version')
    for item in provenance['files']:
        require(sha(Path(item['path']))==item['sha256']==sha(ROOT/item['snapshot']),'pinned source mismatch')
    validate(doc)
    cases=fixtures(); records=[]; nonvacuity=[]; state_evidence=[]
    for case in cases:
        x,mask,values=case['x'],case['mask'],case['parameters']
        ref=Reference().eval(); align(ref,REF_MAP,values)
        src=source_model(); align(src,ROLES,values)
        before=x.clone(); before_mask=mask.clone()
        yr,trace=ref.compute(x,mask)
        ys,observed,fast,paths=source_trace(src,x,mask)
        graph_y,graph_trace=execute(doc,ref.theta,x,mask)
        require(close(ys,yr) and close(fast,yr) and close(graph_y,yr),'final source/reference/graph mismatch')
        evidence=compare(observed,trace); graph_evidence=compare(graph_trace,trace)
        require(torch.equal(x,before) and torch.equal(mask,before_mask),'input mutation')
        check_state(ref,REF_MAP,values); check_state(src,ROLES,values)
        finite=bool(torch.isfinite(yr).all()); nonzero=bool((yr.abs()>0.01).any())
        require(finite and nonzero and bool((x>0).any()) and bool((x<0).any()),'vacuous fixture')
        require(not bool(mask.all(dim=-1).any()),'all-masked input row')
        require(not torch.equal(values['p0'][:8],values['p0'][8:16]),'vacuous packed QK rows')
        for role,p in values.items(): require(bool((p!=0).all()),'zero fixture parameter '+role)
        require(close(trace['weights'].sum(-1),torch.ones_like(trace['weights'].sum(-1))),'probability normalization')
        records.append({'case':case['id'],'source_final':error(ys,yr),'optimized_final':error(fast,yr),
                        'source_trace':evidence,'graph_trace':graph_evidence,'observed_source_paths':paths})
        nonvacuity.append({'case':case['id'],'finite_nonzero_output':True,'x_min':float(x.min()),'x_max':float(x.max()),
                          'y_std':float(yr.std()),'blocked_mask_entries':int(mask.sum()),'packed_qk_unequal':True})
        state_evidence.append({'case':case['id'],'parameter_count':12,'buffer_count':0,
                               'bijective_semantic_roles':True,'distinct_objects_and_storage':True,'exact_values_before_after':True})
    save('validation/source_reference.json',{'status':'PASS','tolerance':{'rtol':RTOL,'atol':ATOL},'cases':records,
          'observation_limits':['context/joined are derived from actual source-returned weights and projected through actual source out_proj; they are not source-returned intermediates',
                                'logits reconstructed via actual source baddbmm operands before mask; masked_logits from the actual source operation']})
    save('validation/nonvacuity.json',{'status':'PASS','cases':nonvacuity})
    save('validation/state_mapping.json',{'status':'PASS','mapping':ROLES,'cases':state_evidence,'initializer_fidelity':'NOT_RUN'})
    controls=json.loads((ROOT/'negative_tests/controls.json').read_text()); negatives=[]
    for name in controls['executable_model_mutations']:
        failures=[]; finite_witnesses=[]
        for case in cases:
            good=Reference().eval(); bad=Reference(name).eval()
            align(good,REF_MAP,case['parameters']); align(bad,REF_MAP,case['parameters'])
            yg,tg=good.compute(case['x'],case['mask']); yb,tb=bad.compute(case['x'],case['mask'])
            mismatched=[role for role in tg if not close(tb[role],tg[role])]
            if mismatched:
                item={'case':case['id'],'detector':'runtime intermediate/final numerical comparison','roles':mismatched,'final':error(yb,yg)}
                failures.append(item)
                if bool(torch.isfinite(yb).all()) and not close(yb,yg): finite_witnesses.append(case['id'])
        require(failures and finite_witnesses,'undetected or non-discriminating executable mutation '+name)
        negatives.append({'name':name,'kind':'executable_model','status':'REJECTED','finite_final_output_witnesses':finite_witnesses,'observed':failures})
    for name in controls['malformed_schema_mutations']:
        try: validate(malformed(doc,name))
        except ValueError as exc:
            negatives.append({'name':name,'kind':'malformed_schema','status':'REJECTED','detector':'static graph/schema','reason':str(exc)})
        else: raise ValueError('undetected malformed schema '+name)
    for name in controls['identity_mutations']:
        ref=Reference().eval()
        if name=='tied_norm_parameters': ref.theta['p10']=ref.theta['p4']
        elif name=='extra_parameter': ref.register_parameter('unrepresented',torch.nn.Parameter(torch.ones(1)))
        elif name=='extra_buffer': ref.register_buffer('unrepresented',torch.ones(1))
        elif name=='distinct_objects_shared_storage': ref.theta['p10']=torch.nn.Parameter(ref.theta['p4'].data)
        try: check_state(ref,REF_MAP)
        except ValueError as exc: negatives.append({'name':name,'kind':'identity','status':'REJECTED','detector':'bijective Parameter/Buffer and object/storage identity','reason':str(exc)})
        else: raise ValueError('undetected identity mutation '+name)
    save('negative_tests/results.json',{'status':'PASS','count':len(negatives),'controls':negatives})
    candidate_selftest(doc,cases)
    elapsed=time.monotonic()-start; require(elapsed<120,'preflight time budget')
    checks={k:'PASS' for k in ['source_pinned','source_reference_match','source_to_yaml_mapping','schema_consistency',
                              'reference_acceptance','negative_tests','state_mapping','fixture_nonvacuity']}
    evidence=['validation/source_reference.json','validation/nonvacuity.json','validation/state_mapping.json',
              'negative_tests/results.json','validation/schema.json','validation/candidate_interface.json','logs/scheduling.json']
    save('validation/schema.json',{'status':'PASS','scope':'submitted functional subset only',
         'factor_count':len(doc['factors']),'leaf_count':len(validate(doc)['leaves']),'edge_count':len(doc['dataflow']),
         'schema_delta_used':False,'source_mapping':'source_mapping.md','module_object_hierarchy':'OUTSIDE_DECLARED_SCOPE'})
    artifact_paths=[p for folder in ['source','fixtures','negative_tests'] for p in (ROOT/folder).rglob('*') if p.is_file()]
    artifact_paths += [ROOT/name for name in ['reference.py','validator.py','graph_engine.py','candidate_validation.py',
               'representation.yaml','schema_delta.md','source_mapping.md','prepare_artifacts.py','run_preflight.py']+evidence]
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(artifact_paths))}
    report={'status':'PASS','common_manifest_sha256':setup['common_manifest_sha256'],'artifacts':artifacts,
            'checks':checks,'schema_delta_used':False,'evidence':evidence,'duration_seconds':elapsed,
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),'fixture_count':len(cases),'negative_control_count':len(negatives),
            'scope':'Functional eval block; full source Module object/initializer/training fidelity excluded',
            'scheduling':'USER_RESERVED: transformer 1 slot; other compute at most 1; no enforced shared lock'}
    save('validation/preflight.json',report)
    print(json.dumps({k:report[k] for k in ['status','duration_seconds','fixture_count','negative_control_count','checks']},indent=2))
    return 0

def candidate(path,report):
    from candidate_validation import evaluate
    output=evaluate(path,ROOT,PLAN,fixtures(),RTOL,ATOL)
    report=Path(report).resolve(); require(report.is_relative_to(ROOT),'report outside worker boundary')
    report.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':output['status'],'report':str(report)}))
    return 0 if output['status']=='PASS' else 2

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true'); group.add_argument('--candidate',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.candidate:
        require(args.report is not None,'--candidate requires --report')
        return candidate(args.candidate,args.report)
    try: return preflight()
    except Exception as exc:
        save('validation/preflight.json',{'status':'FAIL','schema_delta_used':False,'reason':str(exc),
              'common_manifest_sha256':sha(ROOT.parent/'common/manifest.json'),'artifacts':{},'checks':{},
              'timestamp_utc':datetime.now(timezone.utc).isoformat()})
        raise

if __name__=='__main__': raise SystemExit(main())
