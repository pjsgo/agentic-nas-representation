"""Frozen worker validator. Preflight is offline; this file has no HTTP client."""
import argparse,ast,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'common'))
from verify_setup import verify_integrity

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    target=ROOT/p;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

def imports():
    global torch,Reference,Mutation,KINDS,schema_mutations,schema_check,execute,infer,traced,LinearRoles
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from reference import Reference
    from negative_tests.mutations import Mutation,KINDS,schema_mutations
    from schema_check import check as schema_check
    from yaml_executor import execute
    from runtime_trace import infer,traced,LinearRoles

def load_plan():return json.loads((ROOT/'fixtures/plan.json').read_text())
def fixtures(create=False):
    plan=load_plan();out=[]
    for i,case in enumerate(plan['cases']):
        p=ROOT/f'fixtures/case_{i}.pt'
        if create:
            gen=torch.Generator().manual_seed(case['seed'])
            def signed(shape,offset):
                a=torch.randn(shape,generator=gen)*.27+offset
                a[a.abs()<.005]=.037+offset
                return a
            data={'x':signed((case['B'],case['T'],4),.13),'h0':signed((case['B'],5),.21),'c0':signed((case['B'],5),-.19),'parameters':{k:signed(shape,offset) for k,shape,offset in [('p0',(20,4),.08),('p1',(20,5),-.07),('p2',(20,),.11),('p3',(20,),-.14)]}}
            torch.save(data,p)
        out.append(torch.load(p,map_location='cpu',weights_only=True))
    return out

def align(roles,data):
    assert set(roles)==set(data['parameters'])
    with torch.no_grad():
        for k,p in roles.items():
            v=data['parameters'][k];assert p.shape==v.shape and p.dtype==v.dtype;p.copy_(v)
            assert torch.equal(p,v)

def compare(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype,'shape/dtype mismatch'
    assert torch.isfinite(a).all() and torch.isfinite(b).all(),'nonfinite tensor'
    diff=(a-b).abs();max_abs=diff.max().item() if diff.numel() else 0.
    max_rel=(diff/b.abs().clamp_min(1e-12)).max().item() if diff.numel() else 0.
    assert torch.allclose(a,b,rtol=load_plan()['rtol'],atol=load_plan()['atol']),f'numerical mismatch max_abs={max_abs}'
    return {'max_abs':max_abs,'max_rel':max_rel,'elements':a.numel()}

def evaluate(model,data,rep,require_structure=True):
    # Shape-and-live-use inference, independent of exact Parameter/attribute names.
    first=next(c for c in data if c['x'].shape[1]>0);inputs=tuple(first[k] for k in ['x','h0','c0'])
    roles=infer(model,inputs);model.eval();cases=[]
    for fixture in data:
        inputs=tuple(fixture[k] for k in ['x','h0','c0']);align(roles,fixture)
        before=[v.clone() for v in inputs];params_before={k:v.clone() for k,v in roles.items()}
        result,tr,fp=traced(model,roles,inputs)
        semantic_observations={}
        expected,et,efp=traced(lambda *xs:execute(rep,roles,*xs,observed=semantic_observations),roles,inputs)
        errors=[compare(a,b) for a,b in zip(result,expected)]
        if require_structure:
            assert fp==efp,'runtime operator/dataflow/carry ancestry mismatch'
            # Compare all expected observable intermediates, not only final tensors.
            for node in et.values:
                assert node in tr.values,'missing intermediate tensor role'
                compare(tr.values[node],et.values[node])
            assert set(tr.records)==set(et.records),'extra/missing runtime operator'
        for a,b in zip(inputs,before):assert torch.equal(a,b),'external input/carry mutated'
        for k,p in roles.items():assert torch.equal(p,params_before[k]),'parameter mutation'
        # Reobserve actual live object uses on every step after alignment.
        obs=LinearRoles(model)
        with torch.no_grad(),obs:model(*inputs)
        if inputs[0].shape[1]:
            assert len(obs.calls)==2*inputs[0].shape[1]
            assert all(obs.roles[k] is roles[k] for k in roles),'sharing changed'
        else:
            assert result[1] is inputs[1] and result[2] is inputs[2],'empty carry identity'
        hierarchy={}
        for factor,invocations in semantic_observations.items():
            nodes=[[et.node(t) for t in outputs.values()] for outputs in invocations]
            assert all(node in tr.values or node in tr.nodes.values() for call in nodes for node in call),'semantic factor runtime mapping absent'
            hierarchy[factor]={'invocations':len(invocations),'verified_runtime_output_nodes':nodes}
        if inputs[0].shape[1]:assert set(hierarchy)==set(rep['factors']),'untraced hierarchy'
        cases.append({'semantic_hierarchy':hierarchy,'T':inputs[0].shape[1],'outputs':errors,'runtime_nodes':len(tr.records),'linear_calls':len(obs.calls),'graph_match':fp==efp,'parameter_identity_per_step':'PASS'})
    return {'status':'PASS','parameter_mapping':{k:{'shape':list(p.shape),'dtype':str(p.dtype),'requires_grad':p.requires_grad} for k,p in roles.items()},'buffers':0,'module_resources':0,'cases':cases}

def source_compare(data):
    r=Reference().eval();roles={k:p for k,p in zip(['p0','p1','p2','p3'],[r.weight_ih,r.weight_hh,r.bias_ih,r.bias_hh])}
    native=torch.nn.LSTMCell(4,5,bias=True,device='cpu',dtype=torch.float32).eval()
    nr=dict(zip(['p0','p1','p2','p3'],[native.weight_ih,native.weight_hh,native.bias_ih,native.bias_hh]))
    reports=[]
    for fixture in data:
        if fixture['x'].shape[1]==0:continue
        align(roles,fixture);align(nr,fixture);x,h,c=[fixture[k] for k in ['x','h0','c0']]
        rh,rc=h,c;ys=[];steps=[];ids=[id(p) for p in native.parameters()]
        with torch.no_grad():
            for t in range(x.shape[1]):
                h,c=native(x[:,t,:],(h,c));rh,rc=r.step(x[:,t,:],rh,rc)
                assert ids==[id(p) for p in native.parameters()]
                steps.append({'h':compare(rh,h),'c':compare(rc,c),'same_cell_module':True,'same_parameter_objects':True});ys.append(h)
            y,hn,cn=r(x,fixture['h0'],fixture['c0'])
            errors=[compare(y,torch.stack(ys,dim=1)),compare(hn,h),compare(cn,c)]
        reports.append({'T':x.shape[1],'steps':steps,'outputs':errors})
    return {'status':'PASS','source':'torch.nn.LSTMCell','cases':reports,'empty_source_sequence':'NOT_RUN: cell has no sequence API','native_gate_intermediates':'not exposed; source h/c checked each step'}

def nonvacuity(data):
    for f in data:
        for name in ['h0','c0']:
            assert (f[name]!=0).all() and f[name].std()>.01
        assert not torch.equal(f['h0'],f['c0'])
        assert not torch.equal(f['parameters']['p2'],f['parameters']['p3'])
        for p in f['parameters'].values():assert (p!=0).all() and p.min()<0 and p.max()>0
    assert {f['x'].shape[1] for f in data}=={0,1,3,5}
    return {'status':'PASS','nonzero_unequal_carries':True,'distinct_signed_parameters_and_biases':True,'B_T_I_H_distinct_case':[2,3,4,5]}

def negatives(data,rep):
    results=[]
    for kind in KINDS:
        model=Mutation(kind).eval()
        # Untied copies receive exact equal fixture values, so rejection proves identity, not random values.
        if kind=='untie':
            with torch.no_grad():
                for cell in model.copies:
                    for p,v in zip(cell.parameters(),data[0]['parameters'].values()):p.copy_(v)
        try:
            evaluate(model,data,rep)
        except (AssertionError,RuntimeError,ValueError) as e:
            detector=str(e)
        else:raise AssertionError('accepted negative control '+kind)
        numerical=[]
        # Independently show discriminating effects on aligned source-equation outputs.
        # untie is intentionally numerical-equivalent and must be rejected structurally.
        if kind=='untie':
            for fixture in data:
                r=Reference().eval()
                with torch.no_grad():
                    for cell in [model,*model.copies]:
                        for p,v in zip([cell.weight_ih,cell.weight_hh,cell.bias_ih,cell.bias_hh],fixture['parameters'].values()):p.copy_(v)
                    for p,v in zip(r.parameters(),fixture['parameters'].values()):p.copy_(v)
                    xs=tuple(fixture[k] for k in ['x','h0','c0'])
                    for a,b in zip(model(*xs),r(*xs)):compare(a,b)
                numerical.append(fixture['x'].shape[1])
        else:
            for fixture in data:
                if not fixture['x'].shape[1]:continue
                r=Reference().eval();nr=[r.weight_ih,r.weight_hh,r.bias_ih,r.bias_hh]
                with torch.no_grad():
                    for p,v in zip(model.parameters(),fixture['parameters'].values()):p.copy_(v)
                    for p,v in zip(nr,fixture['parameters'].values()):p.copy_(v)
                    xs=tuple(fixture[k] for k in ['x','h0','c0']);actual=model(*xs);expected=r(*xs)
                    differs=any(a.shape!=b.shape or not torch.allclose(a,b,rtol=1e-5,atol=1e-6) for a,b in zip(actual,expected))
                    if differs:numerical.append(fixture['x'].shape[1])
            assert numerical,'vacuous numerical negative '+kind
        results.append({'name':kind,'status':'PASS','expected':'REJECT','observed':'REJECT','detector':detector,'discriminating_T':[] if kind=='untie' else numerical,'numerically_equivalent_T':numerical if kind=='untie' else [],'nonvacuity':'executed equivalent outputs on all frozen cases with identical values; rejected extra Parameter object classes' if kind=='untie' else 'aligned executable outputs differ on recorded T'})
    schema_results=[]
    for name,mut in schema_mutations(rep).items():
        try:schema_check(mut)
        except (AssertionError,KeyError,ValueError) as e:reason=str(e) or type(e).__name__
        else:raise AssertionError('accepted malformed schema '+name)
        schema_results.append({'name':name,'status':'PASS','observed':'REJECT','detector':reason})
    return {'status':'PASS','model_negatives':results,'malformed_schema_negatives':schema_results}

def provenance_check():
    p=json.loads((ROOT/'source/provenance.json').read_text())
    assert torch.__version__.split('+')[0]==p['version']=='2.2.1'
    for f,h in p['files'].items():assert sha(f)==h,'source drift '+f
    assert sha(p['source_path'])==p['sha256']
    return {'status':'PASS','files':len(p['files']),'source_sha256':p['sha256']}

def preflight():
    start=time.monotonic();integrity=verify_integrity('lstm')
    imports();checks={};evidence={};error=None
    try:
        evidence['integrity']=integrity
        evidence['source_pin']=provenance_check();checks['source_pinned']='PASS'
        rep=json.loads((ROOT/'representation.yaml').read_text())
        evidence['schema']=schema_check(rep);checks['schema_consistency']='PASS'
        data=fixtures(create=True)
        evidence['fixture_tensor_bytes']=sum(t.numel()*t.element_size() for f in data for t in [f['x'],f['h0'],f['c0'],*f['parameters'].values()])
        assert evidence['fixture_tensor_bytes']<512*1024*1024
        evidence['nonvacuity']=nonvacuity(data);checks['fixture_nonvacuity']='PASS'
        evidence['source_reference']=source_compare(data);checks['source_reference_match']='PASS'
        evidence['reference_acceptance']=evaluate(Reference(),data,rep);checks['reference_acceptance']='PASS';checks['state_mapping']='PASS';checks['source_to_yaml_mapping']='PASS'
        evidence['negative_tests']=negatives(data,rep);checks['negative_tests']='PASS'
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);evidence['exception']=error;evidence['traceback']=traceback.format_exc()
    elapsed=time.monotonic()-start
    if elapsed>120:error='preflight budget exceeded'
    passed=not error and len(checks)==8
    evidence.update(status='PASS' if passed else 'FAIL',duration_seconds=elapsed,threads=torch.get_num_threads(),device='cpu',api_requests=0)
    write('validation/preflight_evidence.json',evidence)
    required=['source_pinned','source_reference_match','source_to_yaml_mapping','schema_consistency','reference_acceptance','negative_tests','state_mapping','fixture_nonvacuity']
    report={'status':'PASS' if passed else 'FAIL','common_manifest_sha256':sha(ROOT.parent/'common/manifest.json'),'checks':{k:checks.get(k,'FAIL' if error else 'NOT_RUN') for k in required},'schema_delta_used':False,'evidence':['validation/preflight_evidence.json','source_mapping.md','source/provenance.json','fixtures/plan.json','logs/compute_reservation.json'],'duration_seconds':elapsed,'error':error,'artifacts':{}}
    # Freeze every implementation/test/source/fixture artifact and the evidence, excluding evolving bookkeeping/request files.
    paths=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ['.py','.yaml'] or p.parts[-2] in ['source','fixtures','negative_tests'])]
    paths += [ROOT/p for p in ['source_mapping.md','schema_delta.md','validation/preflight_evidence.json','logs/compute_reservation.json']]
    report['artifacts']={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    write('validation/preflight.json',report)
    print(json.dumps({'status':report['status'],'checks':report['checks'],'duration_seconds':elapsed,'error':error},indent=2))
    return 0 if passed else 1

def candidate(path,report_path):
    verify_integrity('lstm');frozen=json.loads((ROOT/'validation/preflight.json').read_text());assert frozen['status']=='PASS'
    for rel,h in frozen['artifacts'].items():assert sha(ROOT/rel)==h,'frozen artifact drift '+rel
    path=Path(path).resolve();assert path.is_relative_to(ROOT),'candidate must be owned'
    # Manual source review record is a precondition, never inferred from candidate assertions.
    review=json.loads((ROOT/'validation/candidate_review.json').read_text())
    assert review['status']=='PASS' and review['candidate_sha256']==sha(path),'review entire untouched source before import'
    ast.parse(path.read_text())
    metadata=json.loads((ROOT/'requests/response_metadata.json').read_text())
    assert metadata['status']=='ok' and not metadata['assumptions'] and not metadata['missing_information']
    imports();spec=importlib.util.spec_from_file_location('reviewed_candidate',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ep=metadata['entrypoint'];factory=getattr(module,ep['factory']);model=factory(**ep['constructor_kwargs'])
    assert set(ep['forward_inputs'])=={'x','h0','c0'} and len(ep['forward_inputs'])==3
    assert set(ep['forward_outputs'])=={'y','hn','cn'} and len(ep['forward_outputs'])==3
    rep=json.loads((ROOT/'representation.yaml').read_text())
    assert set(rep['factors'])<=set(metadata['semantic_mapping']),'missing semantic hierarchy hints'
    class PortAdapter(torch.nn.Module):
        def __init__(self,inner):super().__init__();self.inner=inner
        def forward(self,x,h0,c0):
            values={'x':x,'h0':h0,'c0':c0}
            raw=self.inner(*(values[p] for p in ep['forward_inputs']))
            if isinstance(raw,dict):outputs=raw
            else:
                assert len(raw)==3
                outputs=dict(zip(ep['forward_outputs'],raw))
            return tuple(outputs[k] for k in ['y','hn','cn'])
    evidence=evaluate(PortAdapter(model),fixtures(),json.loads((ROOT/'representation.yaml').read_text()))
    write(report_path,{'status':'PASS','candidate_sha256':sha(path),'evidence':evidence,'scope':'frozen cell scan; initializer source defaults/training/grad/Agent Edit NOT_RUN'})
    return 0

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--preflight',action='store_true');group.add_argument('--candidate');parser.add_argument('--report',default='validation/results.json');args=parser.parse_args()
    if args.preflight:return preflight()
    target=Path(args.report)
    if target.is_absolute():target=target.resolve().relative_to(ROOT)
    assert '..' not in target.parts
    try:return candidate(args.candidate,str(target))
    except Exception as e:
        write(str(target),{'status':'FAIL','reason':type(e).__name__+': '+str(e)});return 1

if __name__=='__main__':raise SystemExit(main())
