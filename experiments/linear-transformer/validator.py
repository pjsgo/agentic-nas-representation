"""Worker preflight. Candidate imports are forbidden while common schema is blocked.

--preflight executes static checks and, only with an explicit compute reservation,
source/reference and executable negative controls. No network or API sender.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml
from schema_checks import check, negatives

W=Path(__file__).resolve().parent
COMMON=W.parent/'common'
REQUIRED=['source_pinned','source_reference_match','source_to_yaml_mapping','schema_consistency','reference_acceptance','negative_tests','state_mapping','fixture_nonvacuity']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
    p=Path(p)
    if not p.resolve().is_relative_to(W):raise ValueError('Report outside owned directory')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def artifacts():
    selected=[]
    for p in W.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        rel=str(p.relative_to(W))
        if rel.startswith(('requests/','logs/','validation/')) or rel in ['status.json','result.md','README.md']:continue
        selected.append(p)
    return {str(p.relative_to(W)):sha(p) for p in sorted(selected)}


def numeric():
    # Import only after a global compute reservation has been affirmed externally.
    import torch
    torch.set_num_threads(1)
    from reference import IndependentCore,trace_reference
    from source_adapter import SourceCore
    from negative_tests.mutants import run as mutate
    plan=json.loads((W/'fixtures/plan.json').read_text())
    negative_plan=json.loads((W/'negative_tests/plan.json').read_text())
    metrics=[];nonvacuity=[];mutation_results=[];mapping=[];reference_acceptance=[]
    failures=[]
    def load(case):
        raw=json.loads((W/case['tensor_file']).read_text())
        return [torch.tensor(raw[n],dtype=torch.float32,device='cpu') for n in ['q','k','v','s','z']]
    def copies(xs):return [x.clone() for x in xs]
    def compare(a,b):
        if a.shape!=b.shape:return {'pass':False,'reason':'shape','left':list(a.shape),'right':list(b.shape)}
        diff=(a-b).abs(); finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        return {'pass':finite and bool(torch.allclose(a,b,rtol=1e-5,atol=1e-6)), 'max_abs':float(diff.max()),'max_rel':float((diff/b.abs().clamp_min(1e-30)).max()),'finite':finite}
    with torch.no_grad():
        for case in plan['cases']:
            xs=load(case);q,k,v,s,z=xs
            src=SourceCore().eval();ref=IndependentCore().eval()
            source_args=copies(xs);ref_args=copies(xs)
            actual=src(*source_args);expected,ref_history=trace_reference(*ref_args)
            # Parameter/Buffer mapping is checked by full enumeration, not assumed.
            src_parameters=list(src.named_parameters(remove_duplicate=False));ref_parameters=list(ref.named_parameters(remove_duplicate=False))
            src_buffers=list(src.named_buffers(remove_duplicate=False));ref_buffers=list(ref.named_buffers(remove_duplicate=False))
            tensor_mapping=(not src_parameters and not ref_parameters and not src_buffers and not ref_buffers)
            feature_shared=len(src.feature_calls)==2*q.shape[1] and len(set(src.feature_calls))==1
            identity=all(out[-2] is args[-2] and out[-1] is args[-1] for out,args in [(actual,source_args),(expected,ref_args)])
            mutation=all(not torch.equal(args[-2],s) and not torch.equal(args[-1],z) for args in [source_args,ref_args])
            qkv_unchanged=all(torch.equal(args[i],xs[i]) for args in [source_args,ref_args] for i in range(3))
            mapping.append({'case':case['name'],'status':'PASS' if tensor_mapping and feature_shared and identity and mutation and qkv_unchanged else 'FAIL','source_parameters':[n for n,_ in src_parameters],'reference_parameters':[n for n,_ in ref_parameters],'source_buffers':[n for n,_ in src_buffers],'reference_buffers':[n for n,_ in ref_buffers],'feature_module_same_for_q_k':feature_shared,'final_carries_are_input_objects':identity,'input_carries_mutated':mutation,'qkv_immutable':qkv_unchanged,'projection_sharing':'NOT_APPLICABLE: absent in core'})
            for port,a,b in zip(['y','s_final','z_final'],actual,expected):
                metrics.append({'case':case['name'],'port':port,**compare(a,b)})
            for t,(ah,bh) in enumerate(zip(src.history,ref_history)):
                for role in ['phi_q','phi_k','s','z','reciprocal']:metrics.append({'case':case['name'],'prefix':t+1,'port':role,**compare(ah[role],bh[role])})
            if len(src.history)!=q.shape[1]:failures.append('source trace length')
            # Independent repeated-prefix invocations, not only one final output.
            for length in range(1,q.shape[1]+1):
                prefix=[q[:,:length],k[:,:length],v[:,:length],s,z]
                source_prefix=SourceCore().eval()(*copies(prefix));reference_prefix=ref(*copies(prefix))
                for port,a,b in zip(['y','s_final','z_final'],source_prefix,reference_prefix):metrics.append({'case':case['name'],'prefix':length,'port':'fresh_'+port,**compare(a,b)})
            split=case['chunk_split']
            if split:
                for label,model in [('source',SourceCore().eval()),('reference',ref)]:
                    ss,zz=s.clone(),z.clone()
                    first=model(q[:,:split],k[:,:split],v[:,:split],ss,zz)
                    second=model(q[:,split:],k[:,split:],v[:,split:],ss,zz)
                    joined=(torch.cat([first[0],second[0]],1),second[1],second[2])
                    for port,a,b in zip(['y','s_final','z_final'],joined,expected):metrics.append({'case':case['name'],'port':label+'_chunk_'+port,**compare(a,b)})
            direct=ref(*copies(xs))
            checks=[compare(a,b) for a,b in zip(direct,expected)]
            reference_acceptance.append({'case':case['name'],'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','checks':checks})
            nonvacuity.append({'case':case['name'],'status':'PASS' if all(bool(torch.isfinite(x).all()) for x in actual) and float(expected[0].abs().max())>1e-5 and bool((q<0).any()) and bool((v<0).any()) and bool((s!=0).any()) and bool((z>0).all()) else 'FAIL','output_abs_max':float(expected[0].abs().max()),'output_variance':float(expected[0].var()),'q_shape':list(q.shape),'value_shape':list(v.shape),'denominator_sensitive':case['name']=='denominator'})
        for name in negative_plan['executable']:
            detections=[]
            for case in plan['cases']:
                xs=load(case)
                correct=IndependentCore().eval()(*copies(xs))
                wrong=mutate(name,*copies(xs))
                differences=[compare(a,b) for a,b in zip(wrong,correct)]
                if not all(d['pass'] for d in differences):detections.append({'case':case['name'],'observed':differences})
            mutation_results.append({'mutation':name,'status':'PASS' if detections else 'FAIL','detector':'y/S/Z comparison against independent recurrence at frozen tolerance','discriminating_cases':detections})
    for group in [mapping,reference_acceptance,nonvacuity,mutation_results]:
        if any(r['status']!='PASS' for r in group):failures.append('nonpassing check group')
    if any(not m['pass'] for m in metrics):failures.append('source/reference tensor mismatch')
    report={'status':'FAIL' if failures else 'PASS','source_reference_match':'FAIL' if any(not m['pass'] for m in metrics) else 'PASS','reference_acceptance':'PASS' if all(x['status']=='PASS' for x in reference_acceptance) else 'FAIL','state_mapping':'PASS' if all(x['status']=='PASS' for x in mapping) else 'FAIL','fixture_nonvacuity':'PASS' if all(x['status']=='PASS' for x in nonvacuity) else 'FAIL','negative_tests':'PASS' if all(x['status']=='PASS' for x in mutation_results) else 'FAIL','metrics':metrics,'reference_checks':reference_acceptance,'mapping':mapping,'nonvacuity':nonvacuity,'negative_controls':mutation_results,'failures':failures,'torch_version':torch.__version__,'torch_threads':torch.get_num_threads(),'device':'cpu','scope':'Source/reference inference core only. No generated model evaluated.'}
    write(W/'validation/numerical.json',report)
    return report


def preflight():
    start=time.monotonic()
    command=[sys.executable,'-B',str(COMMON/'verify_setup.py'),'--worker','linear_transformer']
    process=subprocess.run(command,capture_output=True,text=True,timeout=30)
    integrity={'command':command,'exit_code':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
    write(W/'logs/integrity.json',integrity)
    if process.returncode:raise RuntimeError('Frozen setup changed; refuse further phases')
    provenance=json.loads((W/'source/provenance.json').read_text())
    source_matches=all((W/r['local_path']).is_file() and sha(W/r['local_path'])==r['sha256'] for r in provenance['files'])
    local_matches=all(Path(r['path']).is_file() and sha(Path(r['path']))==r['sha256'] for r in provenance.get('operator_dependencies',[]))
    if not source_matches or not local_matches:raise RuntimeError('Pinned source drift')
    for p in [W/'reference.py',W/'source_adapter.py',W/'validator.py',W/'schema_checks.py',W/'negative_tests/mutants.py']:compile(p.read_text(),str(p),'exec')
    doc=yaml.safe_load((W/'representation.yaml').read_text());schema=check(doc);bad_schema=negatives(doc)
    write(W/'validation/schema.json',schema);write(W/'validation/schema_negatives.json',bad_schema)
    mapping=json.loads((W/'source/mapping.json').read_text())
    mapping_good=bool(mapping['facts']) and all(all(f in doc['factors'] for f in fact['factors']) for fact in mapping['facts']) and schema['source_mapping_status']=='PASS'
    reservation_file=W/'validation/compute_reservation.json'
    compute={'status':'BLOCKED','reason':'No externally confirmed global compute slot. Peer status snapshots are not a concurrency guarantee. Torch and source modules not imported.'}
    if reservation_file.is_file():
        reservation=json.loads(reservation_file.read_text())
        if reservation.get('confirmed') is True and reservation.get('max_other_compute_workers') in [0,1] and reservation.get('authority') in ['user','coordinator'] and reservation.get('evidence'):
            compute=numeric()
        else:compute={'status':'BLOCKED','reason':'Reservation is incomplete or exceeds global 2-worker ceiling'}
    write(W/'validation/compute.json',{k:v for k,v in compute.items() if k not in ['metrics','reference_checks','mapping','nonvacuity','negative_controls']})
    checks={'source_pinned':'PASS' if source_matches and local_matches else 'FAIL','source_reference_match':compute.get('source_reference_match','BLOCKED'),'source_to_yaml_mapping':('BLOCKED' if any(fact['status']=='BLOCKED' for fact in mapping['facts']) else 'PASS') if mapping_good else 'FAIL','schema_consistency':schema['status'],'reference_acceptance':compute.get('reference_acceptance','BLOCKED'),'negative_tests':('PASS' if all(x['status']=='PASS' for x in bad_schema) and compute.get('negative_tests')=='PASS' else ('FAIL' if any(x['status']=='FAIL' for x in bad_schema) or compute.get('negative_tests')=='FAIL' else 'BLOCKED')),'state_mapping':compute.get('state_mapping','BLOCKED'),'fixture_nonvacuity':compute.get('fixture_nonvacuity','BLOCKED')}
    status='PASS' if all(checks.get(k)=='PASS' for k in REQUIRED) else ('FAIL' if 'FAIL' in checks.values() else 'BLOCKED')
    evidence=['logs/integrity.json','source/provenance.json','source/mapping.json','validation/schema.json','validation/schema_negatives.json','validation/compute.json','validation/scheduling.json']
    if (W/'validation/numerical.json').is_file():evidence.append('validation/numerical.json')
    frozen=artifacts()
    for rel in evidence:frozen[rel]=sha(W/rel)
    if reservation_file.is_file():frozen['validation/compute_reservation.json']=sha(reservation_file)
    result={'status':status,'architecture':'linear_transformer','common_manifest_sha256':sha(COMMON/'manifest.json'),'artifacts':frozen,'checks':checks,'evidence':evidence,'schema_delta_used':True,'schema_delta_proposed':True,'schema_gaps':schema['gaps'],'compute':{k:v for k,v in compute.items() if k in ['status','reason','torch_version','torch_threads','device']},'duration_seconds':time.monotonic()-start,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'api_requests_made':0,'request_preparation':'BLOCKED until every mandatory check PASS; no prepare_request.py invocation','scope':'Preflight only; representation is a proposed graph with unapproved elu, not common-schema-compliant.'}
    write(W/'validation/preflight.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
    return 0 if status=='PASS' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true');parser.add_argument('--candidate',type=Path);parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.preflight:
        if args.candidate:parser.error('Choose preflight or candidate')
        return preflight()
    if args.candidate:
        if args.report is None:parser.error('--candidate requires --report')
        # No code import: this frozen common profile cannot express the source phi.
        result={'status':'BLOCKED','failure_categories':['SCHEMA_UNSUPPORTED'],'candidate':str(args.candidate),'candidate_sha256':sha(args.candidate) if args.candidate.is_file() else None,'candidate_imported':False,'reason':'Frozen profile lacks ELU. Candidate execution/evaluation must not bypass the failed schema gate. A coordinated new schema run and evaluator freeze are required.','numerical_fidelity':'NOT_RUN','structural_fidelity':'BLOCKED'}
        write(args.report,result);print(json.dumps(result,indent=2));return 2
    parser.error('Use --preflight or --candidate PATH --report PATH')


if __name__=='__main__':
    raise SystemExit(main())
