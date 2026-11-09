"""Frozen restricted original-S4 reconstruction preflight and candidate evaluation.
All execution must be wrapped in the verified compute-v1.0 coordinator.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
COMMON=ROOT.parent/'common'
CHECKS=['source_pinned','source_reference_match','source_to_yaml_mapping','schema_consistency','reference_acceptance','negative_tests','state_mapping','fixture_nonvacuity']
AXES=['source_grounding','representable_scope','structural_fidelity','dataflow_attributes','resource_identity','state_recurrence_dependency','numerical_fidelity','schema_consistency','source_reference_match','validator_preflight']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def artifacts():
    files=[p for p in ROOT.glob('*.py') if p.name!='generated_model.py']
    files += [ROOT/p for p in ['source_mapping.md','schema_delta.md','representation.yaml']]
    for folder in ['source','fixtures','negative_tests']:files+=list((ROOT/folder).rglob('*'))
    files += [ROOT/'logs/concurrency.json']
    files += [ROOT/'validation'/name for name in ['numerical.json','negative_tests.json','schema.json','identity.json','nonvacuity.json','trace_baseline.json','source_reference_exploration.json'] if (ROOT/'validation'/name).is_file()]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files) if p.is_file()}

def integrity():
    command=[sys.executable,'-B',str(COMMON/'verify_setup.py'),'--worker','s4']
    run=subprocess.run(command,capture_output=True,text=True,timeout=30)
    obj=json.loads(run.stdout)
    if run.returncode or obj['status']!='PASS':raise ValueError('Immutable setup integrity failed')
    return obj

def preflight():
    import torch
    from reference import Reference,ROLES
    from evaluation import cases,IDS,identity_check,compare,differences
    from source.oracle_adapter import make_source,evaluate_source,ROLE_PATHS,SOURCE_SHA
    from schema_runtime import TensorProgram,load,validate_schema
    from negative_tests.models import MUTATIONS,make
    from trace_check import fingerprint,RenamedHost
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    report={'status':'FAIL','scope':'Original S4 rank-1 DPLR raw-parameter bilinear/impulse-recurrence core, Nfull=4,H=channels=1. Verified equivalent outputs to official naive FFT path; no FFT/cache/initializer/training claim.','checks':{k:'NOT_RUN' for k in CHECKS},'axes':{k:'NOT_RUN' for k in AXES},'common_manifest_sha256':sha(COMMON/'manifest.json'),'schema_delta_used':False,'api_requests_made':0,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'evidence':['validation/numerical.json','validation/negative_tests.json','validation/schema.json','validation/identity.json','validation/nonvacuity.json','source/provenance.json','source/factor_mapping.json','fixtures/plan.json','logs/local_execution.json'],'compute_policy':'compute-v1.0','torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'failure_categories':[]}
    try:
        report['integrity']=integrity()
        assert torch.__version__.split('+')[0]=='2.2.1','operator profile torch version mismatch'
        acquired=json.loads((ROOT/'source/acquisition.json').read_text())
        assert acquired['commit_sha']=='e757cef57d89e448c413de7325ed5601aceaac13'
        for name,item in acquired['files'].items():
            assert 'sha256' in item and sha(ROOT/'source/upstream'/name)==item['sha256'],'Source file hash mismatch'
        assert sha(ROOT/'source/upstream/models/s4/s4.py')==SOURCE_SHA
        operator_sources=json.loads((ROOT/'source/operator_sources.json').read_text())
        for item in operator_sources['installed_source_files']:
            assert sha(Path(item['path']))==item['sha256'],'Installed operator source changed'
        report['checks']['source_pinned']='PASS'
        doc=load(ROOT/'representation.yaml');schema=validate_schema(doc)
        write(ROOT/'validation/schema.json',schema)
        report['checks']['schema_consistency']='PASS'
        mapping=json.loads((ROOT/'source/factor_mapping.json').read_text())
        assert {x['factor'] for x in mapping['factors']}=={fid for fid,f in doc['factors'].items() if 'operator' in f}
        assert set(mapping['parameters'])==set(doc['parameters'])
        assert all(x['source_spans'][0]>0 and x['source_spans'][1]<=1951 for x in mapping['factors'])
        numerical=[];identities=[];nonvacuity=[];fixtures=[];trace_records=[]
        with torch.no_grad():
            for case,raw,u,h in cases():
                torch.manual_seed(case['seed'])
                before_u=u.clone();before_h=h.clone()
                reference=Reference(raw).eval()
                ref_mapping={role:role for role in ROLES}
                reference_id=identity_check(reference,ref_mapping,raw)
                expected=reference(u,h)
                source=evaluate_source(raw,u,h)
                source_id=identity_check(source['raw_model'],ROLE_PATHS,raw,excluded_buffers=('kernel.l_kernel',))
                direct_id=identity_check(make_source(raw),ROLE_PATHS,raw,excluded_buffers=('kernel.l_kernel',))
                program=TensorProgram(doc,{IDS[role]:value for role,value in raw.items()}).eval()
                program_map={role:'tensors.'+IDS[role] for role in ROLES}
                program_id=identity_check(program,program_map,raw)
                got=program(u,h)
                assert torch.equal(u,before_u) and torch.equal(h,before_h),'input/state mutation'
                errors={'yaml':compare(got,expected),'official_dense':compare((source['kernel'],source['y'],source['h_final']),expected),'official_naive_fft':compare((source['kernel'],source['fft_y'],source['fft_h']),expected),'official_linear_kernel':compare((source['linear_kernel'],source['y'],source['h_final']),expected)}
                da,db,_=reference.discrete()
                torch.testing.assert_close(da,source['dA'],rtol=1e-9,atol=1e-10)
                torch.testing.assert_close(db,source['dB'],rtol=1e-9,atol=1e-10)
                intermediate={}
                for names,role in [(('dr','di'),'D'),(('er','ei'),'E'),(('pr','pi'),'P'),(('qr','qi'),'Q'),(('rr','ri'),'R'),(('br','bi'),'B')]:
                    actual=torch.complex(program.trace['prepare.'+names[0]][0],program.trace['prepare.'+names[1]][0])
                    target=source['raw_model'].kernel.step_params[role]
                    if target.ndim==3:target=target.squeeze(0)
                    torch.testing.assert_close(actual,target,rtol=1e-9,atol=1e-10)
                    intermediate[role]=float((actual-target).abs().max())
                assert all(ids=={id(program.tensors[pid])} for pid,ids in program.identity_reads.items()) and len(program.identity_reads)==7
                assert len(program.scan_history['sequence_scan'])==u.shape[-1]
                assert len(program.scan_history['kernel_scan'])==u.shape[-1]
                assert all(torch.count_nonzero(x['kernel_scan_pulse'])==0 for x in program.scan_history['kernel_scan'])
                identities.append({'case':case['id'],'source':source_id,'fresh_source':direct_id,'reference':reference_id,'yaml':program_id,'input_and_external_state_unchanged':True,'parameter_identity_reads':{pid:len(ids) for pid,ids in program.identity_reads.items()}})
                numerical.append({'case':case['id'],'comparisons':errors,'prepared_intermediate_max_abs':intermediate,'discrete_A_max_abs':float((source['dA']-da).abs().max()),'discrete_B_max_abs':float((source['dB']-db).abs().max())})
                assert all(torch.isfinite(v).all() and torch.count_nonzero(v)>0 for v in expected)
                sensitivities={}
                for role in ROLES:
                    perturbed={k:v.clone() for k,v in raw.items()};perturbed[role]+=0.13
                    changed=Reference(perturbed).eval()(u,h)
                    err=differences(changed,expected)
                    assert max(err.values())>1e-8,'vacuous Parameter role '+role
                    sensitivities[role]=err
                    if role in ['C','D']:
                        assert torch.equal(changed[2],expected[2]),'output-only parameter affects carry'
                    if role=='D':assert torch.equal(changed[0],expected[0]),'feedthrough enters kernel'
                alternate=reference(u,h+0.17)
                altered_signal=program(u+0.17,h)
                assert torch.equal(altered_signal[0],got[0]),'kernel depends on input signal values'
                assert torch.equal(alternate[0],expected[0]),'kernel depends on h0'
                nonvacuity.append({'case':case['id'],'input_shape':list(u.shape),'h0_nonzero':bool(torch.count_nonzero(h)),'input_positive_and_negative':bool((u>0).any() and (u<0).any()),'output_max_abs':[float(x.abs().max()) for x in expected],'parameter_perturbation_effects':sensitivities,'kernel_independent_of_h0':True,'kernel_independent_of_input_values':True,'h0_perturbation':differences(alternate,expected)})
                # Executed ATen graph fingerprints freeze a structural baseline independently of names.
                f1=fingerprint(program,program_map,u,h)
                renamed=RenamedHost(TensorProgram(doc,{IDS[role]:value for role,value in raw.items()})).eval()
                f2=fingerprint(renamed,{role:'free_scope.tensors.'+IDS[role] for role in ROLES},u,h)
                assert f1==f2,'runtime DAG signature unstable'
                trace_records.append({'case':case['id'],**f1})
                fixtures.append((case,raw,u,h,expected))
            assert len({u.shape[-1] for _,_,u,_,_ in fixtures})>1
            assert len({float(raw['inv_dt'].exp().item()) for _,raw,_,_,_ in fixtures})>1
            assert any(u.shape[0]!=u.shape[1] and u.shape[1]!=u.shape[2] for _,_,u,_,_ in fixtures)
            negatives=[]
            for mutation in MUTATIONS:
                record={'mutation':mutation,'kind':'executable_model','cases':[],'detected':False}
                for case,raw,u,h,expected in fixtures:
                    mutant=make(raw,mutation).eval();detectors=[]
                    try:identity_check(mutant,{role:role for role in ROLES},raw)
                    except ValueError as exc:detectors.append('state_mapping: '+str(exc))
                    got=mutant(u,h);delta=differences(got,expected)
                    try:compare(got,expected)
                    except (AssertionError,ValueError):detectors.append('numerical')
                    if mutation=='derived_trainable':assert max(delta.values())==0.0,'identity control must remain numerically identical'
                    record['cases'].append({'case':case['id'],'detectors':detectors,'max_abs_errors':delta})
                    record['detected']|=bool(detectors)
                assert record['detected'],'undetected mutation: '+mutation
                negatives.append(record)
            for mutation in ['swap_resolvent_components','replace_step_exp','reverse_scan']:
                altered=deepcopy(doc)
                if mutation=='swap_resolvent_components':
                    m=altered['factors']['sequence_scan']['scan']['invariant_inputs'];m['dr'],m['di']=m['di'],m['dr']
                elif mutation=='replace_step_exp':
                    key=next(x['factor'] for x in mapping['factors'] if x['role']=='parameter_transform: positive step size');altered['factors'][key]['operator']='neg'
                else:altered['factors']['sequence_scan']['scan']['direction']='reverse'
                validate_schema(altered)
                detected=[]
                for case,raw,u,h,expected in fixtures:
                    mutant=TensorProgram(altered,{IDS[role]:value for role,value in raw.items()}).eval();got=mutant(u,h)
                    try:compare(got,expected)
                    except (AssertionError,ValueError):detected.append({'case':case['id'],'max_abs_errors':differences(got,expected)})
                assert detected,'undetected graph mutation: '+mutation
                negatives.append({'mutation':mutation,'kind':'well_formed_executable_graph','detected':True,'detector':'numerical','cases':detected})
            case,raw,u,h,expected=fixtures[0]
            aliased=TensorProgram(doc,{IDS[role]:value for role,value in raw.items()})
            aliased.tensors['p4']=aliased.tensors['p3']
            try:identity_check(aliased,{role:'tensors.'+IDS[role] for role in ROLES},raw)
            except ValueError as exc:negatives.append({'mutation':'raw_parameter_identity_alias','kind':'identity','detected':True,'detector':str(exc)})
            else:raise AssertionError('undeclared sharing not detected')
            malformed=[]
            for mutation in ['unsupported_fft','missing_producer','duplicate_producer','broken_scan_cover','unapproved_field']:
                altered=deepcopy(doc)
                if mutation=='unsupported_fft':
                    key=next(fid for fid,v in altered['factors'].items() if 'operator' in v);altered['factors'][key]['operator']='fft'
                elif mutation=='missing_producer':altered['dataflow'].pop()
                elif mutation=='duplicate_producer':altered['dataflow'].append(altered['dataflow'][0])
                elif mutation=='broken_scan_cover':altered['factors']['sequence_scan']['scan']['invariant_inputs'].pop('dr')
                else:altered['custom_discretization']='opaque'
                try:validate_schema(altered)
                except ValueError as exc:malformed.append({'mutation':mutation,'detected':True,'reason':str(exc)})
                else:raise AssertionError('malformed schema not rejected: '+mutation)
            # Correct executable program must also match its declared source-role DAG.
            baseline=TensorProgram(doc,{IDS[role]:value for role,value in raw.items()})
            correct_sig=fingerprint(baseline,{role:'tensors.'+IDS[role] for role in ROLES},u,h)
            altered=deepcopy(doc);altered['factors']['sequence_scan']['scan']['direction']='reverse'
            reversed_program=TensorProgram(altered,{IDS[role]:value for role,value in raw.items()})
            bad_sig=fingerprint(reversed_program,{role:'tensors.'+IDS[role] for role in ROLES},u,h)
            assert correct_sig['sha256']!=bad_sig['sha256'],'dataflow trace detector is vacuous'
        write(ROOT/'validation/numerical.json',{'status':'PASS','rtol':1e-9,'atol':1e-10,'cases':numerical})
        write(ROOT/'validation/identity.json',{'status':'PASS','cases':identities})
        write(ROOT/'validation/nonvacuity.json',{'status':'PASS','cases':nonvacuity})
        write(ROOT/'validation/negative_tests.json',{'status':'PASS','executable_and_identity':negatives,'malformed_schema':malformed,'trace_mutation_detected':True,'source_or_reference_derived_from_generated_code':False})
        write(ROOT/'validation/trace_baseline.json',{'status':'PASS','cases':trace_records,'policy':'Conservative ATen DAG equality normalized by semantic Parameter role and external input order; free Python names ignored.'})
        report['checks']={k:'PASS' for k in CHECKS}
        report['axes']={k:'PASS' for k in AXES}
        report['status']='PASS'
        report['limits']={'source_grounding':'Audited unmodified AST definitions with tested explicit layout-only adapter; full third-party imports not executed.','schema_scope':'Only the chosen rank-1 recurrent/impulse-kernel core; no original FFT/cache algorithm structure claim.','initializer_fidelity':'NOT_RUN','training_gradients':'NOT_RUN','generated_reconstruction':'NOT_RUN','agent_edit':'NOT_RUN'}
    except Exception as exc:
        report['error']={'type':type(exc).__name__,'message':str(exc)[:3000]}
        report['failure_categories']=['PREFLIGHT_FAILED']
        report['axes']={k:'BLOCKED' for k in AXES}
    report['artifacts']=artifacts()
    return report

def candidate(path):
    # Check review/gates BEFORE any candidate import or torch execution.
    pre=json.loads((ROOT/'validation/preflight.json').read_text())
    if pre['status']!='PASS':return {'status':'BLOCKED','reason':'Full preflight required','candidate_imported':False}
    for name,digest in pre['artifacts'].items():
        if sha(ROOT/name)!=digest:return {'status':'BLOCKED','reason':'Frozen artifact drift: '+name,'candidate_imported':False}
    review_path=ROOT/'validation/candidate_review.json'
    if not review_path.exists():return {'status':'BLOCKED','reason':'Whole-source candidate review and pinned hash required','candidate_imported':False}
    reviewed=json.loads(review_path.read_text())
    path=Path(path).resolve()
    if not reviewed.get('full_source_review') or reviewed.get('candidate_sha256')!=sha(path):return {'status':'BLOCKED','reason':'Candidate review hash mismatch','candidate_imported':False}
    if path in [ROOT/'reference.py',ROOT/'schema_runtime.py']:return {'status':'BLOCKED','reason':'Independent oracle cannot be submitted as a reconstruction','candidate_imported':False}
    tree=ast.parse(path.read_text(),filename=str(path))
    forbidden={'open','exec','eval','compile','__import__','getattr','setattr','globals','locals','breakpoint','input'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(m.split('.')[0] not in ['torch','math','typing','collections'] for m in modules):return {'status':'BLOCKED','reason':'Unreviewed candidate import','candidate_imported':False}
        if isinstance(node,ast.Name) and node.id in forbidden:return {'status':'BLOCKED','reason':'Forbidden dynamic/filesystem builtin','candidate_imported':False}
    metadata=json.loads(path.with_suffix('.entrypoint.json').read_text())
    entry=metadata['entrypoint'];semantics=metadata['semantic_mapping']
    if entry['forward_inputs']!=['u','h0'] or entry['forward_outputs']!=['kernel','y','h_final']:return {'status':'FAIL','failure_category':'STRUCTURE_MISMATCH','candidate_imported':False}
    from evaluation import cases,ROLES,IDS,aligned_copy,compare
    from reference import Reference
    from schema_runtime import TensorProgram,load
    from trace_check import fingerprint
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('reviewed_candidate',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mapping={role:semantics['parameters.'+IDS[role]] for role in ROLES}
    records=[];structural=True
    with torch.no_grad():
        for case,raw,u,h in cases():
            model=getattr(module,entry['factory'])(**entry['constructor_kwargs']).eval()
            aligned_copy(model,mapping,raw)
            before_u=u.clone();before_h=h.clone()
            expected=Reference(raw).eval()(u,h);actual=model(u,h)
            errors=compare(actual,expected)
            if not torch.equal(before_u,u) or not torch.equal(before_h,h):raise ValueError('candidate mutates input/external state')
            expected_program=TensorProgram(load(ROOT/'representation.yaml'),{IDS[role]:value for role,value in raw.items()})
            a=fingerprint(model,mapping,u,h);b=fingerprint(expected_program,{role:'tensors.'+IDS[role] for role in ROLES},u,h)
            match=a['sha256']==b['sha256'];structural &= match
            records.append({'case':case['id'],'numerical':errors,'runtime_dataflow_match':match,'candidate_graph':a})
    return {'status':'PASS' if structural else 'BLOCKED','candidate_imported':True,'candidate_sha256':sha(path),'numerical_fidelity':'PASS','structural_fidelity':'PASS' if structural else 'BLOCKED','reason':None if structural else 'Trace differs: semantic review required, numerical agreement alone is insufficient. No candidate repaired.','cases':records}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--preflight',action='store_true');group.add_argument('--candidate')
    parser.add_argument('--report');args=parser.parse_args()
    if args.candidate and not args.report:parser.error('--candidate requires --report')
    output=Path(args.report).absolute() if args.report else ROOT/'validation/preflight.json'
    if not output.resolve().is_relative_to((ROOT/'validation').resolve()) or output.is_symlink() or output.suffix!='.json':parser.error('Report must be JSON inside worker validation directory')
    if args.candidate and output==ROOT/'validation/preflight.json':parser.error('Candidate reports must not overwrite frozen preflight')
    start=time.monotonic()
    if os.environ.get('AGENTIC_COMPUTE_POLICY')!='compute-v1.0':
        report={'status':'BLOCKED','reason':'Verified coordinator invocation required; no torch/source/candidate imported','candidate_imported':False,'api_requests_made':0}
    else:
        try:report=preflight() if args.preflight else candidate(args.candidate)
        except Exception as exc:report={'status':'FAIL','error':{'type':type(exc).__name__,'message':str(exc)[:3000]},'api_requests_made':0}
    report['elapsed_seconds']=round(time.monotonic()-start,6)
    write(output,report)
    print(json.dumps({'status':report['status'],'report':str(output),'error':report.get('error'),'elapsed_seconds':report['elapsed_seconds']}))
    return {'PASS':0,'BLOCKED':2}.get(report['status'],1)
if __name__=='__main__':raise SystemExit(main())
