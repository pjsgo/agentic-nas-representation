"""Source / independent-reference / representation preflight and frozen candidate evaluator."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
import yaml
from reference import Reference, ENTRYPOINT, SEMANTIC_MAPPING, OUTPUTS, SHAPES
from representation_runtime import Runtime, Trace, validate
import source_oracle
from negative_tests.mutations import Mutant, NUMERICAL, IDENTITY, SCHEMA, mutate_schema

HERE=Path(__file__).resolve().parent
WORKER=HERE.parent.parent
RUN=WORKER.parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(HERE):raise ValueError('Report must stay in revision')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        # Never discard a previous result, even during local implementation checks.
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        copy=HERE/'history'/('superseded_'+stamp+'_'+path.name)
        shutil.copyfile(path,copy)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def need(test,message):
    if not test:raise ValueError(message)


def integrity():
    cmd=['/opt/conda/bin/python','-B',str(RUN/'common/verify_setup.py'),'--worker','seq2seq']
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    need(result.returncode==0,'Common integrity: '+result.stdout)
    parsed=json.loads(result.stdout)
    need(parsed['status']=='PASS','Common integrity')
    return parsed


def preserve_check():
    prior=json.loads((HERE/'history/prior_worker_manifest.json').read_text())
    changed=[name for name,digest in prior.items() if sha(WORKER/name)!=digest]
    need(not changed,'Prior worker artifacts changed: '+repr(changed))
    return {'status':'PASS','existing_files_checked':len(prior),'changed':changed}


def source_pin():
    p=json.loads((HERE/'source/provenance.json').read_text())
    need(p['commit_sha']==json.loads((HERE/'source/commit.json').read_text())['sha'],'Commit pin')
    for filename,item in p['files'].items():need(sha(HERE/'source'/filename)==item['sha256'],'Downloaded source hash: '+filename)
    need(sha(HERE/'source/extracted_classes.py')==p['extracted_classes_sha256'],'Extracted source hash')
    for item in p['operator_sources'].values():need(sha(Path(item['path']))==item['sha256'],'Operator source drift')
    need(p['source_review']['reviewed'] is True,'Source review')
    return {'status':'PASS','commit_sha':p['commit_sha'],'notebook_sha256':p['files']['notebook.ipynb']['sha256']}


def fixtures(plan):
    result=[]
    for case in plan['cases']:
        path=HERE/'fixtures'/('seed_'+str(case['seed'])+'.pt')
        if not path.exists():
            generator=torch.Generator(device='cpu').manual_seed(case['seed'])
            def values(shape):
                n=1
                for dim in shape:n*=dim
                return (torch.linspace(-0.45,0.45,n)+0.05*(torch.rand(n,generator=generator)-0.5)).reshape(shape)
            s,t=case['source_length'],case['decoder_steps']
            target=torch.arange((t+1)*2,dtype=torch.long).reshape(t+1,2)%7
            target[0]=0
            obj={**case,'parameters':{key:values(shape) for key,shape in SHAPES.items()},
                 'source_embedding_weight':values((9,3)),'target_embedding_weight':values((7,5)),
                 'source_tokens':torch.arange(s*2,dtype=torch.long).reshape(s,2)%9,
                 'target_tokens':target,'nonzero_h0':values((2,4))+0.17,
                 'nonzero_c0':values((2,4))-0.23}
            torch.save(obj,path)
        result.append(torch.load(path,map_location='cpu',weights_only=True))
    return result


def state_map(model,mapping,doc,fixture=None,copy_values=True):
    expected=doc['parameters'];mapped={}
    for pid,spec in expected.items():
        role='parameters.'+pid
        need(role in mapping,'Missing Parameter role '+role)
        need(isinstance(mapping[role],str),'Parameter location must be a dotted object path')
        tensor=source_oracle.resolve(model,mapping[role])
        need(isinstance(tensor,torch.nn.Parameter),'Role is not a Parameter: '+pid)
        need(list(tensor.shape)==spec['shape'] and tensor.dtype==torch.float32 and tensor.device.type=='cpu' and tensor.requires_grad==spec['requires_grad'],'Parameter dtype/shape/domain: '+pid)
        mapped[pid]=tensor
    need(len({id(p) for p in mapped.values()})==len(expected),'Accidental Parameter object sharing')
    need(len({p.untyped_storage().data_ptr() for p in mapped.values()})==len(expected),'Accidental storage alias')
    actual={id(p) for p in model.parameters()}
    need(actual=={id(p) for p in mapped.values()},'Extra/missing Parameter tensors')
    need(not list(model.buffers()),'Unexpected Buffer tensor')
    if fixture is not None:
        if copy_values:
            with torch.no_grad():
                for pid,p in mapped.items():p.copy_(fixture['parameters'][pid])
        need(all(torch.equal(p,fixture['parameters'][pid]) for pid,p in mapped.items()),'State values not exactly aligned')
    return mapped


def compare(expected,actual,labels=OUTPUTS):
    need(isinstance(actual,(tuple,list)) and len(actual)==len(expected),'Output tuple contract')
    records={}
    for role,a,b in zip(labels,expected,actual):
        need(isinstance(b,torch.Tensor) and a.shape==b.shape and b.dtype==a.dtype and b.device.type=='cpu','Output shape/dtype: '+role)
        need(bool(torch.isfinite(b).all()),'Nonfinite output: '+role)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        difference=(a-b).abs()
        records[role]={'max_absolute_error':float(difference.max()),
                       'max_relative_error':float((difference/a.abs().clamp_min(1e-12)).max())}
    return records


def traced(model,weights,inputs):
    mode=Trace(dict(zip(['source','target_prev'],inputs)),weights)
    with mode:output=model(*inputs)
    return output,mode.outputs(output),mode


def evaluate_model(model,mapping,doc,fixture,inputs,expected,expected_graph):
    model.eval();weights=state_map(model,mapping,doc,fixture)
    before=[v.clone() for v in inputs]
    output,graph,trace=traced(model,weights,inputs)
    numeric=compare(expected,output)
    need(graph==expected_graph,'Runtime operator/dataflow/attributes/scan DAG mismatch')
    need(output[2] is output[4] and output[3] is output[5], 'Identity state handoff must preserve Tensor objects')
    need(all(torch.equal(a,b) for a,b in zip(inputs,before)),'Unexpected input mutation')
    state_map(model,mapping,doc,fixture,copy_values=False)
    return {'status':'PASS','numerical':numeric,'runtime_dag_nodes':len(trace.nodes),
            'output_dag_sha256':graph,'parameter_count':len(weights),'buffers':0,
            'identity_handoff':True,'input_and_parameter_mutation':False}


def schema_negatives(doc,cases):
    records=[]
    for name in SCHEMA:
        mutated=mutate_schema(doc,name)
        try:validate(mutated)
        except (ValueError,KeyError,IndexError,TypeError) as exc:
            need(name!='swapped_scan_carry','Well-formed carry mutation must reach execution')
            records.append({'name':name,'status':'PASS','detector':'schema','observed_failure':str(exc)})
            continue
        if name!='swapped_scan_carry':raise ValueError('Malformed schema unexpectedly accepted: '+name)
        failures=[]
        for fixture,inputs,expected,graph in cases:
            weights=fixture['parameters'];runtime=Runtime(mutated,weights)
            def run(*args):return tuple(runtime.run(dict(zip(['source','target_prev'],args))).values())
            output,actual_graph,_=traced(run,weights,inputs)
            max_error=max(float((a-b).abs().max()) for a,b in zip(expected,output))
            if actual_graph!=graph and max_error>1e-6:failures.append({'seed':fixture['seed'],'max_absolute_error':max_error})
        need(bool(failures),'Carry mutation not discriminated')
        records.append({'name':name,'status':'PASS','detector':'runtime_DAG_and_numerical',
                        'observed_failure':'Simultaneous carry outputs swapped','discriminating_cases':failures})
    return records


def negatives(doc,cases):
    records=[]
    for name in NUMERICAL+IDENTITY:
        failures=[]
        for fixture,inputs,expected,graph in cases:
            model=Mutant(name)
            try:evaluate_model(model,SEMANTIC_MAPPING,doc,fixture,inputs,expected,graph)
            except (ValueError,AssertionError,RuntimeError) as exc:
                entry={'seed':fixture['seed'],'observed_failure':str(exc)[:1200]}
                if name in NUMERICAL:
                    # Ensure a numerical mutation is executable and has real witness
                    # differences; a mapping/exception-only rejection is insufficient.
                    weights=state_map(model,SEMANTIC_MAPPING,doc,fixture)
                    result=model(*inputs)
                    mismatches=[]
                    for role,a,b in zip(OUTPUTS,expected,result):
                        if a.shape!=b.shape:mismatches.append({'role':role,'shape_expected':list(a.shape),'shape_observed':list(b.shape)})
                        elif not torch.allclose(a,b,rtol=1e-5,atol=1e-6):
                            mismatches.append({'role':role,'max_absolute_error':float((a-b).abs().max())})
                    need(bool(mismatches),'Numerical mutation lacks a discriminating output: '+name)
                    entry['numerical_witnesses']=mismatches
                failures.append(entry)
        need(bool(failures),'Mutation accepted on every fixture: '+name)
        records.append({'name':name,'status':'PASS','detector':'identity' if name in IDENTITY else 'numerical_and_runtime',
                        'discriminating_cases':failures})
    return records


def preflight():
    began=time.monotonic();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir=HERE/'validation'/('run_'+stamp);run_dir.mkdir()
    checks={key:'NOT_RUN' for key in ['source_pinned','source_reference_match','source_to_yaml_mapping',
        'schema_consistency','reference_acceptance','negative_tests','state_mapping','fixture_nonvacuity']}
    evidence={};status='FAIL'
    try:
        need(os.environ.get('AGENTIC_COMPUTE_POLICY')=='compute-v1.0','Run through verified Compute Coordinator')
        need(torch.get_num_threads()==1 and len(os.sched_getaffinity(0))==1,'Compute thread/affinity limits')
        common=integrity();evidence['integrity_before']=common
        evidence['prior_preservation']=preserve_check()
        evidence['source']=source_pin();checks['source_pinned']='PASS'
        doc=yaml.safe_load((HERE/'representation.yaml').read_text())
        evidence['schema']=validate(doc);checks['schema_consistency']='PASS'
        mapping=json.loads((HERE/'source/mapping.json').read_text())
        need(set(mapping['factor_source'])==set(doc['factors']),'Missing factor-to-source mapping')
        need(set(mapping['parameter_source'])==set(doc['parameters']),'Missing Parameter-to-source mapping')
        checks['source_to_yaml_mapping']='PASS'
        plan=json.loads((HERE/'fixtures/plan.json').read_text());frozen_fixtures=fixtures(plan)
        need(len(frozen_fixtures)==3,'Fixture count')
        source,source_params=source_oracle.create();cases=[];source_reports=[];acceptance=[];supplement=[]
        with torch.no_grad():
            for fixture in frozen_fixtures:
                inputs,expected,source_record=source_oracle.evaluate(source,source_params,fixture)
                need(fixture['source_length']!=fixture['decoder_steps'],'Unequal source/target lengths')
                need(all(torch.count_nonzero(v)==v.numel() and v.min()<0 and v.max()>0 for v in fixture['parameters'].values()),'Signed nonzero Parameter fixture')
                need(all(torch.count_nonzero(x)>0 for x in inputs),'Nonzero inputs')
                need(all(torch.count_nonzero(expected[k])==expected[k].numel() for k in [2,3]),'Nonzero bridge hidden/cell')
                need(not torch.allclose(expected[2],expected[3]),'Hidden/cell discriminability')
                reference=Reference();weights=state_map(reference,SEMANTIC_MAPPING,doc,fixture)
                manual=reference(*inputs)
                source_record['seed']=fixture['seed'];source_record['numerical']=compare(expected,manual)
                source_reports.append(source_record)
                supplement.append({'seed':fixture['seed'],'checks':source_oracle.nonzero_state_checks(source,weights,inputs,fixture)})
                runtime=Runtime(doc,weights)
                def graph_model(*args):return tuple(runtime.run(dict(zip(['source','target_prev'],args))).values())
                yaml_output,graph,mode=traced(graph_model,weights,inputs)
                compare(expected,yaml_output)
                need(all(runtime.counts[name]>0 for name in doc['factors'] if 'operator' in doc['factors'][name]),'Unexecuted operator factor')
                need(runtime.counts['step_a_lin_i']==fixture['source_length'] and runtime.counts['cell_b_lin_i']==fixture['decoder_steps'],'Scan step counts')
                cases.append((fixture,inputs,expected,graph))
                acceptance.append({'seed':fixture['seed'],**evaluate_model(reference,SEMANTIC_MAPPING,doc,fixture,inputs,expected,graph),
                                   'factor_execution_counts':dict(runtime.counts)})
            checks['source_reference_match']='PASS';checks['reference_acceptance']='PASS'
            checks['state_mapping']='PASS';checks['fixture_nonvacuity']='PASS'
            executable=negatives(doc,cases);malformed=schema_negatives(doc,cases)
            evidence['negative_controls']={'status':'PASS','executed_model_mutations':executable,
                'schema_and_semantic_graph_mutations':malformed,
                'context_omission':{'status':'NOT_APPLICABLE','reason':'Pinned source has no extra context/attention port; identity bridge omission is exercised.'}}
            checks['negative_tests']='PASS'
        evidence['source_reference']=source_reports;evidence['nonzero_state_supplement']=supplement
        evidence['reference_acceptance']=acceptance
        evidence['integrity_after']=integrity();evidence['preservation_after']=preserve_check()
        status='PASS'
    except Exception as exc:
        evidence['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    duration=round(time.monotonic()-began,6)
    report={'status':status,'checks':checks,'duration_seconds':duration,'evidence':evidence,
            'compute_policy':os.environ.get('AGENTIC_COMPUTE_POLICY'),'torch':torch.__version__,
            'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'device':'cpu','api_requests_made':0,
            'scope':'Pre-embedded one-layer teacher-forced unit only; no full-model/initializer/train/Agent Edit claim'}
    save(run_dir/'report.json',report)
    axes=['source_grounding','representable_scope','structural_fidelity','dataflow_attributes',
          'resource_identity','state_recurrence_dependency','numerical_fidelity','schema_consistency',
          'source_reference_match','validator_preflight']
    save(HERE/'validation/results.json',{'status':status,'evaluation':{axis:status for axis in axes},
        'scope':report['scope'],'evidence':[str((run_dir/'report.json').relative_to(HERE))],
        'initializer_fidelity':'NOT_RUN','train_grad':'NOT_RUN','candidate_evaluation':'NOT_RUN',
        'api_requests_made':0,'schema_delta_used':False})
    # Freeze all evaluation input/output evidence, excluding the self-hashed report.
    paths=[]
    for folder in ['source','fixtures','negative_tests']:
        paths.extend(p for p in (HERE/folder).rglob('*') if p.is_file())
    paths.extend(HERE.glob('*.py'));paths.extend([HERE/'source_mapping.md',HERE/'schema_delta.md',HERE/'representation.yaml'])
    paths.extend([run_dir/'report.json',HERE/'validation/results.json',HERE/'history/prior_worker_manifest.json'])
    pre={'status':status,'execution':'EXECUTED','common_manifest_sha256':sha(RUN/'common/manifest.json'),
         'artifacts':{str(path.relative_to(HERE)):sha(path) for path in sorted(set(paths))},
         'checks':checks,'schema_delta_used':False,
         'evidence':[str((run_dir/'report.json').relative_to(HERE))],
         'compute_policy':'compute-v1.0','duration_seconds':duration,'failure_categories':[] if status=='PASS' else ['PREFLIGHT_FAILED']}
    save(HERE/'validation/preflight.json',pre)
    print(json.dumps({'status':status,'checks':checks,'report':str(run_dir/'report.json'),
                      'duration_seconds':duration,'error':evidence.get('error',{}).get('message')},indent=2))
    return 0 if status=='PASS' else 1


def candidate(path,report):
    """Untouched generated Python requires separate, hash-bound source review.

    Metadata sidecar contains the response entrypoint and semantic_mapping; a
    review sidecar is created only after external review, never by candidate code.
    """
    integrity();preserve_check()
    pre=json.loads((HERE/'validation/preflight.json').read_text())
    need(pre['status']=='PASS','Frozen full preflight required')
    for name,digest in pre['artifacts'].items():need(sha(HERE/name)==digest,'Frozen evaluation artifact drift: '+name)
    path=path.resolve();review=json.loads(path.with_suffix('.review.json').read_text())
    need(review.get('reviewed') is True and review.get('sha256')==sha(path),'Entire candidate must be reviewed before import')
    text=path.read_text();tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):need(all(n.name.split('.')[0] in ['torch','math'] for n in node.names),'Candidate import outside allowlist')
        if isinstance(node,ast.ImportFrom):need((node.module or '').split('.')[0] in ['torch','math'],'Candidate import outside allowlist')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):need(node.func.id not in ['open','eval','exec','compile','__import__'],'Unsafe candidate call')
    metadata=json.loads(path.with_suffix('.metadata.json').read_text())
    module_spec=importlib.util.spec_from_file_location('reviewed_candidate',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    ep=metadata['entrypoint'];need(ep['forward_inputs']==ENTRYPOINT['forward_inputs'] and ep['forward_outputs']==OUTPUTS,'Entrypoint port order')
    model=getattr(module,ep['factory'])(**ep['constructor_kwargs'])
    doc=yaml.safe_load((HERE/'representation.yaml').read_text());validate(doc)
    source,params=source_oracle.create();records=[]
    with torch.no_grad():
        for fixture in fixtures(json.loads((HERE/'fixtures/plan.json').read_text())):
            inputs,expected,_=source_oracle.evaluate(source,params,fixture)
            runtime=Runtime(doc,fixture['parameters'])
            def graph_model(*args):return tuple(runtime.run(dict(zip(['source','target_prev'],args))).values())
            _,graph,_=traced(graph_model,fixture['parameters'],inputs)
            records.append({'seed':fixture['seed'],**evaluate_model(model,metadata['semantic_mapping'],doc,fixture,inputs,expected,graph)})
    save(report,{'status':'PASS','candidate_sha256':sha(path),'cases':records,
                 'scope':pre.get('scope','Same pre-embedded unit as frozen plan'),'schema_consistency':'PASS'})
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true');group.add_argument('--candidate',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.preflight:return preflight()
    need(args.report is not None,'--candidate requires --report')
    need(os.environ.get('AGENTIC_COMPUTE_POLICY')=='compute-v1.0','Run candidate through Coordinator')
    try:return candidate(args.candidate,args.report)
    except Exception as exc:
        save(args.report,{'status':'FAIL','reason':str(exc),'candidate_execution':'Not PASS'})
        print(json.dumps({'status':'FAIL','reason':str(exc)}));return 1


if __name__=='__main__':raise SystemExit(main())
