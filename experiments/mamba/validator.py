"""Owned preflight/candidate interfaces; no tensor import until a reserved slot.

Exit codes: 0 PASS, 1 FAIL, 2 BLOCKED. No request builder or API sender.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OWN=Path(__file__).resolve().parent
COMMON=OWN.parent/'common'
CHECK_NAMES=['source_pinned','source_reference_match','source_to_yaml_mapping','schema_consistency','reference_acceptance','negative_tests','state_mapping','fixture_nonvacuity']
COMMANDS=[]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(relative,value):
    path=OWN/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def require(test,message):
    if not test:raise ValueError(message)


def run(argv,timeout=120):
    start=time.monotonic()
    overrides={k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE']}
    try:
        proc=subprocess.run(argv,cwd=OWN,env={**os.environ,**overrides},capture_output=True,text=True,timeout=timeout)
        result={'argv':argv,'environment_overrides':overrides,'exit_code':proc.returncode,'duration_seconds':round(time.monotonic()-start,6),'stdout':proc.stdout,'stderr':proc.stderr}
    except subprocess.TimeoutExpired:
        result={'argv':argv,'environment_overrides':overrides,'exit_code':124,'duration_seconds':round(time.monotonic()-start,6),'stdout':'','stderr':'Timeout; numerical check not completed.'}
    COMMANDS.append(result)
    return result


def integrity():
    result=run(['/opt/conda/bin/python','-B',str(COMMON/'verify_setup.py'),'--worker','mamba'])
    write('logs/integrity.json',result)
    require(result['exit_code']==0,'Frozen/common/preexisting integrity verification failed')
    return json.loads(result['stdout'])


def source_check():
    provenance=json.loads((OWN/'source/provenance.json').read_text())
    require(provenance['status']=='PASS','Source pin unavailable')
    require(provenance['source_spans']['selected_calculation']==[263,279],'Source scope drift')
    for record in provenance['source_records']:
        require(sha(Path(record['original_path']))==record['sha256'],'Original source/metadata drift')
        require(sha(OWN/record['snapshot'])==record['sha256'],'Source snapshot drift')
    for record in provenance['dependent_operator_sources']:
        require(sha(Path(record['path']))==record['sha256'],'Pinned PyTorch source drift')
    return {'status':'PASS','source_sha256':provenance['sha256'],'scope':provenance['selected_scope']}


def static_checks():
    import yaml
    from schema_check import validate
    syntax=[]
    for path in sorted(OWN.rglob('*.py')):
        ast.parse(path.read_text(),filename=str(path));syntax.append(str(path.relative_to(OWN)))
    doc=yaml.safe_load((OWN/'representation.yaml').read_text())
    schema=validate(doc)
    mapping=json.loads((OWN/'source/mapping.json').read_text())
    mapped=set()
    for item in mapping['mappings']:
        require(all(263<=n<=279 for n in item['source_lines']),'Mapping source span')
        for path in item['yaml_paths']:
            current=doc
            for part in path.split('.'):current=current[part]
            if path.startswith('factors.'):mapped.add(path.split('.')[1])
    require({name for name,f in doc['factors'].items() if 'operator' in f}<=mapped,'Unmapped YAML operation')
    require((OWN/'source_mapping.md').is_file() and (OWN/'schema_delta.md').is_file(),'Required mapping/delta document missing')
    tests={}
    def mutate(name,fn):
        changed=copy.deepcopy(doc);fn(changed)
        try:validate(changed)
        except (ValueError,KeyError,IndexError,TypeError) as exc:tests[name]={'status':'PASS','observed_rejection':str(exc)}
        else:raise ValueError('Undetected malformed schema: '+name)
    mutate('missing_edge',lambda d:d['dataflow'].pop())
    mutate('duplicate_producer',lambda d:d['dataflow'].append(d['dataflow'][0]))
    mutate('unknown_operator',lambda d:d['factors']['gated'].update(operator='opaque_mamba'))
    mutate('bad_axis',lambda d:d['factors']['delta_layout']['attributes'].update(dim1=99))
    mutate('shape_mismatch',lambda d:d['interface']['gated']['outputs']['output']['shape'].__setitem__(1,4))
    mutate('cross_scope_edge',lambda d:next(edge for edge in d['dataflow'] if edge[1]=='step.next').__setitem__(0,'entry.initial_state'))
    mutate('undeclared_state',lambda d:d['state'].clear())
    mutate('unknown_computational_field',lambda d:d.update(secret_formula='unapproved'))
    mutate('dependency_order_unsupported',lambda d:d['dependency'].update(order=['a','b']))
    fixtures=json.loads((OWN/'fixtures/tensors.json').read_text())
    for case in fixtures['cases']:
        for value in case['tensors'].values():
            count=1
            for n in value['shape']:count*=n
            require((OWN/value['path']).stat().st_size==count*4,'Fixture byte count')
    return {'status':'PASS','syntax_files':syntax,'schema':schema,'source_mapping':'PASS_STATIC_PATH_COVERAGE_ONLY','malformed_schema_negatives':tests,'fixture_byte_sizes':'PASS','numerical_execution':False}


def reservation():
    path=OWN/'validation/compute_reservation.json'
    if not path.exists():return False,'No coordinated slot reservation. User explicitly selected: 슬롯 확보 전 수치 검증 대기.'
    record=json.loads(path.read_text())
    required=record.get('status')=='RESERVED' and record.get('architecture')=='mamba' and record.get('slots')==1 and record.get('global_compute_ceiling')==2 and bool(record.get('coordinator_evidence'))
    require(required,'Invalid compute reservation; no tensor execution')
    expiry=datetime.fromisoformat(record['expires_at_utc'])
    require(expiry.tzinfo is not None and expiry>datetime.now(timezone.utc),'Compute reservation expired')
    return True,'Confirmed coordinator reservation; cooperative scheduling, no cgroup enforcement.'


def artifacts():
    paths=[]
    for path in OWN.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(OWN)
        if relative.parts[0] in ('source','fixtures','negative_tests') or (len(relative.parts)==1 and path.suffix in ('.py','.yaml','.md') and path.name not in ('result.md','README.md','AGENTS.md')) or str(relative)=='validation/static_checks.json':paths.append(path)
    return {str(p.relative_to(OWN)):sha(p) for p in sorted(paths)}


def preflight():
    start=time.monotonic();checks={k:'BLOCKED' for k in CHECK_NAMES};details={};status='BLOCKED'
    frozen_hash=sha(COMMON/'manifest.json')
    try:
        verified=integrity();require(verified['common_manifest_sha256']==frozen_hash,'manifest mismatch')
        details['source_pinned']=source_check();checks['source_pinned']='PASS'
        static=static_checks();write('validation/static_checks.json',static)
        checks['schema_consistency']='PASS';checks['source_to_yaml_mapping']='PASS'
        reserved,reason=reservation();details['compute_reservation']=reason
        if reserved:
            result=run(['/opt/conda/bin/python','-B',str(OWN/'numeric_checks.py')])
            write('logs/numeric_preflight.json',result)
            if result['exit_code']==0:
                numeric=json.loads(result['stdout']);require(numeric['status']=='PASS','Numerical report not PASS')
                write('validation/numerical_checks.json',numeric)
                checks.update({key:'PASS' for key in CHECK_NAMES});status='PASS'
            else:
                for key in ('source_reference_match','reference_acceptance','negative_tests','state_mapping','fixture_nonvacuity'):checks[key]='FAIL'
                status='FAIL';details['numerical_error']='See logs/numeric_preflight.json'
        else:
            details['pending_checks']=['source vs reference outputs/intermediates','YAML interpreter vs reference outputs/intermediates','executable negative controls','Parameter identity and exact fixture transfer','fixture numerical nonvacuity']
    except Exception as exc:
        status='FAIL';details['error']={'type':type(exc).__name__,'reason':str(exc)}
    report={'status':status,'common_manifest_sha256':frozen_hash,'checks':checks,'schema_delta_used':False,'artifacts':artifacts(),'evidence':['source/provenance.json','source/mapping.json','source_mapping.md','logs/source_search.json','logs/integrity.json','validation/static_checks.json','validation/scheduling.json'],'details':details,'duration_seconds':round(time.monotonic()-start,6),'timestamp_utc':datetime.now(timezone.utc).isoformat(),'numerical_execution':any('numeric_checks.py' in ' '.join(c['argv']) for c in COMMANDS),'api_requests_made':0}
    if report['numerical_execution']:report['evidence'].append('logs/numeric_preflight.json')
    write('validation/preflight.json',report);write('logs/preflight_commands.json',COMMANDS)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0 if status=='PASS' else 2 if status=='BLOCKED' else 1


def candidate(path,report_path):
    result={'status':'BLOCKED','candidate':str(path),'numerical_execution':False}
    try:
        integrity();source_check()
        frozen=json.loads((OWN/'validation/preflight.json').read_text())
        require(frozen['status']=='PASS','Candidate evaluation requires a full preflight PASS')
        for relative,digest in frozen['artifacts'].items():require(sha(OWN/relative)==digest,'Preflight artifact drift: '+relative)
        reserved,reason=reservation();require(reserved,reason)
        review=json.loads((OWN/'validation/candidate_review.json').read_text())
        require(review.get('entire_source_reviewed') is True and review.get('candidate_sha256')==sha(path),'Entire candidate source review/hash required before import')
        ast.parse(path.read_text(),filename=str(path))
        adapter=OWN/'validation/candidate_adapter.json'
        require(adapter.is_file(),'Reviewed entrypoint and semantic Parameter role mapping required')
        execution=run(['/opt/conda/bin/python','-B',str(OWN/'numeric_checks.py'),'--candidate',str(path),'--adapter',str(adapter)])
        write('logs/candidate_commands.json',COMMANDS)
        result=json.loads(execution['stdout']);result['numerical_execution']=True
    except Exception as exc:
        result['reason']=str(exc)
    report_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']=='PASS' else 2 if result['status']=='BLOCKED' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--preflight',action='store_true');group.add_argument('--candidate',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.preflight:return preflight()
    parser.error('--report is required with --candidate') if args.report is None else None
    target=args.report.resolve();require(target.is_relative_to(OWN),'Report must remain in Mamba directory')
    return candidate(args.candidate.resolve(),target)

if __name__=='__main__':raise SystemExit(main())
