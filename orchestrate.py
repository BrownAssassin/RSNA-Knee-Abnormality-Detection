"""Read-only polling and one-time Kaggle training launch after complete cache jobs.

No model fitting or inference occurs in this Windows process. No submissions are
sent by this preparation watcher; final notebooks require actual trained results.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from fetch_artifact import fetch

OWNER = 'mrinanksivakumar'
CACHES = [f'rsna-independent-physical-cache-{letter}-20261022' for letter in 'abcd']
LABELS = 'rsna-independent-supervision-20261022'
NAME = 'rsna-independent-main-training-20261022'
LABEL_SOURCES = ['stevenleehans/rsna-knee-llm-report-labels','pilkwang/rsna-knee-llm-labels',
                 'lixin73/rsna-knee-llm-report-labels-sol56','dreaddevelopment/rsna-knee-labels',
                 'riadmohamed42/jev-knee-labels','omerzalman/rsna-knee-98-accuracy-11-soft-labels',
                 'omerzalman/rsna-knee-99-accuracy-11-soft-labels']


def atomic_json(path,value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def push_error(response):
    if response is None:
        return 'Kaggle push returned no receipt'
    if response.error:
        return str(response.error)
    for field in ('invalidDatasetSources','invalidCompetitionSources','invalidKernelSources'):
        if getattr(response,field,None):
            return f'{field}: {getattr(response,field)}'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds',type=int,default=300)
    parser.add_argument('--max-hours',type=float,default=20)
    parser.add_argument('--suite',choices=['main','warm','ablation-a','ablation-b','cache-audit'],default='main')
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    root = Path(__file__).resolve().parent
    kernel_name = NAME if args.suite=='main' else f'rsna-independent-{args.suite}-training-20261022'
    if args.suite=='cache-audit':
        kernel_name = 'rsna-independent-full-cache-audit-20261022'
    state_path = root/'private'/('orchestrator.json' if args.suite=='main'
                               else f'orchestrator_{args.suite.replace("-","_")}.json')
    dependencies = CACHES+[LABELS]
    if args.suite=='warm':
        dependencies += ['rsna-independent-public-init-audit-20261022',
                         'rsna-independent-pretraining-exposure-20261022']
    state = json.loads(state_path.read_text()) if state_path.exists() else {'launched':False,'polls':[]}
    started = time.time()
    while time.time()-started<args.max_hours*3600:
        try:
            statuses = {name:str(api.kernels_status(f'{OWNER}/{name}').status) for name in dependencies}
        except Exception as exc:
            # A read outage is not a failed model or permission to duplicate a job.
            state['last_read_error'] = {'type':type(exc).__name__,
                                       'time_utc':datetime.now(timezone.utc).isoformat()}
            atomic_json(state_path,state)
            print(json.dumps({'poll_read_error':state['last_read_error'],'retry_after_seconds':args.poll_seconds}),flush=True)
            time.sleep(args.poll_seconds)
            continue
        record = {'time_utc':datetime.now(timezone.utc).isoformat(),'statuses':statuses}
        state['polls'].append(record)
        atomic_json(state_path,state)
        print(json.dumps(record),flush=True)
        if any('ERROR' in status.upper() or 'CANCEL' in status.upper() for status in statuses.values()):
            state['needs_diagnosis'] = True
            atomic_json(state_path,state)
            return  # Never launch training against incomplete or failed shards.
        labels_current = False
        if 'COMPLETE' in statuses[LABELS].upper():
            audit_dir = root/'private'/'current_supervision'
            audit_dir.mkdir(parents=True,exist_ok=True)
            manifest_path = audit_dir/'supervision_manifest.json'
            try:
                # Request one known artifact, not three pages of irrelevant source files.
                # Latest completed output is checked by implementation SHA below.
                fetch(api,f'{OWNER}/{LABELS}','supervision_manifest.json',manifest_path,version=None)
            except RuntimeError as exc:
                state['last_artifact_read_error'] = str(exc)
                atomic_json(state_path,state)
                print(json.dumps({'artifact_read_error':str(exc),'retry_after_seconds':args.poll_seconds}),flush=True)
                time.sleep(args.poll_seconds)
                continue
            expected = hashlib.sha256((root/'independent_mri'/'supervision.py').read_bytes()).hexdigest()
            if manifest_path.is_file():
                labels_current = json.loads(manifest_path.read_text()).get('implementation_sha256')==expected
            if not labels_current:
                command = [sys.executable,str(root/'build_kernel.py'),'--name',LABELS,'--module','supervision']
                for source in LABEL_SOURCES:
                    command += ['--dataset',source]
                subprocess.run(command,check=True,cwd=root)
                response = api.kernels_push(str(root/'private'/LABELS))
                error = push_error(response)
                state['label_refresh'] = {'response':str(response),'error':error}
                atomic_json(state_path,state)
                if error and 'session count' not in error.lower():
                    raise RuntimeError(error)
                print(json.dumps({'label_refresh':str(response)}),flush=True)
        if labels_current and all('COMPLETE' in status.upper() for status in statuses.values()):
            if not state['launched']:
                if args.suite.startswith('ablation'):
                    priority = [root/'private'/'orchestrator.json',root/'private'/'orchestrator_warm.json']
                    if not all(p.is_file() and json.loads(p.read_text()).get('launched') for p in priority):
                        print(json.dumps({'waiting_for_priority_candidate_launches':True}),flush=True)
                        time.sleep(args.poll_seconds)
                        continue
                if state.get('push_intent'):
                    # A previous HTTP push may have succeeded before the connection
                    # failed. Stop this launcher for explicit remote reconciliation.
                    state['requires_push_reconciliation'] = True
                    atomic_json(state_path,state)
                    print(json.dumps({'requires_push_reconciliation':kernel_name}),flush=True)
                    return
                command = [sys.executable,str(root/'build_kernel.py'),'--name',kernel_name,'--module',
                           'cache_audit' if args.suite=='cache-audit' else 'suite',
                           '--kernel',f'{OWNER}/{LABELS}']
                if args.suite!='cache-audit': command += ['--gpu']
                for name in CACHES:
                    command += ['--kernel',f'{OWNER}/{name}']
                if args.suite!='cache-audit': command += ['--','--suite',args.suite]
                subprocess.run(command,check=True,cwd=root)
                subprocess.run([str(root/'.venv'/'Scripts'/'kaggle.exe'),'quota'],check=True,cwd=root)
                # Persist the intent before push: an interrupted push must be reconciled, not duplicated.
                state['push_intent'] = datetime.now(timezone.utc).isoformat()
                atomic_json(state_path,state)
                push_options = {'acc':'NvidiaTeslaT4'} if args.suite!='cache-audit' else {}
                result = api.kernels_push(str(root/'private'/kernel_name),**push_options)
                error = push_error(result)
                if error:
                    state['launch_error'] = error
                    state.pop('push_intent',None)
                    atomic_json(state_path,state)
                    if 'session count' in error.lower():
                        print(json.dumps({'waiting_for_remote_capacity':error}),flush=True)
                        time.sleep(args.poll_seconds)
                        continue
                    raise RuntimeError(error)
                state['launched'] = True
                state['launch_response'] = str(result)
                atomic_json(state_path,state)
                print(json.dumps({'job_launched':f'{OWNER}/{kernel_name}','response':str(result),
                                  'model_training':args.suite!='cache-audit'}),flush=True)
            return
        time.sleep(args.poll_seconds)
    state['watcher_limit_reached'] = True
    atomic_json(state_path,state)


if __name__=='__main__': main()
