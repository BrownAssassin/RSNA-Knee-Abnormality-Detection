"""Bounded read of our own Kaggle live logs, using the official CLI endpoint.

This does not download images or enumerate the thousands of cache output files.
The SDK RPC and completed-output API cannot provide this running-session stream.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import time


def event_payload(line):
    if not line.startswith('data:'):
        return None
    raw = line[5:].strip()
    if not raw or raw == 'END_OF_LOG':
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return {'data':raw}
    return value if isinstance(value,dict) else {'data':str(value)}


def snapshot(api,kernel,seconds=25,limit=12):
    from kaggle.api.kaggle_api_extended import KaggleEnv
    from requests import RequestException
    owner,slug = kernel.split('/')
    if owner != 'mrinanksivakumar':
        raise ValueError('Live monitoring is restricted to our authorized account')
    recent = deque(maxlen=limit)
    start = time.monotonic()
    count,total_bytes = 0,0
    status,read_state = None,'not_connected'
    with api.build_kaggle_client() as client:
        http = client._http_client
        http._init_session()
        base = http._endpoint if http._env==KaggleEnv.PROD else f'{http._endpoint}/api'
        headers = dict(http._session.headers)
        headers['Accept'] = 'text/event-stream, */*'
        headers.pop('Content-Type',None)
        try:
            with http._session.get(f'{base}/v1/kernels/logs/stream/{owner}/{slug}',
                    stream=True,headers=headers,auth=http._session.auth,timeout=(8,5)) as response:
                status = response.status_code
                response.raise_for_status()
                response.encoding = 'utf-8'
                read_state = 'connected'
                for line in response.iter_lines(decode_unicode=True):
                    total_bytes += len(line)
                    event = event_payload(line)
                    if event:
                        count += 1
                        # Favor meaningful progress/exception lines over package progress bars.
                        data = str(event.get('data',''))
                        if any(marker in data for marker in ('{','Error','Traceback','Exception')):
                            recent.append(event)
                    if time.monotonic()-start>=seconds or total_bytes>=4*1024**2:
                        read_state = 'bounded_snapshot'
                        break
                else:
                    read_state = 'stream_ended'
        except RequestException as exc:
            # Read timeouts after replay are normal for a live but currently idle stream.
            # Do not expose auth-bearing URLs, and do not infer remote job failure.
            read_state = type(exc).__name__
    return {'kernel':kernel,'time_utc':datetime.now(timezone.utc).isoformat(),
            'http_status':status,'read_state':read_state,'events_read':count,
            'elapsed_seconds':time.monotonic()-start,'recent_events':list(recent),
            'job_state_must_be_checked_separately':True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel')
    parser.add_argument('--seconds',type=int,default=25)
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    print(json.dumps(snapshot(api,args.kernel,args.seconds),indent=2),flush=True)


if __name__=='__main__': main()
