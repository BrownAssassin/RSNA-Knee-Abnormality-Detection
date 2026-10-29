"""Fetch one named private Kaggle artifact; do not enumerate image-cache outputs."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import os
from pathlib import Path,PurePosixPath
import time
from uuid import uuid4


def retry_delay(value,now=None):
    if not value: return 60
    try: return max(1,int(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            return max(1,int((parsed-(now or datetime.now(timezone.utc))).total_seconds()))
        except (ValueError,TypeError): return 60


def replace_with_retry(source,destination,attempts=8,base_delay=.05,sleep=time.sleep):
    """Atomically publish a downloaded artifact, tolerating transient Windows locks."""
    source,destination=Path(source),Path(destination)
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError as exc:
            if attempt+1==attempts:
                raise RuntimeError(f'Could not atomically publish artifact after {attempts} attempts') from exc
            sleep(base_delay*min(2**attempt,16))


def fetch(api,kernel,file_path,destination,version=1,max_bytes=32*1024**2):
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    from requests import HTTPError
    owner,slug = kernel.split('/')
    if owner!='mrinanksivakumar':
        raise ValueError('This helper is restricted to our authorized private outputs')
    if PurePosixPath(file_path).is_absolute() or '..' in PurePosixPath(file_path).parts:
        raise ValueError('Artifact path must be relative and non-traversing')
    destination = Path(destination).resolve()
    allowed = Path(__file__).resolve().parent/'private'
    if not destination.is_relative_to(allowed):
        raise ValueError('Competition artifacts must remain in the ignored private directory')
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug,request.kernel_slug = owner,slug
    request.file_path,request.version_number = file_path,version
    try:
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.download_kernel_output(request)
            response.raise_for_status()
            if int(response.headers.get('Content-Length',0))>max_bytes:
                raise ValueError('Refusing unexpectedly large artifact download')
            destination.parent.mkdir(parents=True,exist_ok=True)
            # Concurrent orchestrators can fetch the same manifest. A shared
            # fixed .part path races on Windows even when the final replace is
            # atomic, so each downloader owns a unique sibling temporary file.
            temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.{uuid4().hex}.part')
            size = 0
            try:
                with open(temporary,'xb') as target:
                    for block in response.iter_content(1<<20):
                        size += len(block)
                        if size>max_bytes: raise ValueError('Artifact exceeds compact-download limit')
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                replace_with_retry(temporary,destination)
                return size
            finally:
                temporary.unlink(missing_ok=True)
    except HTTPError as exc:
        if exc.response.status_code==429:
            delay = retry_delay(exc.response.headers.get('Retry-After'))
            raise RuntimeError(f'Kaggle rate limited this read; wait at least {delay} seconds before retrying') from None
        raise RuntimeError(f'Kaggle artifact read returned HTTP {exc.response.status_code}; '
                           'check remote job state separately') from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel')
    parser.add_argument('file')
    parser.add_argument('destination')
    parser.add_argument('--version',type=int,default=1)
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    print({'downloaded_bytes':fetch(api,args.kernel,args.file,args.destination,args.version)})


if __name__=='__main__': main()
