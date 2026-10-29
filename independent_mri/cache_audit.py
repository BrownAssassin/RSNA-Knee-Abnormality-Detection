"""Kaggle CPU source-mapping and independent re-render parity audit."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .contracts import UID,sha256,write_json
from .data import discover_caches
from .dicom import read_series
from .prepare import competition_root


def main():
    files,receipts = discover_caches()
    provenance = [json.loads(path.read_text()) for path in receipts]
    if len({r['source_train_sha256'] for r in provenance})!=1:
        raise ValueError('MRI shards were prepared from differing training tables')
    if len({r['series_descriptors_sha256'] for r in provenance})!=1:
        raise ValueError('MRI shards have differing series-descriptor provenance')
    root = competition_root()
    expected = pd.read_csv(root/'train.csv',usecols=[UID],dtype={UID:str})[UID]
    if expected.duplicated().any() or set(files)!=set(expected):
        raise ValueError('Cache does not exactly cover the complete official training cohort')
    renderer_sources = []
    for receipt in receipts:
        source = receipt.parent/'independent_source'/'independent_mri'/'dicom.py'
        if not source.is_file():
            raise FileNotFoundError('Executed preprocessing source is missing from a cache output')
        renderer_sources.append(sha256(source))
    if len(set(renderer_sources))!=1 or renderer_sources[0]!=sha256(Path(__file__).with_name('dicom.py')):
        raise ValueError('Executed cache renderer differs across shards or reference parity audit')
    transfers,missing,planes = Counter(),Counter(),Counter()
    native_counts,sizes,disagreements = [],[],0
    for uid,file in files.items():
        audit = json.loads(file.with_suffix('.json').read_text())
        if audit[UID]!=uid: raise ValueError('Cache source UID mismatch')
        sizes.append(file.stat().st_size)
        for info in audit['slots']:
            if info['missing']:
                missing[info['slot']]+=1; continue
            planes[info['plane']]+=1
            native_counts.append(info['native_slice_count'])
            disagreements+=int(info['plane_disagreement'])
            for source in info['sources']:
                transfers[source['transfer_syntax']]+=1
    ordered = sorted(files)
    selected = [ordered[i] for i in np.unique(np.linspace(0,len(ordered)-1,3).round().astype(int))]
    checks = []
    for uid in selected:
        file = files[uid]
        audit = json.loads(file.with_suffix('.json').read_text())
        if sha256(file)!=audit['cache_sha256']: raise ValueError('Cache SHA-256 mismatch')
        with np.load(file,allow_pickle=False) as stored:
            for slot,info in enumerate(audit['slots']):
                if info['missing']: continue
                actual,new_audit = read_series(root/'train_series'/uid/info['series_uid'],info['descriptor_plane'])
                if not np.array_equal(actual,stored[f'slot_{slot}']):
                    raise ValueError('Source DICOM re-render differs from cached representation')
                if new_audit['sources']!=info['sources']:
                    raise ValueError('Re-rendered source ordering/metadata differs from cache audit')
                checks.append({UID:uid,'slot':slot,'exact_reconstruction':True,
                               'source_sop_order_matches':new_audit['sources']==info['sources']})
    receipt = {'format':'independent-cache-parity-audit-v1','studies':len(files),'cache_bytes':sum(sizes),
               'missing_series_by_slot':dict(missing),'plane_counts':dict(planes),
               'dicom_transfer_syntax_slice_counts':dict(transfers),'descriptor_plane_disagreements':disagreements,
               'native_slice_count_quantiles':np.quantile(native_counts,[0,.5,.95,1]).tolist(),
               'executed_dicom_renderer_sha256':renderer_sources[0],
               'parity_checks':checks,'parity_scope':'exact uint8 canonical representation, NOT lossless raw DICOM',
               'source_report_text_read':False,'aggregate_only_export':True}
    write_json('/kaggle/working/cache_audit.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__': main()
