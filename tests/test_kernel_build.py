import base64
import io
import json
import re
from types import SimpleNamespace
import zipfile

import build_kernel
from independent_mri.contracts import TARGETS


def test_offline_ensemble_manifest_is_embedded_without_local_model_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(build_kernel,'__file__',str(tmp_path/'build_kernel.py'))
    (tmp_path/'independent_mri').mkdir()
    (tmp_path/'independent_mri'/'fixture.py').write_text('# CPU packaging fixture\n',encoding='utf-8')
    manifest = {'format':'independent-evidence-ensemble-v1','targets':TARGETS,'transform':'probability',
                'members':[{'checkpoint':'pathology-p1/best.pt','checkpoint_sha256':'a'*64,'weight':1.}]}
    path = tmp_path/'fixture_manifest.json'
    path.write_text(json.dumps(manifest),encoding='utf-8')
    destination = build_kernel.build(SimpleNamespace(name='fixture',module='ensemble',arguments=[],
        ensemble_manifest=str(path),offline=True,gpu=True,kernel=[],dataset=[]))
    runner = (destination/'runner.py').read_text(encoding='utf-8')
    match = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)",runner)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group(1)))) as archive:
        assert json.loads(archive.read('ensemble_manifest.json'))==manifest
    metadata = json.loads((destination/'kernel-metadata.json').read_text())
    assert metadata['is_private'] and metadata['enable_gpu']
    assert not metadata['enable_internet']
    assert '/kaggle/working/independent_source/ensemble_manifest.json' in runner
