import pytest

from independent_mri.contracts import sha256
from independent_mri.infer import validate_runtime_sources


def test_runtime_source_changes_are_rejected(tmp_path):
    hashes = {}
    for name in ('model.py','data.py','dicom.py','contracts.py'):
        file = tmp_path/name
        file.write_text('# fixture only\n',encoding='utf-8')
        hashes[name] = sha256(file)
    manifest = {'source_sha256':hashes}
    validate_runtime_sources(manifest,tmp_path)
    (tmp_path/'data.py').write_text('# changed transform\n',encoding='utf-8')
    with pytest.raises(ValueError,match='data.py'):
        validate_runtime_sources(manifest,tmp_path)


def test_missing_source_provenance_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='model.py'):
        validate_runtime_sources({},tmp_path)
