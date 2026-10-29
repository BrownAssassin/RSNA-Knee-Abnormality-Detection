"""Package only this independent source tree. Generated artifacts remain private."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

OWNER = "mrinanksivakumar"


def build(args):
    root = Path(__file__).resolve().parent
    destination = root / "private" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    embedded_manifest = None
    arguments = list(args.arguments)
    if getattr(args,'ensemble_manifest',None):
        if args.module!='ensemble' or '--manifest' in arguments:
            raise ValueError('Use one embedded manifest with the ensemble module only')
        from independent_mri.ensemble import validate_manifest
        embedded_manifest = json.loads(Path(args.ensemble_manifest).read_text(encoding='utf-8'))
        validate_manifest(embedded_manifest)
        arguments += ['--manifest','/kaggle/working/independent_source/ensemble_manifest.json']
    sources = {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted((root / "independent_mri").glob("*.py")):
            bundle.write(file, f"independent_mri/{file.name}")
            sources[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        if embedded_manifest is not None:
            bundle.writestr('ensemble_manifest.json',json.dumps(embedded_manifest,sort_keys=True,allow_nan=False))
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    packages = ["pydicom==3.0.1", "pylibjpeg==2.1.0", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"]
    if args.module != "prepare":
        packages += ["timm==1.0.29"]
    if args.module == 'anatomy_screen':
        packages += ['keras>=3.10','h5py']
    install = f"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', {', '.join(repr(p) for p in packages)}], check=True)\n"
    if args.offline:
        install = ("candidates = list(Path('/kaggle/input').glob('*/offline_wheels')) + list(Path('/kaggle/input').glob('notebooks/*/*/offline_wheels'))\n"
                   "if len(candidates) != 1: raise RuntimeError(f'Expected one offline dependency wheelhouse, got {candidates}')\n"
                   f"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--find-links', str(candidates[0]), {', '.join(repr(p) for p in packages)}], check=True)\n")
    runner = ("import base64, io, zipfile, subprocess, sys, os\nfrom pathlib import Path\n"
              + install +
              "root = Path('/kaggle/working/independent_source'); root.mkdir(exist_ok=True)\n"
              f"zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))).extractall(root)\n"
              "os.environ['PYTHONPATH'] = str(root)\n"
              f"subprocess.run([sys.executable, '-u', '-m', 'independent_mri.{args.module}', *{arguments!r}], cwd=root, check=True)\n")
    (destination / "runner.py").write_text(runner, encoding="utf-8")
    kernel_sources = list(args.kernel)
    initialization_source = f'{OWNER}/rsna-independent-real-mri-preflight-20261022'
    if args.module in ('suite','train') and initialization_source not in kernel_sources:
        kernel_sources.append(initialization_source)
    if args.module in ('suite','train') and ('warm' in args.arguments or 'public_s6' in args.arguments):
        kernel_sources.append(f'{OWNER}/rsna-independent-public-init-audit-20261022')
        kernel_sources.append(f'{OWNER}/rsna-independent-pretraining-exposure-20261022')
    metadata = {"id": f"{OWNER}/{args.name}", "title": args.name.replace("-", " ").title(),
                "code_file": "runner.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": args.gpu, "enable_internet": not args.offline,
                "competition_sources": ["rsna-knee-abnormality-detection"],
                "dataset_sources": args.dataset, "kernel_sources": kernel_sources}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    build_receipt = {'module':args.module,'sources_sha256':sources,
                     'runner_sha256':hashlib.sha256(runner.encode('utf-8')).hexdigest(),
                     'embedded_ensemble_manifest':embedded_manifest,'all_model_execution_location':'Kaggle',
                     'private':True,'gpu':args.gpu,'internet_disabled':args.offline}
    (destination/'build_receipt.json').write_text(json.dumps(build_receipt,indent=2),encoding='utf-8')
    print(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ensemble-manifest")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--kernel", action="append", default=[])
    parser.add_argument("arguments", nargs="*")
    build(parser.parse_args())
