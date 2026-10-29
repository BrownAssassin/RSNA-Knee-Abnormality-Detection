from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.ndimage import affine_transform

from .contracts import PLANE_AXES


def orientation(iop):
    a = np.asarray(iop, dtype=float)
    if a.shape != (6,) or not np.isfinite(a).all():
        raise ValueError("Missing/invalid ImageOrientationPatient")
    col, row = a[:3], a[3:]
    if abs(np.linalg.norm(col) - 1) > .01 or abs(np.linalg.norm(row) - 1) > .01 or abs(col @ row) > .01:
        raise ValueError("Nonorthonormal DICOM axes")
    normal = np.cross(col, row)
    plane = ["Sagittal", "Coronal", "Axial"][int(np.abs(normal).argmax())]
    return col, row, normal, plane


def canonical_basis(iop):
    col, row, normal, plane = orientation(iop)
    target_col, target_row, depth = map(np.asarray, PLANE_AXES[plane])
    # In-plane orthonormal reorientation, NOT an invented 3D reformat of oblique data.
    out_col = target_col - normal * (target_col @ normal)
    out_col /= np.linalg.norm(out_col)
    out_row = np.cross(normal, out_col)
    if out_row @ target_row < 0:
        out_row = -out_row
    return col, row, out_col, out_row, np.asarray(depth), plane


def render_slice(pixels, iop, spacing, size=384, fov_mm=192.0):
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError("Expected single-frame scalar DICOM slice")
    spacing = np.asarray(spacing, float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Invalid PixelSpacing")
    col, row, out_col, out_row, _, _ = canonical_basis(iop)
    scale = fov_mm / size
    matrix = np.array([[row @ out_row / spacing[0], row @ out_col / spacing[0]],
                       [col @ out_row / spacing[1], col @ out_col / spacing[1]]]) * scale
    center = (np.asarray(pixels.shape) - 1) / 2
    offset = center - matrix @ np.full(2, (size - 1) / 2)
    return affine_transform(pixels, matrix, offset, output_shape=(size, size),
                            order=1, mode="constant", cval=0, prefilter=False)


def read_series(folder, expected_plane=None, size=384, fov_mm=192.0):
    import pydicom
    paths = sorted(Path(folder).glob("*.dcm"))
    if not paths:
        raise ValueError(f"No DICOMs in selected series {folder}")
    headers = [(p, pydicom.dcmread(p, stop_before_pixels=True)) for p in paths]
    first = headers[0][1]
    *_, depth, plane = canonical_basis(first.ImageOrientationPatient)
    ordered = []
    for path, ds in headers:
        if orientation(ds.ImageOrientationPatient)[3] != plane:
            raise ValueError("Mixed anatomical planes within a series")
        if not np.allclose(ds.ImageOrientationPatient, first.ImageOrientationPatient, atol=.02):
            raise ValueError("Variable slice orientation needs explicit 3D resampling")
        position = np.asarray(ds.ImagePositionPatient, float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Invalid ImagePositionPatient")
        ordered.append((float(position @ depth), path, ds))
    ordered.sort(key=lambda value: (value[0], str(value[1])))
    positions = np.array([x[0] for x in ordered])
    if len(positions) > 1 and (np.diff(positions) <= 1e-5).any():
        raise ValueError("Duplicate physical slice positions; unsupported mixed acquisition")
    stack, mappings = [], []
    for coordinate, path, header in ordered:
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array.astype(np.float32)  # Decoding errors are fatal, not missing-view flags.
        pixels = pixels * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixels = float(pixels.max() + pixels.min()) - pixels
        elif ds.PhotometricInterpretation != "MONOCHROME2":
            raise ValueError(f"Unsupported photometric interpretation {ds.PhotometricInterpretation}")
        stack.append(render_slice(pixels, ds.ImageOrientationPatient, ds.PixelSpacing, size, fov_mm))
        mappings.append({"sop_uid": str(ds.SOPInstanceUID), "file": path.name,
                         "position_patient": list(map(float, ds.ImagePositionPatient)),
                         "coordinate_mm": coordinate, "pixel_spacing": list(map(float, ds.PixelSpacing)),
                         "iop": list(map(float, ds.ImageOrientationPatient)),
                         "transfer_syntax": str(ds.file_meta.TransferSyntaxUID)})
    stack = np.stack(stack)
    positive = stack[stack > 0]
    if positive.size < 100:
        raise ValueError("Selected series has no usable nonzero MRI signal")
    low, high = np.percentile(positive, [.5, 99.5])
    if high <= low:
        raise ValueError("Collapsed series intensity distribution")
    normalized = np.rint(np.clip((stack - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return normalized, {"plane": plane, "descriptor_plane": expected_plane,
                         "plane_disagreement": expected_plane is not None and plane != expected_plane,
                         "normalization": [float(low), float(high)], "fov_mm": fov_mm,
                         "size": size, "native_slice_count": len(stack),
                         "spacing_irregular": bool(len(positions) > 2 and np.std(np.diff(positions)) > .1),
                         "sources": mappings}


def select_series(frame):
    """Five deterministic slots; fluid sensitivity and fat suppression stay distinct."""
    slots, used = [None] * 5, set()
    preferences = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", None), ("Axial", None)]
    for slot, (plane, fluid) in enumerate(preferences):
        rows = frame[frame.Anatomical_Plane.eq(plane)].copy()
        rows = rows[~rows.SeriesInstanceUID.astype(str).isin(used)]
        if fluid is not None:
            rows = rows[rows.Fluid_Sensitive.eq(fluid)]
        if rows.empty:
            continue
        rows = rows.sort_values(["Fluid_Sensitive", "Fat_Suppression", "SeriesInstanceUID"],
                                ascending=[False, False, True])
        selected = rows.iloc[0]
        slots[slot] = selected.to_dict()
        used.add(str(selected.SeriesInstanceUID))
    return slots
