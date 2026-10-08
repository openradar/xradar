#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.virtual.iris.parser` (``IrisParser`` → ManifestStore).

Requires the ``xradar[virtual]`` extra; the whole module skips without it
(the codec/format tests in ``test_iris_virtual.py`` do not). Parity oracle
is ``open_iris_datatree``; comparisons are azimuth-ALIGNED because the
virtual view keeps acquisition order while the eager reader places rays by
azimuth-derived index (a north-straddling sweep start differs by a cyclic
roll — values identical).
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("zarr", minversion="3.1.6")  # zarr v3 only
pytest.importorskip("virtualizarr")

from obspec_utils.registry import ObjectStoreRegistry  # noqa: E402
from obstore.store import LocalStore  # noqa: E402
from virtualizarr.manifests import (  # noqa: E402
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata  # noqa: E402

from xradar.io.backends.iris import (  # noqa: E402
    SIGMET_DATA_TYPES,
    decode_array,
    iris_mapping,
    open_iris_datatree,
)
from xradar.io.virtual import IrisParser  # noqa: E402
from xradar.io.virtual.iris.format import (  # noqa: E402
    RECORD_SIZE,
    azimuth_sort_order,
    index_sweeps,
    parse_ingest_header,
)
from xradar.io.virtual.manifest import inline_variable  # noqa: E402

#: Sigmet type name -> code, for looking up eager decoders in parity checks.
_CODE_BY_NAME = {
    e["name"]: c for c, e in SIGMET_DATA_TYPES.items() if isinstance(e, dict)
}


@pytest.fixture(scope="module")
def local_registry():
    return ObjectStoreRegistry({"file://": LocalStore()})


def _open_tree(store) -> xr.DataTree:
    return xr.open_datatree(
        store,
        engine="zarr",
        consolidated=False,
        zarr_format=3,
        mask_and_scale=False,
        decode_times=False,
    )


@pytest.fixture(scope="module")
def cor_store(iris0_file, local_registry):
    return IrisParser()(f"file://{iris0_file}", local_registry)


@pytest.fixture(scope="module")
def cor_vdt(cor_store):
    return cor_store.to_virtual_datatree()


@pytest.fixture(scope="module")
def cor_hdr(iris0_file):
    return parse_ingest_header(Path(iris0_file).read_bytes())


@pytest.fixture(scope="module")
def cor_tree(cor_store) -> xr.DataTree:
    return _open_tree(cor_store)


@pytest.fixture(scope="module")
def cor_truth(iris0_file) -> xr.DataTree:
    return open_iris_datatree(iris0_file)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def test_single_ref_per_sweep_shared_across_moments(cor_vdt, iris0_file):
    buf_len = Path(iris0_file).stat().st_size

    for name, node in cor_vdt.children.items():
        ds = node.to_dataset()
        refs = []
        for var in ds.data_vars:
            marr = ds[var].data
            assert isinstance(marr, ManifestArray)
            assert ds[var].dims == ("azimuth", "range")
            entries = marr.manifest.dict()
            assert len(entries) == 1, (name, var)
            (ref,) = entries.values()
            assert ref["offset"] % RECORD_SIZE == 0
            assert ref["offset"] + ref["length"] <= buf_len
            refs.append((ref["offset"], ref["length"]))
        # every moment of the sweep references the SAME span
        assert len(set(refs)) == 1, name


def test_codec_config_concat_safe(cor_vdt):
    ds = cor_vdt["sweep_0"].to_dataset()
    for var in ds.data_vars:
        codecs = [c.to_dict() for c in ds[var].data.metadata.codecs]
        assert [c["name"] for c in codecs] == ["xradar-iris-sweep"]
        assert set(codecs[0]["configuration"]) == {
            "moment_index",
            "ndatatypes",
            "sort_rays",
            "pad_missing_rays",
        }


def test_root_metadata(cor_tree, cor_truth):
    root = cor_tree.attrs
    assert root["scan_name"] == "SURV_HV_300"
    assert "Corozal" in root["instrument_name"]
    for scalar in ("latitude", "longitude", "altitude"):
        np.testing.assert_allclose(
            float(cor_tree[scalar].values),
            float(cor_truth[scalar].values),
            atol=1e-4,
        )
    for scalar in ("time_coverage_start", "time_coverage_end"):
        assert cor_tree[scalar].values.item() == cor_truth[scalar].values.item()


# ---------------------------------------------------------------------------
# parity vs the eager backend (az-aligned)
# ---------------------------------------------------------------------------


def _decoded_like_eager(ds: xr.Dataset, var: str, wavelength_cm: float):
    """Stored raw words -> physical values via the eager per-type decoders."""
    from xradar.io.backends.iris import decode_kdp, decode_sqi

    raw = ds[var].values.astype("float64")
    attrs = ds[var].attrs
    if "scale_factor" in attrs:
        vals = raw * attrs["scale_factor"] + attrs["add_offset"]
        if "_FillValue" in attrs:
            vals = np.where(raw == attrs["_FillValue"], np.nan, vals)
        return vals
    entry = SIGMET_DATA_TYPES.get(_CODE_BY_NAME.get(attrs["sigmet_data_type"], -1))
    func = (entry or {}).get("func")
    # the eager decode views the wire words through the entry's declared
    # dtype (e.g. DB_KDP is int8) before applying the decoder
    wire = ds[var].values
    entry_dtype = np.dtype((entry or {}).get("dtype", "uint8"))
    if entry_dtype.kind == "i" and wire.dtype.itemsize == entry_dtype.itemsize:
        wire = wire.view(entry_dtype)
    typed = wire.astype("float64")
    if func is decode_sqi:
        with np.errstate(invalid="ignore"):
            out = decode_sqi(typed, **entry.get("fkw", {}))
        return np.ma.filled(out, np.nan)
    if func is decode_kdp:
        out = decode_kdp(typed, wavelength=wavelength_cm, **entry.get("fkw", {}))
        return np.ma.filled(out, np.nan)
    return None  # categorical / unmapped: no value gate


def _assert_sweep_parity(ds: xr.Dataset, xds: xr.Dataset, wavelength_cm: float):
    assert dict(ds.sizes) == dict(xds.sizes)
    order = azimuth_sort_order(ds["azimuth"].values)
    xorder = np.argsort(xds["azimuth"].values, kind="stable")

    np.testing.assert_allclose(
        ds["azimuth"].values[order], xds["azimuth"].values[xorder], atol=1e-6
    )
    np.testing.assert_allclose(
        ds["elevation"].values[order], xds["elevation"].values[xorder], atol=1e-6
    )
    np.testing.assert_allclose(ds["range"].values, xds["range"].values, atol=1e-3)
    ours_ms = ds["time"].values[order]
    theirs_ms = xds["time"].values[xorder].astype("datetime64[ms]").astype("float64")
    np.testing.assert_allclose(ours_ms, theirs_ms, atol=1000)  # dtime is 1 s
    np.testing.assert_allclose(
        ds["sweep_fixed_angle"].values, xds["sweep_fixed_angle"].values, atol=1e-6
    )

    checked = 0
    for var in ds.data_vars:
        attrs = ds[var].attrs
        if "sigmet_data_type" not in attrs:
            continue
        if var not in xds.data_vars:
            continue
        if attrs["sigmet_data_type"] == "DB_HCLASS":
            # categorical, no decoder: the raw class byte per bin must match
            # the eager reader exactly (it unpacks per range bin since #444)
            np.testing.assert_array_equal(
                ds[var].values[order], xds[var].values[xorder], err_msg=var
            )
            checked += 1
            continue
        if attrs["sigmet_data_type"] == "DB_HCLASS2":
            continue  # no fixture carries it
        for attr in ("long_name", "standard_name", "units"):
            assert attrs.get(attr) == xds[var].attrs.get(attr), (var, attr)
        mine = _decoded_like_eager(ds, var, wavelength_cm)
        if mine is None:
            continue
        theirs = xds[var].values.astype("float64")
        mine = mine[order]
        theirs = theirs[xorder]
        both = np.isfinite(mine) & np.isfinite(theirs)
        assert both.any(), var
        np.testing.assert_allclose(mine[both], theirs[both], atol=1e-5, err_msg=var)
        checked += 1
    assert checked >= 5


def test_parity_all_sweeps_corozal(cor_tree, cor_truth, cor_hdr):
    """All 10 sweeps of the 8-bit IDEAM volume — includes the north-
    straddling sweep_9 that requires az-aligned comparison."""
    assert set(cor_tree.children) == {
        n for n in cor_truth.children if n.startswith("sweep_")
    }
    for name in cor_tree.children:
        _assert_sweep_parity(
            cor_tree[name].dataset, cor_truth[name].ds, cor_hdr.wavelength_cm
        )


def test_parity_surgavere_16bit(iris1_file, local_registry):
    """All-16-bit volume with DB_XHDR and a missing ray: the full
    iris_mapping gives the *2 types their CfRadial names, XHDR is framing
    (not a variable), and the missing ray is compacted like the eager
    reader."""
    tree = _open_tree(IrisParser()(f"file://{iris1_file}", local_registry))
    truth = open_iris_datatree(iris1_file)
    wavelength_cm = parse_ingest_header(Path(iris1_file).read_bytes()).wavelength_cm

    ds = tree["sweep_0"].dataset
    assert "DB_XHDR" not in ds.data_vars
    assert "DBZH" in ds.data_vars  # DB_DBZ2 under iris_mapping
    assert ds["DBZH"].attrs["sigmet_data_type"] == "DB_DBZ2"
    assert ds.sizes["azimuth"] == 360 - 1  # one missing ray, compacted
    _assert_sweep_parity(ds, truth["sweep_0"].ds, wavelength_cm)


def test_cf_scaling_matches_eager_decoders(iris1_file, local_registry):
    """Every stored scale_factor/add_offset reproduces the eager decoder on
    a full raw ramp (linear decode_array types)."""
    tree = _open_tree(IrisParser()(f"file://{iris1_file}", local_registry))
    ds = tree["sweep_0"].dataset
    ramp = np.arange(2, 65534, 97, dtype="float64")
    checked = 0
    for var in ds.data_vars:
        attrs = ds[var].attrs
        if "scale_factor" not in attrs or "sigmet_data_type" not in attrs:
            continue
        entry = SIGMET_DATA_TYPES[_CODE_BY_NAME[attrs["sigmet_data_type"]]]
        if entry.get("func") is not decode_array:
            continue  # nyquist/phidp wrappers verified via file parity
        ours = ramp * attrs["scale_factor"] + attrs["add_offset"]
        theirs = np.ma.filled(decode_array(ramp.copy(), **entry.get("fkw", {})), np.nan)
        both = np.isfinite(theirs)
        np.testing.assert_allclose(ours[both], theirs[both], atol=1e-9, err_msg=var)
        checked += 1
    assert checked >= 5


def test_signed_wire_types_stay_raw():
    """Types whose table entry declares a signed dtype must NOT get linear
    CF scaling — the stored unsigned words would decode sign-wrapped
    (e.g. DB_SHEAR raw 255 is -25.8, not +25.4)."""
    from xradar.io.virtual.iris.parser import _cf_scaling

    signed = [
        code
        for code, entry in SIGMET_DATA_TYPES.items()
        if isinstance(entry, dict)
        and np.dtype(entry.get("dtype", "uint8")).kind == "i"
        and (entry.get("fkw") or {}).get("scale")
    ]
    assert signed  # the table does contain signed linear types
    for code in signed:
        assert _cf_scaling(code, 8.5, 17.0) is None, code


def test_velc_gets_scaling_and_fill():
    """DB_VELC is linear-with-mask like DB_VEL: it must carry CF scaling
    AND ``_FillValue`` so masked bins decode to NaN, matching the eager
    decoder on the full uint8 ramp."""
    from xradar.io.virtual.iris.parser import _cf_scaling

    code = _CODE_BY_NAME["DB_VELC"]
    scaling = _cf_scaling(code, 8.5, 17.0)
    assert scaling is not None
    scale_factor, add_offset, fill = scaling
    assert fill == 0

    ramp = np.arange(0, 256, dtype="float64")
    entry = SIGMET_DATA_TYPES[code]
    theirs = np.ma.filled(decode_array(ramp.copy(), **entry["fkw"]), np.nan)
    ours = np.where(ramp == fill, np.nan, ramp * scale_factor + add_offset)
    both = np.isfinite(ours) & np.isfinite(theirs)
    assert np.array_equal(np.isfinite(ours), np.isfinite(theirs))
    np.testing.assert_allclose(ours[both], theirs[both], atol=1e-9)


# ---------------------------------------------------------------------------
# flavors
# ---------------------------------------------------------------------------


def _sorted_like_builder(group: ManifestGroup) -> ManifestGroup:
    """What a store builder does to publish azimuth-sorted sweeps: flag every
    ``xradar-iris-sweep`` moment with ``sort_rays: true`` and write the per-ray
    coordinates in the codec's own ``azimuth_sort_order``."""
    groups = {k: _sorted_like_builder(g) for k, g in group.groups.items()}
    arrays = dict(group.arrays)
    if "azimuth" in arrays:

        def values(ma):
            (data,) = ma.manifest._inlined.values()
            return np.frombuffer(data, dtype="<f8")

        order = azimuth_sort_order(values(arrays["azimuth"]))
        for name, ma in list(arrays.items()):
            meta = ma.metadata.to_dict()
            if meta["codecs"][0]["name"] == "xradar-iris-sweep":
                codec = meta["codecs"][0]
                codec = {**codec, "configuration": {**codec["configuration"]}}
                codec["configuration"]["sort_rays"] = True
                arrays[name] = ManifestArray(
                    metadata=create_v3_array_metadata(
                        shape=ma.metadata.shape,
                        chunk_shape=ma.metadata.chunks,
                        data_type=ma.metadata.data_type.to_native_dtype(),
                        fill_value=ma.metadata.fill_value,
                        codecs=[codec],
                        attributes=meta.get("attributes", {}),
                        dimension_names=ma.metadata.dimension_names,
                    ),
                    chunkmanifest=ma.manifest,
                )
            elif name in ("azimuth", "elevation", "time"):
                arrays[name] = inline_variable(
                    ("azimuth",), values(ma)[order], meta.get("attributes", {})
                )
    return ManifestGroup(
        arrays=arrays, groups=groups, attributes=group.metadata.attributes
    )


def test_parser_emits_pure_pointers(cor_store, cor_tree):
    """The parser takes no ``sort_rays`` option: every moment declares
    ``sort_rays: false`` and rays keep the file's ray order."""
    with pytest.raises(TypeError):
        IrisParser(sort_rays=True)
    unsorted = 0
    for name, node in cor_store._group.groups.items():
        for var, ma in node.arrays.items():
            codec = ma.metadata.to_dict()["codecs"][0]
            if codec["name"] == "xradar-iris-sweep":
                assert codec["configuration"]["sort_rays"] is False, (name, var)
        unsorted += bool(np.any(np.diff(cor_tree[name]["azimuth"].values) < 0))
    assert unsorted  # not vacuous: some sweep's file order crosses north


def test_sorted_store_decodes(cor_store, cor_tree, local_registry):
    """Stores published with ``sort_rays: true`` keep reading: the codec
    sorts the rows by the same key the builder sorted the coordinates by."""
    group = _sorted_like_builder(cor_store._group)
    sorted_tree = _open_tree(ManifestStore(group=group, registry=local_registry))
    for name in sorted_tree.children:
        plain = cor_tree[name].dataset
        srt = sorted_tree[name].dataset
        assert np.all(np.diff(srt["azimuth"].values) > 0), f"{name}: not sorted"
        order = azimuth_sort_order(plain["azimuth"].values)
        np.testing.assert_array_equal(
            srt["DBZH"].values, plain["DBZH"].values[order], err_msg=name
        )
        np.testing.assert_array_equal(srt["time"].values, plain["time"].values[order])


def test_pad_missing_rays_keeps_slots(iris1_file, local_registry):
    """pad_missing_rays=True: azimuth dim = EXPECTED slots; the missing
    SUR ray reads as fill with NaN per-ray coords, at its acquisition slot.
    Sorted by a builder, the NaN slot sorts last."""
    url = f"file://{iris1_file}"
    compact = _open_tree(IrisParser()(url, local_registry))["sweep_0"].dataset
    padded_store = IrisParser(pad_missing_rays=True)(url, local_registry)
    padded = _open_tree(padded_store)["sweep_0"].dataset

    assert compact.sizes["azimuth"] == 359
    assert padded.sizes["azimuth"] == 360
    az = padded["azimuth"].values
    missing = np.isnan(az)
    assert int(missing.sum()) == 1
    np.testing.assert_allclose(az[~missing], compact["azimuth"].values, atol=1e-9)
    np.testing.assert_array_equal(
        padded["DBZH"].values[~missing], compact["DBZH"].values
    )
    assert (padded["DBZH"].values[missing] == 0).all()

    sorted_group = _sorted_like_builder(padded_store._group)
    srt = _open_tree(ManifestStore(group=sorted_group, registry=local_registry))
    srt = srt["sweep_0"].dataset
    assert np.isnan(srt["azimuth"].values[-1])  # NaN slot sorts last
    assert (srt["DBZH"].values[-1] == 0).all()


def test_string_scalars_are_vlen_not_fixed_width(cor_tree):
    """String scalars are stored as zarr's VARIABLE-LENGTH ``string`` dtype
    (vlen-utf8), never a fixed ``<U*`` width — numpy's transient input dtype
    (e.g. ``<U20`` from ``np.asarray("azimuth_surveillance")``) is discarded
    at encode time, so no width can ever be wrong or truncate."""
    from numcodecs import VLenUTF8

    from xradar.io.virtual.manifest import encode_vlen_utf8, inline_scalar

    # the stored declaration is variable-length regardless of input width
    marr = inline_scalar("azimuth_surveillance")  # np.asarray gives <U20
    assert marr.metadata.data_type.to_native_dtype().kind == "T"  # StringDType
    assert [c.to_dict()["name"] for c in marr.metadata.codecs] == ["vlen-utf8"]

    # encoding preserves arbitrary lengths and non-ASCII; decoded with the
    # SAME numcodecs codec zarr's vlen-utf8 uses on the read side
    values = np.asarray(["short", "x" * 500, "ñandú — 🌧"])
    assert values.dtype.kind == "U"  # fixed-width exists on input only
    assert list(VLenUTF8().decode(encode_vlen_utf8(values))) == list(values)

    # end to end: scalars read back full-length through a real store
    assert cor_tree["sweep_0"]["sweep_mode"].values.item() == "azimuth_surveillance"
    tcs = cor_tree["time_coverage_start"].values.item()
    assert len(tcs) == 20 and tcs.endswith("Z")


def test_colliding_moment_names_keep_both(iris0_file, cor_hdr):
    """Two Sigmet types mapping to one CfRadial name (DB_DBZ + DB_DBZ2 ->
    DBZH) keep both moments: the first under the CfRadial name, the second
    under its Sigmet name (the eager reader's ``_moment_names`` rule), each
    pointing at its own interleave ordinal."""
    import dataclasses

    from xradar.io.virtual.iris.format import range_centers
    from xradar.io.virtual.iris.parser import _sweep_group

    buf = memoryview(Path(iris0_file).read_bytes())
    sweep = index_sweeps(buf)[0]
    names = [h.type_name for h in sweep.headers]
    dbz, vel = names.index("DB_DBZ"), names.index("DB_VEL")
    headers = list(sweep.headers)
    headers[vel] = dataclasses.replace(headers[vel], type_code=9, type_name="DB_DBZ2")
    doctored = dataclasses.replace(sweep, headers=tuple(headers))

    group, _ = _sweep_group(
        "file://collision",
        buf,
        cor_hdr,
        doctored,
        0,
        range_centers(cor_hdr),
        8.5,
        17.0,
        False,
        set(),
    )
    ordinal = {
        name: arr.metadata.to_dict()["codecs"][0]["configuration"]["moment_index"]
        for name, arr in group.arrays.items()
        if arr.metadata.to_dict()["codecs"][0]["name"] == "xradar-iris-sweep"
    }
    assert ordinal["DBZH"] == dbz
    assert ordinal["DB_DBZ2"] == vel
    attrs = group.arrays["DB_DBZ2"].metadata.attributes
    assert attrs["sigmet_data_type"] == "DB_DBZ2"
    assert attrs["units"] == "dBZ"  # still reflectivity under its Sigmet name
    assert "VRADH" not in group.arrays


def test_drop_variables(iris0_file, local_registry):
    tree = _open_tree(
        IrisParser(drop_variables=["VRADH"])(f"file://{iris0_file}", local_registry)
    )
    ds = tree["sweep_0"].dataset
    assert "VRADH" not in ds.data_vars
    assert "DBZH" in ds.data_vars


# ---------------------------------------------------------------------------
# virtual dataset mechanics
# ---------------------------------------------------------------------------


def test_moment_name_mapping_follows_iris_mapping(iris0_file, cor_vdt):
    buf = Path(iris0_file).read_bytes()
    sweeps = index_sweeps(buf)
    expected = {
        iris_mapping.get(h.type_name, h.type_name)
        for h in sweeps[0].headers
        if h.type_name != "DB_XHDR"
    }
    assert set(cor_vdt["sweep_0"].to_dataset().data_vars) == expected


def test_vel_scales_with_unfolded_nyquist_width_with_nyquist():
    """DB_VEL uses the multi-PRF (unfolded) nyquist, DB_WIDTH the plain one;
    the fixtures are single-PRF, where both are equal, so pin it here."""
    from xradar.io.virtual.iris.parser import _cf_scaling

    for name, nyquist in (("DB_VEL", 17.0), ("DB_WIDTH", 8.5)):
        code = _CODE_BY_NAME[name]
        scale = SIGMET_DATA_TYPES[code]["fkw"]["scale"]
        assert _cf_scaling(code, 8.5, 17.0)[0] == pytest.approx(nyquist / scale), name


def test_elevation_midpoint_folds_below_the_horizon():
    from types import SimpleNamespace

    from xradar.io.virtual.iris.parser import _angle_midpoints

    rays = [SimpleNamespace(elevation_start=359.9, elevation_stop=0.1)]
    np.testing.assert_allclose(_angle_midpoints(rays, "elevation"), [0.0], atol=1e-9)
    rays = [SimpleNamespace(elevation_start=359.8, elevation_stop=359.9)]
    np.testing.assert_allclose(_angle_midpoints(rays, "elevation"), [-0.15], atol=1e-9)


@pytest.mark.parametrize(
    "doctor, match",
    [
        ("inconsistent", "not group-consistent"),
        ("short", r"hold \[.*\] rays"),
    ],
)
def test_missing_ray_census_must_be_group_consistent(
    iris1_file, local_registry, monkeypatch, doctor, match
):
    """Rows only align across a sweep's moments if every data type misses
    the same rays and holds as many; anything else is refused, never
    silently misaligned."""
    import dataclasses

    from xradar.io.virtual.iris import parser as sigmet_parser

    real_walk = sigmet_parser.walk_sweep

    def doctored(words, ndt, *args, **kwargs):
        census = list(real_walk(words, ndt, *args, **kwargs))
        last = census[-1]
        if doctor == "inconsistent":  # the last type misses another ray
            census[-1] = dataclasses.replace(last, missing=(*last.missing, 5))
        else:  # the stream ends before the last type's final ray
            census[-1] = dataclasses.replace(last, headers=last.headers[:-1])
        return tuple(census)

    monkeypatch.setattr(sigmet_parser, "walk_sweep", doctored)
    with pytest.raises(ValueError, match=match):
        IrisParser()(f"file://{iris1_file}", local_registry)


# ---------------------------------------------------------------------------
# shared with the eager reader (Phase R / S2 / D)
# ---------------------------------------------------------------------------


def test_root_matches_eager(cor_tree, cor_truth):
    """The root is the eager reader's: ``_root_attrs`` (source, stripped
    scan/instrument names, task description) over ``_assign_root``."""
    for key in ("source", "scan_name", "instrument_name", "comment", "Conventions"):
        assert cor_tree.attrs[key] == cor_truth.attrs[key], key
    assert cor_tree.attrs["source"] == "Sigmet"
    eager_only = {"sweep_group_name", "sweep_fixed_angle"}  # documented
    assert set(cor_tree.ds.variables) == set(cor_truth.ds.variables) - eager_only


def test_rhi_task_is_refused(iris0_file, local_registry, monkeypatch):
    import dataclasses

    from xradar.io.virtual.iris import parser as iris_parser

    real = iris_parser.read_volume

    def as_rhi(buf):
        volume = real(buf)
        return volume._replace(
            header=dataclasses.replace(volume.header, antenna_scan_mode=2)
        )

    monkeypatch.setattr(iris_parser, "read_volume", as_rhi)
    with pytest.raises(NotImplementedError, match="RHI"):
        IrisParser()(f"file://{iris0_file}", local_registry)


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda buf: b"", "not a whole number"),
        (lambda buf: buf[:RECORD_SIZE], "not a whole number"),
        (lambda buf: buf[: 5 * RECORD_SIZE + 100], "not a whole number"),
        (lambda buf: bytes(4 * RECORD_SIZE), "PRODUCT_HDR"),
        (lambda buf: bytes(range(256)) * (RECORD_SIZE // 64), "PRODUCT_HDR"),
        (lambda buf: buf[:RECORD_SIZE] + bytes(RECORD_SIZE), "INGEST_HEADER"),
        (lambda buf: buf[: 300 * RECORD_SIZE], "truncated or not an IRIS"),
    ],
    ids=[
        "empty",
        "one-record",
        "partial-record",
        "zeros",
        "foreign",
        "no-ingest",
        "truncated",
    ],
)
def test_malformed_files_raise_value_error(iris0_file, make, match):
    """Anything that is not a well-formed IRIS RAW file raises ValueError
    with a reason, never an unrelated KeyError/EOFError from deep inside."""
    from xradar.io.virtual.iris.format import read_volume

    buf = Path(iris0_file).read_bytes()
    with pytest.raises(ValueError, match=match):
        read_volume(make(buf))


def test_padded_rays_never_reach_the_coverage_strings(iris1_file, local_registry):
    """NaN times of padded missing rays are ignored (``_assign_root`` would
    otherwise write "NaTZ"): padding never changes the coverage. (Against
    the unpadded store, not the eager reader: this file's DB_XHDR
    millisecond times are not read, row 6.4.)"""
    padded = _open_tree(
        IrisParser(pad_missing_rays=True)(f"file://{iris1_file}", local_registry)
    )
    compact = _open_tree(IrisParser()(f"file://{iris1_file}", local_registry))
    assert padded["sweep_0"].sizes["azimuth"] > compact["sweep_0"].sizes["azimuth"]
    for key in ("time_coverage_start", "time_coverage_end"):
        value = padded[key].values.item()
        assert "NaT" not in value
        assert value == compact[key].values.item()


def _field_offset(struct_dict, path):
    """Byte offset of a dotted field path inside one of the eager reader's
    nested struct tables (so the tests corrupt exactly that field)."""
    import struct as _struct

    from xradar.io.backends.iris import _get_fmt_string

    head, *rest = path.split(".")
    offset = 0
    for key, value in struct_dict.items():
        if key == head:
            if rest:
                return offset + _field_offset(value, ".".join(rest))
            return offset
        if "fmt" in value:
            offset += _struct.calcsize("<" + value["fmt"])
        elif "size" in value:
            offset += _struct.calcsize("<" + value["size"])
        else:
            offset += _struct.calcsize(_get_fmt_string(value))
    raise KeyError(path)


def _patched(buf, offset, fmt, value):
    import struct as _struct

    data = bytearray(buf)
    _struct.pack_into("<" + fmt, data, offset, value)
    return bytes(data)


def test_read_volume_checks_the_sweep_directory(iris0_file):
    """Header checks ``IrisRawFile`` does not do itself: every sweep must
    start with its ingest_data_headers, and those must carry their id."""
    from xradar.io.backends.iris import RAW_PROD_BHDR
    from xradar.io.virtual.iris.format import BHDR_SIZE, read_volume

    buf = Path(iris0_file).read_bytes()
    first_idh = 2 * RECORD_SIZE + BHDR_SIZE
    with pytest.raises(ValueError, match="ingest_data_header identifiers"):
        read_volume(_patched(buf, first_idh, "h", 0))
    # record 2 claims sweep 2: sweep 1 then starts at record 3, away from
    # its ingest_data_headers
    sweep_no = 2 * RECORD_SIZE + _field_offset(RAW_PROD_BHDR, "sweep_number")
    with pytest.raises(ValueError, match="no ingest_data_headers|revisits"):
        read_volume(_patched(buf, sweep_no, "h", 2))


def test_corrupt_range_header_raises_value_error(iris0_file):
    from xradar.io.backends.iris import INGEST_HEADER
    from xradar.io.virtual.iris.format import read_volume

    buf = Path(iris0_file).read_bytes()
    step = RECORD_SIZE + _field_offset(
        INGEST_HEADER, "task_configuration.task_range_info.step_output_bins"
    )
    with pytest.raises(ValueError, match="corrupt IRIS header"):
        read_volume(_patched(buf, step, "i", 0))


def test_non_utf8_header_strings_are_kept(iris0_file, local_registry, tmp_path):
    """A Latin-1 site or task name (not valid UTF-8) is read as text, and
    the store's attrs stay JSON-serializable."""
    import json

    from xradar.io.backends.iris import INGEST_HEADER

    buf = bytearray(Path(iris0_file).read_bytes())
    for field in (
        "ingest_configuration.site_name",
        "task_configuration.task_end_info.task_description",
    ):
        buf[RECORD_SIZE + _field_offset(INGEST_HEADER, field)] = 0xE9  # e-acute
    path = tmp_path / "latin1.RAW"
    path.write_bytes(bytes(buf))
    store = IrisParser()(f"file://{path}", local_registry)
    attrs = dict(store._group.metadata.attributes)
    json.dumps(attrs)
    assert attrs["instrument_name"].startswith("\u00e9")
    assert attrs["comment"].startswith("\u00e9")


def _with_header(monkeypatch, **changes):
    """Make the parser see an ingest header with ``changes`` applied."""
    import dataclasses

    from xradar.io.virtual.iris import parser as iris_parser

    real = iris_parser.read_volume

    def patched(buf):
        volume = real(buf)
        return volume._replace(header=dataclasses.replace(volume.header, **changes))

    monkeypatch.setattr(iris_parser, "read_volume", patched)


def test_sector_task_is_a_sector_sweep(iris0_file, local_registry, monkeypatch):
    _with_header(monkeypatch, antenna_scan_mode=1)
    tree = _open_tree(IrisParser()(f"file://{iris0_file}", local_registry))
    assert tree["sweep_0"]["sweep_mode"].values.item() == "sector"


def test_multi_prf_unfolds_the_velocity_nyquist(
    iris0_file, local_registry, monkeypatch
):
    """The dual-PRF factor applies to the velocity nyquist (and VRADH's
    scaling) only, as in the eager reader."""
    plain = _open_tree(IrisParser()(f"file://{iris0_file}", local_registry))
    _with_header(monkeypatch, multi_prf_mode_flag=1)
    dual = _open_tree(IrisParser()(f"file://{iris0_file}", local_registry))
    before, after = plain["sweep_0"], dual["sweep_0"]
    assert float(after["nyquist_velocity"]) == pytest.approx(
        2 * float(before["nyquist_velocity"])
    )
    assert after["VRADH"].attrs["scale_factor"] == pytest.approx(
        2 * before["VRADH"].attrs["scale_factor"]
    )


def test_sweep_number_is_the_files_own(iris0_file, local_registry, tmp_path):
    """With a sweep missing from the file, groups are named by position and
    ``sweep_number`` is the file's own, exactly like the eager reader."""
    from xradar.io.virtual.iris.format import read_volume

    buf = Path(iris0_file).read_bytes()
    sweeps = read_volume(buf).sweeps
    gone = sweeps[1]
    data = buf[: gone.byte_offset] + buf[gone.byte_offset + gone.byte_length :]
    # the product header's structure size (offset 4) bounds the eager walk
    data = _patched(data, 4, "i", len(data))
    path = tmp_path / "gap.RAW"
    path.write_bytes(data)
    tree = _open_tree(IrisParser()(f"file://{path}", local_registry))
    truth = open_iris_datatree(str(path))
    for name in ("sweep_0", "sweep_1", "sweep_2"):
        assert int(tree[name]["sweep_number"]) == int(truth[name]["sweep_number"])
    assert int(tree["sweep_1"]["sweep_number"]) == 2


def _with_hclass_identifiers(monkeypatch, identifiers):
    """Make IrisRawFile report ``identifiers`` as the task_end_info
    ``echo_class_identifiers`` (no test file names its classifiers)."""
    from xradar.io.backends import iris

    init = iris.IrisRawFile.__init__

    def init_with_identifiers(self, *args, **kwargs):
        init(self, *args, **kwargs)
        task_end_info = self.ingest_header["task_configuration"]["task_end_info"]
        task_end_info["echo_class_identifiers"] = bytes(identifiers)

    monkeypatch.setattr(iris.IrisRawFile, "__init__", init_with_identifiers)


@pytest.mark.parametrize(
    "fixture, var, identifiers",
    [
        ("iris0_file", "DB_HCLASS", [1, 2, 3, 0, 0, 0]),
        ("iris1_file", "DB_HCLASS2", [1, 2, 3, 1, 2, 3]),
    ],
)
def test_hclass_flag_attrs_match_eager(
    fixture, var, identifiers, local_registry, monkeypatch, request
):
    """HydroClass flag attrs (#444) come from the file's task_end_info
    ``echo_class_identifiers`` through the eager helper, on both paths, for
    1- and 2-byte HydroClass."""
    path = request.getfixturevalue(fixture)
    # the files store no identifiers: no flags, the raw-words comment stays
    plain = _open_tree(IrisParser()(f"file://{path}", local_registry))
    assert "flag_meanings" not in plain["sweep_0"][var].attrs
    assert "comment" in plain["sweep_0"][var].attrs

    _with_hclass_identifiers(monkeypatch, identifiers)
    virt_attrs = _open_tree(IrisParser()(f"file://{path}", local_registry))["sweep_0"][
        var
    ].attrs
    eager_attrs = open_iris_datatree(path)["sweep_0"][var].attrs
    assert virt_attrs["flag_meanings"] == eager_attrs["flag_meanings"]
    for key in ("flag_masks", "flag_values"):
        np.testing.assert_array_equal(virt_attrs[key], eager_attrs[key])
    # the CF flags describe the words; no "no CF scaling" comment beside them
    assert "comment" not in virt_attrs


def test_hclass_flag_attrs_in_zarr_json(iris0_file, local_registry, monkeypatch):
    """zarr.json holds the flags as JSON lists of plain ints (zarr attrs
    carry no dtype, so they read back as lists, not uint8 arrays), and the
    stored words decode to the expected classes."""
    _with_hclass_identifiers(monkeypatch, [1, 2, 3, 0, 0, 0])
    store = IrisParser()(f"file://{iris0_file}", local_registry)
    stored = store._group.groups["sweep_0"].arrays["DB_HCLASS"].metadata.attributes
    assert stored["flag_masks"] == [0b111] * 7 + [0b111_000] * 8 + [0b11_000_000] * 2
    assert stored["flag_values"] == (
        list(range(7)) + [cls << 3 for cls in range(8)] + [cls << 6 for cls in range(2)]
    )
    assert all(type(v) is int for v in stored["flag_masks"] + stored["flag_values"])

    value = 106  # 0b01_101_010
    hclass = _open_tree(store)["sweep_0"]["DB_HCLASS"]
    assert value in np.unique(hclass.values)
    meanings = [
        meaning
        for meaning, mask, flag in zip(
            stored["flag_meanings"].split(),
            stored["flag_masks"],
            stored["flag_values"],
            strict=True,
        )
        if value & mask == flag
    ]
    assert meanings == ["meteo_rain", "precip_light_precipitation", "cell_convection"]


def _with_product_end(monkeypatch, **changes):
    """Make IrisRawFile report ``changes`` in the product header's
    product_end (the task configuration stays as stored)."""
    from xradar.io.backends import iris

    init = iris.IrisRawFile.__init__

    def init_with_product_end(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.product_hdr["product_end"].update(changes)

    monkeypatch.setattr(iris.IrisRawFile, "__init__", init_with_product_end)


def test_nyquist_follows_product_end_like_eager(
    iris0_file, local_registry, monkeypatch
):
    """prf and wavelength come from product_end, as in the eager decode,
    even when they differ from the task configuration (500 Hz, 5.33 cm)."""
    _with_product_end(monkeypatch, prf=1000, wavelength=1066)
    store = IrisParser()(f"file://{iris0_file}", local_registry)
    tree = xr.open_datatree(store, engine="zarr", consolidated=False, zarr_format=3)
    ds = tree["sweep_0"].ds
    # 10.66 cm * 1000 Hz / 4, no multi-PRF on this task
    assert float(ds["nyquist_velocity"]) == pytest.approx(0.1066 * 1000 / 4)
    eager = open_iris_datatree(iris0_file)["sweep_0"].ds
    mine = ds["VRADH"].sortby("azimuth").values
    theirs = eager["VRADH"].sortby("azimuth").values
    both = np.isfinite(mine) & np.isfinite(theirs) & (theirs != 0)
    assert both.sum() > 1000
    np.testing.assert_allclose(mine[both], theirs[both], atol=1e-4)


def test_gate_count_mismatch_raises(iris0_file, local_registry, monkeypatch):
    """The data's gate count (product_end) must match the range bins of
    task_range_info; a mismatch raises instead of misaligning range."""
    _with_product_end(monkeypatch, number_bins=600)
    with pytest.raises(ValueError, match="product_end declares 600 gates"):
        IrisParser()(f"file://{iris0_file}", local_registry)


@pytest.mark.parametrize("multi_prf", [0, 1])
def test_multi_prf_follows_eager(iris0_file, local_registry, monkeypatch, multi_prf):
    """The multi-PRF factor extends the 1-byte velocity's nyquist, on top
    of the product_end prf and wavelength, exactly as in the eager decode."""
    from xradar.io.backends import iris

    _with_product_end(monkeypatch, prf=1000, wavelength=1066)
    init = iris.IrisRawFile.__init__

    def init_with_multi_prf(self, *args, **kwargs):
        init(self, *args, **kwargs)
        task = self.ingest_header["task_configuration"]
        task["task_dsp_info"]["multi_prf_mode_flag"] = multi_prf

    monkeypatch.setattr(iris.IrisRawFile, "__init__", init_with_multi_prf)
    store = IrisParser()(f"file://{iris0_file}", local_registry)
    tree = xr.open_datatree(store, engine="zarr", consolidated=False, zarr_format=3)
    ds = tree["sweep_0"].ds
    base = 0.1066 * 1000 / 4
    assert float(ds["nyquist_velocity"]) == pytest.approx(base * (multi_prf + 1))
    # the patches apply to the eager reader too; eager VRADH no-data
    # decodes to 0, hence the (theirs != 0) mask
    eager = open_iris_datatree(iris0_file)["sweep_0"].ds
    mine = ds["VRADH"].sortby("azimuth").values
    theirs = eager["VRADH"].sortby("azimuth").values
    both = np.isfinite(mine) & np.isfinite(theirs) & (theirs != 0)
    assert both.sum() > 1000
    np.testing.assert_allclose(mine[both], theirs[both], atol=1e-4)


def test_width_scaling_ignores_multi_prf():
    """1-byte DB_WIDTH scales by the single-PRF nyquist and DB_VEL by the
    multi-PRF one, as eager decode_width/decode_vel do (no fixture carries
    1-byte width, so this pins the table-driven scaling directly)."""
    from xradar.io.virtual.iris.parser import _cf_scaling

    nyquist, nyquist_vel = 13.0, 26.0
    assert _cf_scaling(4, nyquist, nyquist_vel)[0] == pytest.approx(nyquist / 256)
    assert _cf_scaling(3, nyquist, nyquist_vel)[0] == pytest.approx(nyquist_vel / 127)


def _rbins_offsets(buf, sweep, group):
    """File byte offsets of the ``rbins`` header word of ray ``group``, one
    per data type, found by walking the sweep's RLE code words."""
    from xradar.io.virtual.iris.format import BHDR_SIZE, IDH_SIZE

    ndt = sweep.ndatatypes
    body = RECORD_SIZE - BHDR_SIZE  # bytes per record after its bhdr
    first = IDH_SIZE * ndt  # the ingest_data_header prologue

    def word(i):
        at = first + 2 * i
        offset = sweep.byte_offset + (at // body) * RECORD_SIZE + BHDR_SIZE
        offset += at % body
        return offset, int.from_bytes(buf[offset : offset + 2], "little", signed=True)

    offsets, i, ray = [], 0, 0
    while len(offsets) < ndt:
        start, code = word(i)
        if ray // ndt == group:
            assert code < 0 and code + 32768 > 4, "header not in one data run"
            offsets.append(word(i + 5)[0])
        i += 1
        while code != 1:
            i += code + 32768 if code < 0 else 0
            _, code = word(i)
            i += 1
        ray += 1
    return offsets


@pytest.mark.parametrize("all_types", [True, False])
def test_ray_with_zero_rbins_is_missing(
    iris0_file, local_registry, tmp_path, all_types
):
    """A ray whose header declares 0 bins is dropped, as the eager reader
    drops every ray with rbins == 0; when only some data types drop it, rows
    would misalign between moments, so the sweep is refused."""
    buf = bytearray(Path(iris0_file).read_bytes())
    sweep = index_sweeps(bytes(buf))[0]
    offsets = _rbins_offsets(buf, sweep, group=10)
    for offset in offsets if all_types else offsets[1:2]:
        buf[offset : offset + 2] = b"\x00\x00"
    if all_types:
        # the eager reader sizes its arrays by the written-ray count
        from xradar.io.backends.iris import INGEST_DATA_HEADER
        from xradar.io.virtual.iris.format import BHDR_SIZE, IDH_SIZE

        field = _field_offset(INGEST_DATA_HEADER, "number_rays_file_written")
        first = sweep.byte_offset + BHDR_SIZE + field
        for t in range(sweep.ndatatypes):
            at = first + t * IDH_SIZE
            written = int.from_bytes(buf[at : at + 2], "little")
            buf[at : at + 2] = (written - 1).to_bytes(2, "little")
    path = tmp_path / "rbins0.RAW"
    path.write_bytes(bytes(buf))

    if not all_types:
        with pytest.raises(ValueError, match="rows would misalign"):
            IrisParser()(f"file://{path}", local_registry)
        return
    ds = _open_tree(IrisParser()(f"file://{path}", local_registry))["sweep_0"].ds
    eager = open_iris_datatree(str(path))["sweep_0"].ds
    full = open_iris_datatree(iris0_file)["sweep_0"].ds
    assert ds.sizes["azimuth"] == full.sizes["azimuth"] - 1
    wavelength_cm = parse_ingest_header(bytes(buf)).wavelength_cm
    _assert_sweep_parity(ds, eager, wavelength_cm)


def test_variable_range_spacing_is_refused(iris0_file, local_registry, monkeypatch):
    """Variable gate spacing has no constant-step range coordinate: refuse
    instead of writing evenly spaced gates."""
    _with_header(monkeypatch, variable_range_spacing=True)
    with pytest.raises(NotImplementedError, match="variable range bin spacing"):
        IrisParser()(f"file://{iris0_file}", local_registry)


def test_read_volume_skips_the_gate_check_for_variable_spacing(iris0_file):
    """With variable spacing the range bins say nothing about the gate
    count, so read_volume leaves the refusal to the parser."""
    from xradar.io.backends.iris import INGEST_HEADER
    from xradar.io.virtual.iris.format import read_volume

    buf = Path(iris0_file).read_bytes()
    path = "task_configuration.task_range_info.variable_range_bin_spacing_flag"
    flag = RECORD_SIZE + _field_offset(INGEST_HEADER, path)
    nbins = RECORD_SIZE + _field_offset(
        INGEST_HEADER, "task_configuration.task_range_info.number_output_bins"
    )
    fewer = _patched(buf, nbins, "h", 100)  # disagrees with product_end
    with pytest.raises(ValueError, match="product_end declares"):
        read_volume(fewer)
    assert read_volume(_patched(fewer, flag, "H", 1)).header.variable_range_spacing


@pytest.mark.parametrize(
    "description", ["  padded task  ", "task\x00\x00", b"caf\xe9 radar "]
)
def test_root_comment_is_kept_like_eager(
    iris0_file, local_registry, monkeypatch, description
):
    """The task description reaches the root ``comment`` as the eager reader
    writes it: no stripping; bytes that are not UTF-8 decode as Latin-1."""
    from xradar.io.backends import iris

    init = iris.IrisRawFile.__init__

    def init_with_description(self, *args, **kwargs):
        init(self, *args, **kwargs)
        task = self.ingest_header["task_configuration"]
        task["task_end_info"]["task_description"] = description

    monkeypatch.setattr(iris.IrisRawFile, "__init__", init_with_description)
    tree = _open_tree(IrisParser()(f"file://{iris0_file}", local_registry))
    eager = open_iris_datatree(iris0_file).attrs["comment"]
    if isinstance(eager, bytes):
        eager = eager.decode("latin-1")
    assert tree.attrs["comment"] == eager
