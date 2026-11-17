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

import contextlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import xradar as xd
from xradar.io.virtual._checks import MIN_ZARR_BUILD  # numpy only

pytest.importorskip("zarr", minversion=MIN_ZARR_BUILD)  # builds stores
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
    read_volume,
)
from xradar.io.virtual.manifest import inline_variable  # noqa: E402

#: Sigmet type name -> code, for looking up eager decoders in parity checks.
_CODE_BY_NAME = {
    e["name"]: c for c, e in SIGMET_DATA_TYPES.items() if isinstance(e, dict)
}


@pytest.fixture(scope="module")
def local_registry():
    return ObjectStoreRegistry({"file://": LocalStore()})


def _open_tree(store, **kwargs) -> xr.DataTree:
    """The store's raw words and times; ``kwargs`` override the decoding."""
    kwargs = {"mask_and_scale": False, "decode_times": False, **kwargs}
    return xr.open_datatree(
        store, engine="zarr", consolidated=False, zarr_format=3, **kwargs
    )


@pytest.fixture(scope="module")
def cor_store(iris0_file, local_registry):
    return IrisParser()(f"file://{iris0_file}", local_registry)


@pytest.fixture(scope="module")
def cor_vdt(cor_store):
    return cor_store.to_virtual_datatree()


@pytest.fixture(scope="module")
def cor_hdr(iris0_file):
    return read_volume(Path(iris0_file).read_bytes()).header


@pytest.fixture(scope="module")
def cor_tree(cor_store) -> Iterator[xr.DataTree]:
    with _open_tree(cor_store) as tree:
        yield tree


@pytest.fixture(scope="module")
def cor_truth(iris0_file) -> Iterator[xr.DataTree]:
    with open_iris_datatree(iris0_file) as tree:
        yield tree


# the all-16-bit volume with DB_XHDR and one missing ray (iris1)
@pytest.fixture(scope="module")
def s16_store(iris1_file, local_registry):
    return IrisParser()(f"file://{iris1_file}", local_registry)


@pytest.fixture(scope="module")
def s16_padded(iris1_file, local_registry):
    return IrisParser(pad_missing_rays=True)(f"file://{iris1_file}", local_registry)


@pytest.fixture(scope="module")
def s16_truth(iris1_file) -> Iterator[xr.DataTree]:
    with open_iris_datatree(iris1_file) as tree:
        yield tree


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def _moments(ds) -> list[str]:
    """The sweep's moments; the sweep scalars are data variables too."""
    return [var for var in ds.data_vars if ds[var].dims == ("azimuth", "range")]


def test_single_ref_per_sweep_shared_across_moments(cor_vdt, iris0_file):
    buf_len = Path(iris0_file).stat().st_size

    for name, node in cor_vdt.children.items():
        ds = node.to_dataset()
        refs = []
        for var in _moments(ds):
            marr = ds[var].data
            assert isinstance(marr, ManifestArray)
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
    decoded = set()
    for var in _moments(ds):
        codecs = [c.to_dict() for c in ds[var].data.metadata.codecs]
        names = [c["name"] for c in codecs]
        # the nonlinear types decode in xradar-iris-decode, before the sweep codec
        assert names[-1] == "xradar-iris-sweep"
        assert names[:-1] in ([], ["xradar-iris-decode"])
        if names[0] == "xradar-iris-decode":
            decoded.add(var)
        assert set(codecs[-1]["configuration"]) == {
            "moment_index",
            "ndatatypes",
            "sort_rays",
            "pad_missing_rays",
        }
    # not linear (KDP, RHOHV) or scaled by the volume's nyquist (VRADH)
    assert decoded == {"KDP", "RHOHV", "VRADH"}


def test_sweep_scalars_are_data_variables(cor_tree, cor_truth):
    """The sweep scalars are data variables and the own coordinates are the
    eager ones (plus the site), so the volume exports like the eager one."""
    for name in [n for n in cor_truth.children if n.startswith("sweep_")]:
        ours = cor_tree[name].to_dataset(inherit=False)
        theirs = cor_truth[name].to_dataset(inherit=False)
        assert set(ours.coords) == set(theirs.coords) | {
            "latitude",
            "longitude",
            "altitude",
        }
        for var in ("sweep_mode", "sweep_number", "sweep_fixed_angle"):
            assert var in ours.data_vars and var in theirs.data_vars, var


@pytest.mark.parametrize("export", [xd.io.to_cfradial1, xd.io.to_cfradial2])
def test_virtual_volume_exports(cor_store, tmp_path, export):
    """A multi-sweep virtual volume exports like the eager one (sweep scalars
    as coordinates broke CfRadial1, StringDType strings broke CfRadial2)."""
    path = tmp_path / "cor.nc"
    with _open_tree(cor_store, mask_and_scale=True, decode_times=True) as tree:
        export(tree, path)
    assert path.stat().st_size > 0


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


def _decoded_like_eager(ds: xr.Dataset, var: str):
    """CF-scaled words decoded with their attrs, ``None`` for the categorical
    ones (the types xradar-iris-decode decodes arrive as float32)."""
    values = ds[var].values
    attrs = ds[var].attrs
    if "scale_factor" not in attrs:
        return None  # categorical: no value gate
    raw = values.astype("float64")
    vals = raw * attrs["scale_factor"] + attrs["add_offset"]
    if "_FillValue" in attrs:
        vals = np.where(raw == attrs["_FillValue"], np.nan, vals)
    return vals


def _assert_same_mask(var, mine, theirs):
    """Virtual and eager mask exactly the same bins."""
    mine_nan, eager_nan = np.isnan(mine), np.isnan(theirs)
    assert not (eager_nan & ~mine_nan).any(), f"{var}: eager masks more bins"
    assert not (mine_nan & ~eager_nan).any(), f"{var}: virtual masks more bins"


def _assert_sweep_parity(ds: xr.Dataset, xds: xr.Dataset):
    assert dict(ds.sizes) == dict(xds.sizes)
    order = azimuth_sort_order(ds["azimuth"].values)
    xorder = np.argsort(xds["azimuth"].values, kind="stable")

    # the eager float32 beam centers (_azimuth_centers) and elevations
    for coord in ("azimuth", "elevation"):
        assert ds[coord].dtype == xds[coord].dtype == np.float32, coord
        np.testing.assert_array_equal(
            ds[coord].values[order], xds[coord].values[xorder], err_msg=coord
        )
    # the eager range: same float32 centers, attrs from the header (#529)
    np.testing.assert_array_equal(ds["range"].values, xds["range"].values)
    assert ds["range"].attrs == xds["range"].attrs
    # int64 ms, the DB_XHDR times where the sweep has them: exact
    ours_ms = ds["time"].values[order]
    theirs_ms = xds["time"].values[xorder].astype("datetime64[ms]").astype("int64")
    np.testing.assert_array_equal(ours_ms, theirs_ms)
    np.testing.assert_allclose(
        ds["sweep_fixed_angle"].values, xds["sweep_fixed_angle"].values, atol=1e-6
    )

    # the sweeps differ only in what importers.md documents
    assert ds.attrs == xds.attrs
    assert set(xds.variables) <= set(ds.variables)
    assert set(ds.variables) - set(xds.variables) <= {
        "nyquist_velocity",
        "latitude",
        "longitude",
        "altitude",
    }
    for var in _moments(xds):
        assert set(xds[var].attrs) <= set(ds[var].attrs), var
        assert set(ds[var].attrs) - set(xds[var].attrs) <= {
            "sigmet_data_type",
            "comment",
            "scale_factor",
            "add_offset",
            "_FillValue",
        }, var

    checked = 0
    for var in ds.data_vars:
        attrs = ds[var].attrs
        if "sigmet_data_type" not in attrs:
            continue
        if var not in xds.data_vars:
            continue
        if attrs["sigmet_data_type"] in ("DB_HCLASS", "DB_HCLASS2"):
            # categorical, no decoder: the raw class word per bin must match
            # the eager reader exactly (it unpacks 1-byte words per range bin
            # since #444; it gives the 2-byte words as int16)
            assert ds[var].dtype.kind == "u", var  # the table's unsigned words
            np.testing.assert_array_equal(
                ds[var].values[order].astype("int64"),
                xds[var].values[xorder].astype("int64"),
                err_msg=var,
            )
            checked += 1
            continue
        for attr in ("long_name", "standard_name", "units"):
            assert attrs.get(attr) == xds[var].attrs.get(attr), (var, attr)
        theirs = xds[var].values.astype("float64")[xorder]
        if ds[var].dtype == np.float32:
            # decoded by the eager decoder in xradar-iris-decode: exact
            np.testing.assert_array_equal(
                ds[var].values[order].astype("float64"), theirs, err_msg=var
            )
            checked += 1
            continue
        mine = _decoded_like_eager(ds, var)
        if mine is None:
            continue
        mine = mine[order]
        _assert_same_mask(var, mine, theirs)
        both = np.isfinite(mine) & np.isfinite(theirs)
        assert both.any(), var
        np.testing.assert_allclose(mine[both], theirs[both], atol=1e-5, err_msg=var)
        checked += 1
    assert checked >= 5


def test_parity_all_sweeps_corozal(cor_tree, cor_truth):
    """All 10 sweeps of the 8-bit IDEAM volume — includes the north-
    straddling sweep_9 that requires az-aligned comparison."""
    assert set(cor_tree.children) == {
        n for n in cor_truth.children if n.startswith("sweep_")
    }
    for name in cor_tree.children:
        _assert_sweep_parity(cor_tree[name].dataset, cor_truth[name].ds)


def test_parity_surgavere_16bit(s16_store, s16_truth):
    """All-16-bit volume with DB_XHDR and a missing ray: the full
    iris_mapping gives the *2 types their CfRadial names, XHDR is framing
    (not a variable), and the missing ray is compacted like the eager
    reader."""
    tree = _open_tree(s16_store)
    truth = s16_truth

    ds = tree["sweep_0"].dataset
    assert "DB_XHDR" not in ds.data_vars
    assert "DBZH" in ds.data_vars  # DB_DBZ2 under iris_mapping
    assert ds["DBZH"].attrs["sigmet_data_type"] == "DB_DBZ2"
    assert ds.sizes["azimuth"] == 360 - 1  # one missing ray, compacted
    _assert_sweep_parity(ds, truth["sweep_0"].ds)


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


def test_signed_types_decode_in_the_codec():
    """Types whose table entry declares a signed dtype must NOT get linear
    CF scaling — the stored unsigned words would decode sign-wrapped
    (DB_VVEL2 word 65535 is -0.01, not +655.35). They decode in
    xradar-iris-decode instead."""
    from xradar.io.virtual.iris.decode import _cf_scaling, decode_codec_config

    signed = [
        code
        for code, entry in SIGMET_DATA_TYPES.items()
        if isinstance(entry, dict)
        and np.dtype(entry.get("dtype", "uint8")).kind == "i"
        and (entry.get("fkw") or {}).get("scale")
    ]
    assert signed  # the table does contain signed linear types
    for code in signed:
        assert _cf_scaling(code) is None, code
        volume = {"wavelength": 5.33, "nyquist": 8.5, "multi_prf_mode_flag": 0}
        assert decode_codec_config(code, volume) is not None, code


def test_velc_gets_scaling_and_fill():
    """DB_VELC is linear-with-mask like DB_VEL: it must carry CF scaling
    AND ``_FillValue`` so masked bins decode to NaN, matching the eager
    decoder on the full uint8 ramp."""
    from xradar.io.virtual.iris.decode import _cf_scaling

    code = _CODE_BY_NAME["DB_VELC"]
    scaling = _cf_scaling(code)
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
            return np.frombuffer(data, dtype=ma.metadata.data_type.to_native_dtype())

        order = azimuth_sort_order(values(arrays["azimuth"]))
        for name, ma in list(arrays.items()):
            meta = ma.metadata.to_dict()
            # the sweep codec is the last one (xradar-iris-decode may precede it)
            if meta["codecs"][-1]["name"] == "xradar-iris-sweep":
                codec = meta["codecs"][-1]
                codec = {**codec, "configuration": {**codec["configuration"]}}
                codec["configuration"]["sort_rays"] = True
                arrays[name] = ManifestArray(
                    metadata=create_v3_array_metadata(
                        shape=ma.metadata.shape,
                        chunk_shape=ma.metadata.chunks,
                        data_type=ma.metadata.data_type.to_native_dtype(),
                        fill_value=ma.metadata.fill_value,
                        codecs=[*meta["codecs"][:-1], codec],
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
            codec = ma.metadata.to_dict()["codecs"][-1]
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


def test_strings_read_back_like_eager(cor_tree, cor_truth):
    """Every string variable reads back as ``<U`` with the eager value."""
    strings = [
        (node.path, name)
        for node in cor_truth.subtree
        for name, var in node.variables.items()
        if var.dtype.kind == "U"
    ]
    assert len(strings) > 10
    for path, name in strings:
        ours, theirs = cor_tree[path][name], cor_truth[path][name]
        assert ours.dtype.kind == "U", (path, name)
        np.testing.assert_array_equal(ours.values, theirs.values)


def test_virtual_sweep_where_and_fillna(cor_store, cor_truth):
    """``where`` and ``fillna`` work on a virtual sweep as on an eager one
    (StringDType sweep strings made both raise)."""
    with xr.open_dataset(
        cor_store, group="sweep_0", engine="zarr", consolidated=False, zarr_format=3
    ) as ds:
        eager = cor_truth["sweep_0"].to_dataset()
        assert int(ds.where(ds.DBZH > 10).DBZH.count()) == int(
            eager.where(eager.DBZH > 10).DBZH.count()
        )
        assert int(ds.fillna(0).DBZH.isnull().sum()) == 0


def test_colliding_moment_names_keep_both(iris0_file, cor_hdr):
    """Two Sigmet types mapping to one CfRadial name (DB_DBZ + DB_DBZ2 ->
    DBZH) keep both moments: the later, finer one under the CfRadial name,
    the earlier one under its Sigmet name (the eager reader's
    ``_moment_names`` rule, #580), each pointing at its own interleave
    ordinal."""
    import dataclasses

    from xradar.io.virtual.iris.format import range_attrs, range_centers
    from xradar.io.virtual.iris.parser import _sweep_group

    buf = memoryview(Path(iris0_file).read_bytes())
    sweep = read_volume(buf).sweeps[0]
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
        range_attrs(cor_hdr),
        {"wavelength": 5.33, "nyquist": 8.5, "multi_prf_mode_flag": 0},
        8.5,
        False,
        set(),
    )
    ordinal = {
        name: arr.metadata.to_dict()["codecs"][-1]["configuration"]["moment_index"]
        for name, arr in group.arrays.items()
        if arr.metadata.to_dict()["codecs"][-1]["name"] == "xradar-iris-sweep"
    }
    assert dbz < vel
    assert ordinal["DBZH"] == vel
    assert ordinal["DB_DBZ"] == dbz
    attrs = group.arrays["DB_DBZ"].metadata.attributes
    assert attrs["sigmet_data_type"] == "DB_DBZ"
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
    sweeps = read_volume(buf).sweeps
    expected = {
        iris_mapping.get(h.type_name, h.type_name)
        for h in sweeps[0].headers
        if h.type_name != "DB_XHDR"
    }
    assert set(_moments(cor_vdt["sweep_0"].to_dataset())) == expected


def test_elevation_midpoint_folds_below_the_horizon():
    from types import SimpleNamespace

    from xradar.io.virtual.iris.parser import _elevation_midpoints

    rays = [SimpleNamespace(elevation_start=359.9, elevation_stop=0.1)]
    assert _elevation_midpoints(rays).dtype == np.float32  # like the eager reader
    np.testing.assert_allclose(_elevation_midpoints(rays), [0.0], atol=1e-9)
    rays = [SimpleNamespace(elevation_start=359.8, elevation_stop=359.9)]
    np.testing.assert_allclose(_elevation_midpoints(rays), [-0.15], atol=1e-9)


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
            missing = np.append(last.missing, 5).astype(last.missing.dtype)
            census[-1] = dataclasses.replace(last, missing=missing)
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


#: malformed files: how to make one from cor-main, and the ValueError's reason
_MALFORMED = [
    pytest.param(lambda buf: b"", "not a whole number", id="empty"),
    pytest.param(lambda buf: buf[:RECORD_SIZE], "not a whole number", id="one-record"),
    pytest.param(
        lambda buf: buf[: 5 * RECORD_SIZE + 100], "not a whole number", id="partial"
    ),
    pytest.param(lambda buf: bytes(4 * RECORD_SIZE), "PRODUCT_HDR", id="zeros"),
    pytest.param(
        lambda buf: bytes(range(256)) * (RECORD_SIZE // 64), "PRODUCT_HDR", id="foreign"
    ),
    pytest.param(
        lambda buf: buf[:RECORD_SIZE] + bytes(RECORD_SIZE),
        "INGEST_HEADER",
        id="no-ingest",
    ),
    pytest.param(
        lambda buf: buf[: 300 * RECORD_SIZE], "truncated or not an IRIS", id="truncated"
    ),
]


@pytest.mark.parametrize("make, match", _MALFORMED)
def test_malformed_files_raise_value_error(iris0_file, make, match):
    """Anything that is not a well-formed IRIS RAW file raises ValueError
    with a reason, never an unrelated KeyError/EOFError from deep inside."""
    from xradar.io.virtual.iris.format import read_volume

    buf = Path(iris0_file).read_bytes()
    with pytest.raises(ValueError, match=match):
        read_volume(make(buf))


@pytest.mark.parametrize("make, match", _MALFORMED)
def test_parser_refuses_malformed_files(
    iris0_file, local_registry, tmp_path, make, match
):
    """End to end: the parser raises the format's ValueError for a malformed
    file, before building any store."""
    path = tmp_path / "bad.RAW"
    path.write_bytes(make(Path(iris0_file).read_bytes()))
    with pytest.raises(ValueError, match=match):
        IrisParser()(f"file://{path}", local_registry)


def test_padded_rays_never_reach_the_coverage_strings(s16_padded, s16_store, s16_truth):
    """The times of padded missing rays are ignored (``_assign_root`` would
    otherwise write "NaTZ"): padding never changes the coverage, which is the
    eager reader's (this file has DB_XHDR millisecond times)."""
    padded = _open_tree(s16_padded)
    compact = _open_tree(s16_store)
    eager = s16_truth
    assert padded["sweep_0"].sizes["azimuth"] > compact["sweep_0"].sizes["azimuth"]
    for key in ("time_coverage_start", "time_coverage_end"):
        value = padded[key].values.item()
        assert "NaT" not in value
        assert value == compact[key].values.item() == eager[key].values.item()


@pytest.mark.file  # the stores come through request.getfixturevalue
@pytest.mark.parametrize(
    "store, truth",
    [("cor_store", "cor_truth"), ("s16_store", "s16_truth")],
    ids=["seconds", "xhdr-milliseconds"],
)
def test_decoded_times_equal_eager(store, truth, request):
    """Decoded to datetime64[ns], the ray times are the eager ones to the
    nanosecond: whole seconds (iris0), DB_XHDR milliseconds (iris1). Stored
    as float ms since 1970 they were up to 128 ns off."""
    store, truth = request.getfixturevalue(store), request.getfixturevalue(truth)
    tree = _open_tree(store, mask_and_scale=True, decode_times=True)
    for name in [n for n in truth.children if n.startswith("sweep_")]:
        ours, theirs = tree[name].ds, truth[name].ds
        order = azimuth_sort_order(ours["azimuth"].values)
        xorder = np.argsort(theirs["azimuth"].values, kind="stable")
        np.testing.assert_array_equal(
            ours["time"].values[order], theirs["time"].values[xorder]
        )


def test_padded_ray_time_is_nat(s16_padded):
    """Ray times are int64 ms; a padded missing ray's time is NaT decoded and
    NaN with ``decode_times=False``, like its NaN azimuth."""
    store = s16_padded
    raw = _open_tree(store)["sweep_0"]
    assert raw["time"].dtype == np.int64
    undecoded = _open_tree(store, mask_and_scale=True)["sweep_0"]
    padded = np.isnan(undecoded["azimuth"].values)
    assert padded.any()
    np.testing.assert_array_equal(np.isnan(undecoded["time"].values), padded)
    decoded = xr.decode_cf(undecoded.to_dataset())
    np.testing.assert_array_equal(np.isnat(decoded["time"].values), padded)


@pytest.mark.parametrize("volume", ["iris1", "iris0-rbins0"])
def test_padded_rows_are_no_data(
    volume, s16_padded, iris0_file, local_registry, tmp_path
):
    """The padded slot of a missing ray reads as no data in every moment:
    NaN (CF-scaled float64 and the float32 decode-codec types such as KDP and
    RHOHV), word 0 for the categorical HydroClass."""
    if volume == "iris1":
        store = s16_padded
    else:  # ray 10 of the first sweep declares 0 bins
        path = _rbins0_file(iris0_file, tmp_path)
        store = IrisParser(pad_missing_rays=True)(f"file://{path}", local_registry)
    node = _open_tree(store, mask_and_scale=True)["sweep_0"]
    padded = np.isnan(node["azimuth"].values)
    assert padded.sum() == 1
    for var in node.data_vars:
        if node[var].dims != ("azimuth", "range"):
            continue
        rows = node[var].values[padded]
        if node[var].dtype.kind == "f":
            assert np.isnan(rows).all(), var
        else:
            assert (rows == 0).all(), var


def test_unpadded_time_has_no_fill_value(s16_store):
    """Without padding the time has no ``_FillValue``, like the eager one."""
    assert "_FillValue" not in _open_tree(s16_store)["sweep_0"]["time"].attrs


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


def test_range_attrs_come_from_the_header(cor_hdr):
    """The range attrs are the header's metres, like the eager reader's
    (#529), not values read back from the float32 range: a 123.45 m step is
    123.45, not 123.45000457763672. (The parser writes these attrs; parity
    checks them against the eager reader.)"""
    import dataclasses

    from xradar.io.virtual.iris.format import range_attrs

    attrs = range_attrs(dataclasses.replace(cor_hdr, step_output_bins_cm=12345))
    assert attrs["meters_between_gates"] == 123.45
    # the first gate center is the header's first bin, 30000 cm
    assert cor_hdr.range_first_bin_cm == 30000
    assert attrs["meters_to_center_of_first_gate"] == 300.0


def test_sector_task_is_a_sector_sweep(iris0_file, local_registry, monkeypatch):
    # as the eager reader (#574)
    _with_header(monkeypatch, antenna_scan_mode=1)
    tree = _open_tree(IrisParser()(f"file://{iris0_file}", local_registry))
    assert tree["sweep_0"]["sweep_mode"].values.item() == "sector"


def test_sweep_number_is_the_files_own(iris0_file, local_registry, tmp_path, closing):
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
    truth = closing(open_iris_datatree(str(path)))
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
    "file_fixture, var, identifiers",
    [
        ("iris0_file", "DB_HCLASS", [1, 2, 3, 0, 0, 0]),
        ("iris1_file", "DB_HCLASS2", [1, 2, 3, 1, 2, 3]),
    ],
)
def test_hclass_flag_attrs_match_eager(
    file_fixture, var, identifiers, local_registry, monkeypatch, request, closing
):
    """HydroClass flag attrs (#444) come from the file's task_end_info
    ``echo_class_identifiers`` through the eager helper, on both paths, for
    1- and 2-byte HydroClass."""
    path = request.getfixturevalue(file_fixture)
    # the files store no identifiers: no flags, the raw-words comment stays
    plain = _open_tree(IrisParser()(f"file://{path}", local_registry))
    assert "flag_meanings" not in plain["sweep_0"][var].attrs
    assert "comment" in plain["sweep_0"][var].attrs

    _with_hclass_identifiers(monkeypatch, identifiers)
    virt_attrs = _open_tree(IrisParser()(f"file://{path}", local_registry))["sweep_0"][
        var
    ].attrs
    eager_attrs = closing(open_iris_datatree(path, sweep=[0]))["sweep_0"][var].attrs
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


def _patch_raw_file(monkeypatch, change):
    """Run ``change(raw)`` on every ``IrisRawFile`` after it reads its
    headers. Both readers build one, so the virtual and the eager reader
    see the same edited headers."""
    from xradar.io.backends import iris

    init = iris.IrisRawFile.__init__

    def patched(self, *args, **kwargs):
        init(self, *args, **kwargs)
        change(self)

    monkeypatch.setattr(iris.IrisRawFile, "__init__", patched)


def _with_product_end(monkeypatch, **changes):
    """Edit the product header's product_end (the task configuration stays
    as stored)."""
    _patch_raw_file(
        monkeypatch, lambda raw: raw.product_hdr["product_end"].update(changes)
    )


def _task(raw):
    return raw.ingest_header["task_configuration"]


def _assert_vradh_matches_eager(ds, iris_file):
    """Virtual VRADH equals the eager decode of the same (patched) file."""
    with open_iris_datatree(iris_file, sweep=[0]) as eager:
        theirs = eager["sweep_0"]["VRADH"].sortby("azimuth").values
    mine = ds["VRADH"].sortby("azimuth").values
    assert np.isfinite(mine).sum() > 1000
    # decoded by the eager decoder in xradar-iris-decode: exact
    np.testing.assert_array_equal(mine, theirs)


def test_nyquist_follows_product_end_like_eager(
    iris0_file, local_registry, monkeypatch
):
    """prf and wavelength come from product_end, as in the eager decode,
    even when they differ from the task configuration (500 Hz, 5.33 cm)."""
    _with_product_end(monkeypatch, prf=1000, wavelength=1066)
    store = IrisParser()(f"file://{iris0_file}", local_registry)
    ds = xr.open_datatree(store, engine="zarr", consolidated=False, zarr_format=3)[
        "sweep_0"
    ].ds
    # 10.66 cm * 1000 Hz / 4, no multi-PRF on this task
    assert float(ds["nyquist_velocity"]) == pytest.approx(0.1066 * 1000 / 4)
    _assert_vradh_matches_eager(ds, iris0_file)


@pytest.mark.parametrize("flag, factor", [(1, 2), (7, 1)], ids=["2-3", "undefined"])
def test_multi_prf_nyquist_like_eager(
    flag, factor, iris0_file, local_registry, monkeypatch
):
    """The multi-PRF factor multiplies the 1-byte velocity's nyquist, on top
    of the product_end prf and wavelength; a flag the guide does not define
    (only 0-3) warns and keeps the single-PRF nyquist (#480). Both exactly as
    in the eager decode."""
    _with_product_end(monkeypatch, prf=1000, wavelength=1066)
    _patch_raw_file(
        monkeypatch,
        lambda raw: _task(raw)["task_dsp_info"].update(multi_prf_mode_flag=flag),
    )
    warns = (
        pytest.warns(UserWarning, match=f"multi_prf_mode_flag={flag}")
        if flag > 3
        else contextlib.nullcontext()
    )
    with warns:
        store = IrisParser()(f"file://{iris0_file}", local_registry)
        ds = _open_tree(store, mask_and_scale=True, decode_times=True)["sweep_0"].ds
        _assert_vradh_matches_eager(ds, iris0_file)
    assert float(ds["nyquist_velocity"]) == pytest.approx(factor * 0.1066 * 1000 / 4)


def test_volumes_with_another_prf_do_not_concatenate(
    iris0_file, local_registry, monkeypatch
):
    """Velocity decodes with its volume's nyquist, which is in the decode
    codec's configuration: volumes with another PRF do not concatenate,
    instead of decoding with the first volume's nyquist. Fixed-scale moments
    still do."""
    first = IrisParser()(f"file://{iris0_file}", local_registry)
    _with_product_end(monkeypatch, prf=665)
    other = IrisParser()(f"file://{iris0_file}", local_registry)
    one, two = (s.to_virtual_datatree()["sweep_0"].to_dataset() for s in (first, other))
    with pytest.raises(NotImplementedError, match="different codecs"):
        xr.concat([one["VRADH"], two["VRADH"]], dim="volume")
    xr.concat([one["DBZH"], two["DBZH"]], dim="volume")


def test_header_without_prf_names_the_sweep(iris0_file, local_registry, monkeypatch):
    """A product_end with no PRF gives no velocity scale: the parser refuses,
    naming the sweep and the data type."""
    _with_product_end(monkeypatch, prf=0)
    with pytest.raises(ValueError, match="sweep 1 DB_VEL: .*'nyquist'"):
        IrisParser()(f"file://{iris0_file}", local_registry)


def test_gate_count_mismatch_raises(iris0_file, local_registry, monkeypatch):
    """The data's gate count (product_end) must match the range bins of
    task_range_info; a mismatch raises instead of misaligning range."""
    _with_product_end(monkeypatch, number_bins=600)
    with pytest.raises(ValueError, match="product_end declares 600 gates"):
        IrisParser()(f"file://{iris0_file}", local_registry)


def _file_offset(sweep, word_index):
    """File byte offset of word ``word_index`` of ``sweep_words`` (the
    stream without record headers and the ingest_data_header prologue)."""
    from xradar.io.virtual.iris.format import BHDR_SIZE, IDH_SIZE

    body = RECORD_SIZE - BHDR_SIZE  # bytes of each record after its bhdr
    at = IDH_SIZE * sweep.ndatatypes + 2 * word_index
    return sweep.byte_offset + (at // body) * RECORD_SIZE + BHDR_SIZE + at % body


def _zero_rbins(buf, sweep, group, ordinals):
    """Set the ``rbins`` header word of ray ``group`` to 0 for ``ordinals``."""
    from xradar.io.virtual.iris.format import sweep_words

    span = bytes(buf[sweep.byte_offset : sweep.byte_offset + sweep.byte_length])
    words = sweep_words(span, sweep.ndatatypes)
    i = 0
    for ray in range((group + 1) * sweep.ndatatypes):
        code = int(words[i])
        if ray // sweep.ndatatypes == group and ray % sweep.ndatatypes in ordinals:
            # the 6 header words open the ray's first data run
            assert code < 0 and code + 32768 >= 6
            at = _file_offset(sweep, i + 1 + 4)  # past the code word
            buf[at : at + 2] = b"\x00\x00"
        while code != 1:  # skip to the end-of-ray code
            i += 1 + (code + 32768 if code < 0 else 0)
            code = int(words[i])
        i += 1


def _rbins0_file(iris0_file, tmp_path, ordinals=None):
    """cor-main with ray 10 of sweep 1 declaring 0 bins in ``ordinals``
    (default: every data type)."""
    from xradar.io.backends.iris import INGEST_DATA_HEADER
    from xradar.io.virtual.iris.format import BHDR_SIZE, IDH_SIZE

    buf = bytearray(Path(iris0_file).read_bytes())
    sweep = read_volume(bytes(buf)).sweeps[0]
    if ordinals is None:
        ordinals = range(sweep.ndatatypes)
    _zero_rbins(buf, sweep, group=10, ordinals=ordinals)
    # the eager reader sizes its arrays by the written-ray count
    field = _field_offset(INGEST_DATA_HEADER, "number_rays_file_written")
    for t in range(sweep.ndatatypes):
        at = sweep.byte_offset + BHDR_SIZE + t * IDH_SIZE + field
        written = int.from_bytes(buf[at : at + 2], "little")
        buf[at : at + 2] = (written - 1).to_bytes(2, "little")
    path = tmp_path / "rbins0.RAW"
    path.write_bytes(bytes(buf))
    return path


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")  # eager sqrt
def test_ray_with_zero_rbins_is_missing(
    iris0_file, local_registry, tmp_path, closing, cor_truth
):
    """A ray whose header declares 0 bins is dropped, as the eager reader
    drops every ray with rbins == 0."""
    path = _rbins0_file(iris0_file, tmp_path)
    ds = _open_tree(IrisParser()(f"file://{path}", local_registry))["sweep_0"].ds
    eager = closing(open_iris_datatree(str(path), sweep=[0]))["sweep_0"].ds
    full = cor_truth["sweep_0"].ds
    assert ds.sizes["azimuth"] == full.sizes["azimuth"] - 1
    _assert_sweep_parity(ds, eager)


def test_zero_rbins_ray_keeps_its_slot_when_padded(
    iris0_file, local_registry, tmp_path
):
    path = _rbins0_file(iris0_file, tmp_path)
    ds = _open_tree(
        IrisParser(pad_missing_rays=True)(f"file://{path}", local_registry)
    )["sweep_0"].ds
    full = _open_tree(IrisParser()(f"file://{iris0_file}", local_registry))[
        "sweep_0"
    ].ds
    assert ds.sizes["azimuth"] == full.sizes["azimuth"]
    assert np.isnan(ds["azimuth"].values[10])
    assert (ds["DBZH"].values[10] == 0).all()  # raw fill row
    np.testing.assert_array_equal(
        np.delete(ds["DBZH"].values, 10, axis=0),
        np.delete(full["DBZH"].values, 10, axis=0),
    )


def test_zero_rbins_in_one_type_only_is_refused(iris0_file, local_registry, tmp_path):
    """When only some data types drop a ray, rows would misalign between
    moments, so the sweep is refused."""
    path = _rbins0_file(iris0_file, tmp_path, ordinals=[1])
    with pytest.raises(ValueError, match="rows would misalign"):
        IrisParser()(f"file://{path}", local_registry)


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
    iris0_file, local_registry, monkeypatch, description, closing
):
    """The task description reaches the root ``comment`` as the eager reader
    writes it: no stripping; bytes that are not UTF-8 decode as Latin-1."""
    _patch_raw_file(
        monkeypatch,
        lambda raw: _task(raw)["task_end_info"].update(task_description=description),
    )
    tree = _open_tree(IrisParser()(f"file://{iris0_file}", local_registry))
    eager = closing(open_iris_datatree(iris0_file, sweep=[0])).attrs["comment"]
    if isinstance(eager, bytes):
        eager = eager.decode("latin-1")
    assert tree.attrs["comment"] == eager
