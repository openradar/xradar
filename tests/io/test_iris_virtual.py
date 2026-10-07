#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.virtual.iris` (format walker + ``xradar-iris-sweep`` codec).

The oracle for decode parity is the eager IRIS backend in this same repo.
Ray-order note: the format walker emits rays in acquisition (stream) order
while the eager reader places rays by azimuth-derived index, so a sweep
whose first ray straddles north differs by a cyclic roll — parity is
therefore asserted on azimuth-ALIGNED rows, not positionally.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("zarr", minversion="3.1.6")  # zarr v3 only; numpy-1 cells skip

import xradar  # resolves the package location for the subprocess test
from xradar.io.backends.iris import SIGMET_DATA_TYPES, decode_array
from xradar.io.virtual.iris.codec import CODEC_NAME, IrisSweepCodec
from xradar.io.virtual.iris.format import (
    BHDR_SIZE,
    IDH_SIZE,
    INGEST_DATA_HEADER_ID,
    INGEST_HEADER_ID,
    PRODUCT_HDR_ID,
    RAY_HEADER_WORDS,
    RECORD_SIZE,
    azimuth_midpoints,
    azimuth_sort_order,
    decode_sweep_moment,
    index_sweeps,
    parse_ingest_header,
    range_centers,
    sweep_words,
    walk_sweep,
)


def test_framing_constants_derive_from_iris_structs():
    """The derived sizes/ids/offsets must equal the IRIS Programmer's
    Manual values — pins the struct-dict derivation to the documented
    format."""
    assert RECORD_SIZE == 6144
    assert BHDR_SIZE == 12
    assert IDH_SIZE == 76
    assert RAY_HEADER_WORDS == 6
    assert (PRODUCT_HDR_ID, INGEST_HEADER_ID, INGEST_DATA_HEADER_ID) == (27, 23, 24)


def test_index_sweeps_corozal(iris0_file):
    """10-sweep 8-bit IDEAM Corozal volume: sweeps tile the file exactly."""
    buf = Path(iris0_file).read_bytes()
    hdr = parse_ingest_header(buf)
    sweeps = index_sweeps(buf)

    assert hdr.task_name == "SURV_HV_300"
    assert "Corozal" in hdr.site_name
    assert len(sweeps) == 10
    total = 2 * RECORD_SIZE
    for s in sweeps:
        assert s.byte_offset % RECORD_SIZE == 0
        assert s.byte_length % RECORD_SIZE == 0
        assert s.ndatatypes == 7
        total += s.byte_length
    assert sweeps[0].byte_offset == 2 * RECORD_SIZE
    assert total == len(buf)
    types = [h.type_name for h in sweeps[0].headers]
    assert types[0] == "DB_DBZ"
    assert "DB_XHDR" not in types


def test_index_sweeps_surgavere_16bit(iris1_file):
    """Single-sweep all-16-bit volume with DB_XHDR and one missing ray."""
    buf = Path(iris1_file).read_bytes()
    sweeps = index_sweeps(buf)

    assert len(sweeps) == 1
    (sweep,) = sweeps
    assert sweep.ndatatypes == 12
    types = [h.type_name for h in sweep.headers]
    assert types[0] == "DB_XHDR"
    assert "DB_DBZ2" in types
    data_headers = [h for h in sweep.headers if h.type_name != "DB_XHDR"]
    assert all(h.bits_per_bin == 16 for h in data_headers)
    assert sweep.headers[0].nrays_expected == 360

    span = buf[sweep.byte_offset : sweep.byte_offset + sweep.byte_length]
    _, headers, missing = walk_sweep(
        sweep_words(span, sweep.ndatatypes), sweep.ndatatypes, None
    )
    assert len(headers) == 359  # one missing ray, compacted
    assert all(len(groups) == 1 for groups in missing.values())


def _decoded_rows_and_azimuths(buf, sweeps, sweep_idx, type_name, nbins):
    """Decode one moment of one sweep plus its per-ray azimuth midpoints."""
    sweep = sweeps[sweep_idx]
    types = [h.type_name for h in sweep.headers]
    ordinal = types.index(type_name)
    dth = sweep.headers[ordinal]
    dtype = np.dtype("uint8") if dth.bits_per_bin == 8 else np.dtype("uint16")
    span = buf[sweep.byte_offset : sweep.byte_offset + sweep.byte_length]
    _, headers, _ = walk_sweep(
        sweep_words(span, sweep.ndatatypes), sweep.ndatatypes, None
    )
    rows = decode_sweep_moment(
        span, ordinal, sweep.ndatatypes, (len(headers), nbins), dtype
    )
    azimuths = azimuth_midpoints(
        [h.azimuth_start for h in headers], [h.azimuth_stop for h in headers]
    )
    return rows, azimuths, dth.type_code


def _assert_parity_az_aligned(rows, azimuths, type_code, eager):
    """Bin-for-bin parity against the eager decode, aligned on azimuth."""
    entry = SIGMET_DATA_TYPES[type_code]
    # this helper only understands plain linear types; pointing it at a
    # nyquist/nonlinear type must fail loudly, not silently misdecode
    assert entry.get("func") is decode_array, entry
    with np.errstate(invalid="ignore"):
        mine = decode_array(rows.astype("float64"), **entry.get("fkw", {}))
    theirs = eager.values.astype("float64")
    assert mine.shape == theirs.shape

    ours_order = azimuth_sort_order(azimuths)
    theirs_order = np.argsort(eager["azimuth"].values, kind="stable")
    np.testing.assert_allclose(
        azimuths[ours_order],
        eager["azimuth"].values[theirs_order],
        atol=1e-6,
    )
    mine = mine[ours_order]
    theirs = theirs[theirs_order]
    both = np.isfinite(mine) & np.isfinite(theirs)
    assert both.any()
    np.testing.assert_allclose(mine[both], theirs[both], atol=1e-5)


def test_decode_moment_parity_corozal(iris0_file):
    buf = Path(iris0_file).read_bytes()
    hdr = parse_ingest_header(buf)
    sweeps = index_sweeps(buf)
    rows, azimuths, code = _decoded_rows_and_azimuths(
        buf, sweeps, 0, "DB_DBZ", hdr.number_output_bins
    )
    with xr.open_dataset(iris0_file, engine="iris", group="sweep_0") as ds:
        _assert_parity_az_aligned(rows, azimuths, code, ds["DBZH"])


def test_decode_moment_parity_surgavere_16bit(iris1_file):
    buf = Path(iris1_file).read_bytes()
    hdr = parse_ingest_header(buf)
    sweeps = index_sweeps(buf)
    rows, azimuths, code = _decoded_rows_and_azimuths(
        buf, sweeps, 0, "DB_DBZ2", hdr.number_output_bins
    )
    with xr.open_dataset(iris1_file, engine="iris", group="sweep_0") as ds:
        _assert_parity_az_aligned(rows, azimuths, code, ds["DBZH"])


def test_codec_from_dict_strict():
    with pytest.raises(ValueError, match="requires"):
        IrisSweepCodec.from_dict(
            {"name": CODEC_NAME, "configuration": {"moment_index": 1}}
        )
    with pytest.raises(ValueError, match="expected codec name"):
        IrisSweepCodec.from_dict(
            {"name": "gzip", "configuration": {"moment_index": 1, "ndatatypes": 2}}
        )
    codec = IrisSweepCodec.from_dict(
        {"name": CODEC_NAME, "configuration": {"moment_index": 1, "ndatatypes": 12}}
    )
    assert codec.sort_rays is False
    assert codec.pad_missing_rays is False
    with pytest.raises(ValueError, match="out of range"):
        IrisSweepCodec(moment_index=12, ndatatypes=12)


def test_codec_to_dict_round_trip():
    codec = IrisSweepCodec(
        moment_index=3, ndatatypes=12, sort_rays=True, pad_missing_rays=True
    )
    assert IrisSweepCodec.from_dict(codec.to_dict()) == codec
    assert set(codec.to_dict()["configuration"]) == {
        "moment_index",
        "ndatatypes",
        "sort_rays",
        "pad_missing_rays",
    }


def test_codec_config_is_frozen():
    """The serialized config is a public contract: every published store
    carries it verbatim, so the key set and order must never change (this
    configuration is byte-identical to the original raw2zarr implementation's;
    only the name gained the ``xradar-`` prefix)."""
    codec = IrisSweepCodec(moment_index=3, ndatatypes=12, sort_rays=True)
    assert json.dumps(codec.to_dict()) == (
        '{"name": "xradar-iris-sweep", "configuration": {"moment_index": 3, '
        '"ndatatypes": 12, "sort_rays": true, "pad_missing_rays": false}}'
    )


def test_codec_resolves_from_zarr_registry():
    """``xradar-iris-sweep`` (and its legacy alias) resolve from the zarr registry. When several
    providers of the name coexist (e.g. a migration window where another
    package still registers it), zarr warns and picks one arbitrarily —
    ``zarr.config`` pins the xradar implementation deterministically."""
    import warnings

    import zarr
    from zarr.registry import get_codec_class

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duplicate-provider warning is env-dependent
        assert get_codec_class(CODEC_NAME).__name__ == "IrisSweepCodec"

    fqcn = f"{IrisSweepCodec.__module__}.{IrisSweepCodec.__qualname__}"
    with zarr.config.set({"codecs": {CODEC_NAME: fqcn}}):
        assert get_codec_class(CODEC_NAME) is IrisSweepCodec


def test_zarr_end_to_end_read(iris1_file, tmp_path):
    """A hand-written zarr v3 store whose single chunk is a raw sweep span
    decodes through the public zarr API purely via the registered codec."""
    buf = Path(iris1_file).read_bytes()
    hdr = parse_ingest_header(buf)
    (sweep,) = index_sweeps(buf)
    types = [h.type_name for h in sweep.headers]
    ordinal = types.index("DB_DBZ2")
    span = buf[sweep.byte_offset : sweep.byte_offset + sweep.byte_length]
    _, headers, _ = walk_sweep(
        sweep_words(span, sweep.ndatatypes), sweep.ndatatypes, None
    )
    nrays, nbins = len(headers), hdr.number_output_bins

    codec = IrisSweepCodec(moment_index=ordinal, ndatatypes=sweep.ndatatypes)
    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [nrays, nbins],
        "data_type": "uint16",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [nrays, nbins]},
        },
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [codec.to_dict()],
    }
    (tmp_path / "zarr.json").write_text(json.dumps(metadata))
    chunk_dir = tmp_path / "c" / "0"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "0").write_bytes(span)

    import zarr

    arr = zarr.open_array(store=str(tmp_path), mode="r")
    expected = decode_sweep_moment(
        span, ordinal, sweep.ndatatypes, (nrays, nbins), np.dtype("uint16")
    )
    np.testing.assert_array_equal(arr[:], expected)


def test_codec_import_does_not_pull_virtualizarr():
    """Resolving the codec (what a reader does) must not import
    virtualizarr — reading virtual stores needs only xradar + zarr."""
    repo_root = Path(xradar.__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import xradar.io.virtual.iris.codec\n"
        "assert 'virtualizarr' not in sys.modules\n"
        "print('CODEC_IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "CODEC_IMPORT_OK" in result.stdout


def test_io_import_does_not_pull_virtual():
    """``import xradar.io`` (every eager-reader session) must not import the
    virtual subpackage, nor zarr through it."""
    repo_root = Path(xradar.__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import xradar.io\n"
        "assert 'xradar.io.virtual' not in sys.modules\n"
        "import xradar.io.virtual\n"  # the lazy package itself stays light
        "assert 'virtualizarr' not in sys.modules\n"
        "print('IO_IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "IO_IMPORT_OK" in result.stdout


def test_range_centers_matches_eager(iris0_file, iris1_file):
    """The mirrored range-gate construction equals the eager backend's."""
    for path in (iris0_file, iris1_file):
        hdr = parse_ingest_header(Path(path).read_bytes())
        with xr.open_dataset(path, engine="iris", group="sweep_0") as ds:
            np.testing.assert_allclose(
                range_centers(hdr), ds["range"].values, atol=1e-3
            )


def test_decode_flavors_and_shape_guard(iris1_file):
    """sort_rays / pad_missing_rays row layouts, and the manifest/file
    mismatch guard, at the format level (SUR has one real missing ray)."""
    buf = Path(iris1_file).read_bytes()
    hdr = parse_ingest_header(buf)
    (sweep,) = index_sweeps(buf)
    ndt = sweep.ndatatypes
    ordinal = [h.type_name for h in sweep.headers].index("DB_DBZ2")
    span = buf[sweep.byte_offset : sweep.byte_offset + sweep.byte_length]
    _, headers, missing = walk_sweep(sweep_words(span, ndt), ndt, None)
    nbins = hdr.number_output_bins
    n_written = len(headers)  # 359
    n_expected = sweep.headers[0].nrays_expected  # 360
    dtype = np.dtype("uint16")

    plain = decode_sweep_moment(span, ordinal, ndt, (n_written, nbins), dtype)

    # sorted flavor = plain rows under the shared azimuth permutation
    srt = decode_sweep_moment(
        span, ordinal, ndt, (n_written, nbins), dtype, sort_rays=True
    )
    key = azimuth_midpoints(
        [h.azimuth_start for h in headers], [h.azimuth_stop for h in headers]
    )
    np.testing.assert_array_equal(srt, plain[azimuth_sort_order(key)])

    # padded flavor keeps every slot; the missing slot reads as fill
    padded = decode_sweep_moment(
        span, ordinal, ndt, (n_expected, nbins), dtype, pad_missing_rays=True
    )
    (missing_slot,) = missing[ordinal]
    assert (padded[missing_slot] == 0).all()
    np.testing.assert_array_equal(np.delete(padded, missing_slot, axis=0), plain)

    # wrong declared shape fails loudly in both flavors
    with pytest.raises(ValueError, match="manifest/file mismatch"):
        decode_sweep_moment(span, ordinal, ndt, (n_expected, nbins), dtype)
    with pytest.raises(ValueError, match="manifest/file mismatch"):
        decode_sweep_moment(
            span, ordinal, ndt, (n_written, nbins), dtype, pad_missing_rays=True
        )


def test_codec_constructor_keeps_the_contract():
    """numpy ordinals (e.g. from np.arange) serialize as plain JSON ints;
    strings never pass as flags."""
    codec = IrisSweepCodec(moment_index=np.int64(2), ndatatypes=np.int64(5))
    assert json.dumps(codec.to_dict())
    assert IrisSweepCodec.from_dict(codec.to_dict()) == codec
    with pytest.raises(ValueError, match="must be bool"):
        IrisSweepCodec(moment_index=0, ndatatypes=2, sort_rays="false")
    with pytest.raises(ValueError, match="must be int"):
        IrisSweepCodec(moment_index=True, ndatatypes=2)


@pytest.mark.parametrize(
    "config, match",
    [
        ({"moment_index": 1, "ndatatypes": 2, "sort_rays": "false"}, "must be bool"),
        (
            {"moment_index": 1, "ndatatypes": 2, "reorder": "azimuth"},
            "unknown configuration keys",
        ),
        ({"moment_index": True, "ndatatypes": 2}, "must be int"),
        ({"moment_index": "1", "ndatatypes": 2}, "must be int"),
        ({"moment_index": 1.0, "ndatatypes": 2}, "must be int"),
    ],
    ids=["flag-string", "unknown-key", "bool-index", "str-index", "float-index"],
)
def test_codec_from_dict_rejects_out_of_contract_configs(config, match):
    """A newer or corrupt store must fail loudly, never decode in the wrong
    row order (bool("false") is True; True is an int)."""
    with pytest.raises(ValueError, match=match):
        IrisSweepCodec.from_dict({"name": CODEC_NAME, "configuration": config})


@pytest.mark.parametrize("fill", [float("inf"), float("nan"), 0.5, "0", True, None])
def test_fill_value_must_be_an_integer(fill):
    """Moments are raw unsigned words: a non-integral fill value raises
    ValueError (never OverflowError, never silent truncation). One rule for
    every virtual codec (``_checks.check_output``)."""
    from xradar.io.virtual._checks import check_output

    with pytest.raises(ValueError, match="not an integer"):
        check_output("uint8", fill, (2, 3))
    assert check_output("uint8", np.uint8(7), (2, 3))[1] == 7
    assert check_output("uint16", 3.0, (2, 3))[1] == 3  # JSON float fills


@pytest.mark.parametrize(
    "dtype, fill, shape, match",
    [
        ("int16", 0, (2, 3), "uint8/uint16"),
        (">u2", 0, (2, 3), "uint8/uint16"),
        ("uint32", 0, (2, 3), "uint8/uint16"),
        ("uint8", 256, (2, 3), "not a valid uint8"),
        ("uint8", -1, (2, 3), "not a valid uint8"),
        ("uint8", 0, (0, 3), "non-empty"),
        ("uint8", 0, (3,), "non-empty"),
    ],
)
def test_output_contract_is_shared(dtype, fill, shape, match):
    """Both codecs refuse the same out-of-contract arrays (dtype, fill
    range, shape) before decoding a byte."""
    from xradar.io.virtual._checks import check_output

    with pytest.raises(ValueError, match=match):
        check_output(dtype, fill, shape)


def test_legacy_codec_name_still_reads():
    """Stores published as ``sigmet-sweep`` keep reading (read-only alias);
    re-serialising writes the canonical name."""
    config = {
        "moment_index": 3,
        "ndatatypes": 12,
        "sort_rays": True,
        "pad_missing_rays": False,
    }
    codec = IrisSweepCodec.from_dict({"name": "sigmet-sweep", "configuration": config})
    assert codec.to_dict() == {"name": "xradar-iris-sweep", "configuration": config}


def test_azimuth_sort_order_is_stable():
    """Codec rows and builder coordinates must apply the SAME permutation,
    tied azimuths included (numpy sorts short inputs stably regardless of
    ``kind``, so use a long one)."""
    assert azimuth_sort_order([1.0, 0.0, 1.0, 0.0]).tolist() == [1, 3, 0, 2]
    many = np.tile([1.0, 0.0, 1.0], 64)
    np.testing.assert_array_equal(
        azimuth_sort_order(many), np.argsort(many, kind="stable")
    )


def test_output_contract_caps_the_chunk_size():
    """A store declaring an absurd chunk shape is refused before anything
    is allocated (a hostile store must not cost the reader gigabytes)."""
    from xradar.io.virtual._checks import MAX_CELLS, check_output

    check_output("uint16", 0, (720, 1840))  # a real super-res sweep
    with pytest.raises(ValueError, match="exceeds"):
        check_output("uint8", 0, (360, MAX_CELLS))
    with pytest.raises(ValueError, match="non-empty"):
        check_output("uint8", 0, (2.5, 3))


def test_lazy_exports_resolve_in_process(monkeypatch):
    """``xradar.io.virtual`` and its subpackages resolve their exports on
    first access, list only what is importable, and explain a missing
    optional dependency instead of failing with a bare ImportError."""
    import importlib

    import xradar.io
    from xradar.io import virtual
    from xradar.io.virtual import iris

    assert xradar.io.virtual is virtual  # lazy attribute of xradar.io
    assert iris.IrisSweepCodec is IrisSweepCodec
    assert "IrisSweepCodec" in dir(iris)
    assert "azimuth_sort_order" in dir(virtual)
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        virtual.Nope

    real_import = importlib.import_module

    def missing(name, *args):
        if name == "xradar.io.virtual.iris.codec":
            raise ImportError("no zarr", name="zarr")
        return real_import(name, *args)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(virtual.MissingDependencyError, match="needs only xradar"):
        virtual.lazy_attribute("xradar.io.virtual", "IrisSweepCodec", virtual._LAZY)


def test_codec_is_decode_only_and_needs_a_mapping():
    """A store's codec entry must be a mapping, and the codec never writes."""
    with pytest.raises(ValueError, match="must be a mapping"):
        IrisSweepCodec.from_dict({"name": CODEC_NAME, "configuration": [1, 2]})
    import asyncio

    codec = IrisSweepCodec(moment_index=0, ndatatypes=1)
    with pytest.raises(NotImplementedError, match="decode-only"):
        codec._encode_sync(None, None)
    with pytest.raises(NotImplementedError, match="decode-only"):
        asyncio.run(codec._encode_single(None, None))
    with pytest.raises(NotImplementedError):
        codec.compute_encoded_size(10, None)


def test_zarr_v2_is_refused_with_a_clear_error():
    """The codecs run on zarr v3 only: on zarr 2.x importing them raises an
    ImportError naming the requirement (not a bare ``No module named
    'zarr.abc'``), which the lazy exports turn into MissingDependencyError."""
    code = (
        "import zarr; zarr.__version__ = '2.18.7'\n"
        "try:\n"
        "    import xradar.io.virtual.iris.codec\n"
        "except ImportError as err:\n"
        "    print(type(err).__name__, err)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(xradar.__file__).parents[1],
    )
    assert "need zarr>=3.1.6 (zarr v3); found zarr 2.18.7" in result.stdout


def test_zarr_version_parsing():
    from xradar.io.virtual._codec import MIN_ZARR, _version_tuple

    assert _version_tuple("3.1.6") == MIN_ZARR
    assert _version_tuple("3.2.0rc1") == (3, 2, 0) > MIN_ZARR
    assert _version_tuple("3.2.1.dev3+g1a2b") == (3, 2, 1)
    assert _version_tuple("2.18.7") < MIN_ZARR
