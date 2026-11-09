"""Coverage-tile helpers and the vector-tile reader. No network: tiles are encoded by hand."""

from __future__ import annotations

import gzip
import struct

import pytest
import requests

from data.mvt import MVTError, decode_tile
from data.tiles import (
    MapillaryTiles,
    TileError,
    parse_image_layer,
    tile_for,
    tiles_around,
)


# --- a tiny Mapbox Vector Tile *encoder*, so the reader is tested on known bytes -------------
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | 0x80 if n else byte)
        if not n:
            return bytes(out)


def _field(num: int, wire: int, payload) -> bytes:
    key = _varint(num << 3 | wire)
    if wire == 0:
        return key + _varint(payload)
    if wire == 2:
        return key + _varint(len(payload)) + payload
    return key + payload  # wire 1 or 5: fixed-width bytes


def _value(v) -> bytes:
    if isinstance(v, bool):
        return _field(7, 0, int(v))
    if isinstance(v, str):
        return _field(1, 2, v.encode())
    if isinstance(v, float):
        return _field(3, 1, struct.pack("<d", v))
    return _field(4, 0, v & (2**64 - 1))  # int64 (two's complement)


def make_tile(layers: dict[str, list[dict]]) -> bytes:
    tile = b""
    for name, features in layers.items():
        keys: list[str] = []
        values: list[bytes] = []
        feats = b""
        for props in features:
            tags = b""
            for k, v in props.items():
                if k not in keys:
                    keys.append(k)
                enc = _value(v)
                if enc not in values:
                    values.append(enc)
                tags += _varint(keys.index(k)) + _varint(values.index(enc))
            feats += _field(2, 2, _field(2, 2, tags) + _field(3, 0, 1))  # one POINT feature
        body = _field(1, 2, name.encode()) + feats
        body += b"".join(_field(3, 2, k.encode()) for k in keys)
        body += b"".join(_field(4, 2, v) for v in values)
        body += _field(5, 0, 4096) + _field(15, 0, 2)
        tile += _field(3, 2, body)
    return tile


IMAGES = [
    {
        "id": 1_000_000_000_000_001,  # Mapillary ids need a full int64
        "sequence_id": "seqA",
        "creator_id": 42,
        "is_pano": True,
        "captured_at": 1_700_000_000_000,
        "compass_angle": 12.5,
    },
    {"id": 1_000_000_000_000_002, "sequence_id": "seqA", "creator_id": 42, "is_pano": False},
    {"id": 1_000_000_000_000_003, "sequence_id": "seqB", "creator_id": 7, "is_pano": True},
]


# --- tile math ------------------------------------------------------------------------------
def test_tile_for_known_points():
    assert tile_for(52.370, 4.895, 14) == (8414, 5384)  # Amsterdam
    assert tile_for(0.0, 0.0, 1) == (1, 1)  # the equator / prime meridian corner
    # Sydney (southern + eastern hemisphere), cross-checked with the OSM wiki's ln(tan+sec) form
    assert tile_for(-33.869, 151.209, 14) == (15_073, 9_831)
    assert tile_for(-0.180, -78.468, 14) == (4_620, 8_200)  # Quito, just south of the equator


def test_tile_for_stays_on_the_grid():
    n = 2**14
    assert tile_for(89.999, 179.9999, 14)[0] == n - 1  # east edge
    assert tile_for(0.0, -180.0, 14)[0] == 0  # west edge
    x, y = tile_for(-89.999, 0.0, 14)
    assert 0 <= y <= n - 1 and 0 <= x <= n - 1


def test_y_grows_southwards():
    assert tile_for(10.0, 0.0, 14)[1] < tile_for(-10.0, 0.0, 14)[1]


def test_tiles_around_block():
    block = tiles_around(100, 200, 14, 1)
    assert len(block) == 9 and len(set(block)) == 9
    assert block[0] == (99, 199) and block[-1] == (101, 201)  # row by row, west to east
    assert tiles_around(100, 200, 14, 0) == [(100, 200)]


def test_tiles_around_clips_at_the_grid_edge():
    assert len(tiles_around(0, 0, 14, 1)) == 4
    n = 2**14
    assert (n, 5) not in tiles_around(n - 1, 5, 14, 2)


# --- the vector-tile reader -------------------------------------------------------------------
def test_decode_tile_reads_layers_and_values():
    tile = make_tile({"image": IMAGES, "sequence": [{"id": "s1", "ratio": 0.5}]})
    out = decode_tile(tile)
    assert set(out) == {"image", "sequence"}
    first = out["image"][0]["properties"]
    assert first["id"] == 1_000_000_000_000_001
    assert first["sequence_id"] == "seqA" and first["is_pano"] is True
    assert first["compass_angle"] == pytest.approx(12.5)
    assert out["sequence"][0]["properties"] == {"id": "s1", "ratio": 0.5}


def test_decode_tile_layer_filter_and_gzip():
    tile = make_tile({"image": IMAGES, "sequence": [{"id": "s1"}]})
    assert set(decode_tile(tile, ["image"])) == {"image"}
    assert len(decode_tile(gzip.compress(tile), ["image"])["image"]) == 3


def test_decode_tile_rejects_garbage():
    with pytest.raises(MVTError):
        decode_tile(make_tile({"image": IMAGES})[:-5])  # truncated mid-field


def test_decode_negative_int64():
    out = decode_tile(make_tile({"image": [{"v": -7}]}))
    assert out["image"][0]["properties"]["v"] == -7


# --- from tile features to candidates ---------------------------------------------------------
def test_parse_image_layer():
    feats = decode_tile(make_tile({"image": IMAGES}))["image"]
    cands = parse_image_layer(feats)
    assert [c["id"] for c in cands] == [
        "1000000000000001",
        "1000000000000002",
        "1000000000000003",
    ]
    assert cands[0] == {
        "id": "1000000000000001",
        "sequence_id": "seqA",
        "creator_id": "42",
        "is_pano": True,
        "captured_at": 1_700_000_000_000,
    }
    assert cands[1]["is_pano"] is False


def test_parse_image_layer_float_ids_and_missing_fields():
    feats = [
        {"properties": {"id": 1.0e15, "sequence_id": "s", "creator_id": 3.0, "is_pano": True}},
        {"properties": {"id": 5, "is_pano": True}},  # no sequence -> dropped
        {"properties": {"sequence_id": "s", "is_pano": True}},  # no id -> dropped
    ]
    cands = parse_image_layer(feats)
    assert len(cands) == 1
    assert cands[0]["id"] == "1000000000000000" and cands[0]["creator_id"] == "3"


class _Resp:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code, self.content = status, content


def _tiles(responses, max_retries=1):
    t = MapillaryTiles(token="SECRET", max_retries=max_retries)
    it = iter(responses)

    def fake_get(url, params=None, timeout=None):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    t.session.get = fake_get
    return t


def test_fetch_keeps_only_panoramas():
    tile = make_tile({"image": IMAGES, "sequence": [{"id": "x"}]})
    cands = _tiles([_Resp(200, tile)]).fetch(14, 1, 2)
    assert [c["id"] for c in cands] == ["1000000000000001", "1000000000000003"]


@pytest.mark.parametrize("status", [204, 404])
def test_fetch_empty_tile(status):
    assert _tiles([_Resp(status)]).fetch(14, 1, 2) == []


def test_fetch_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("data.tiles.time.sleep", lambda s: None)
    tile = make_tile({"image": IMAGES})
    t = _tiles([_Resp(503), _Resp(502), _Resp(200, tile)], max_retries=3)
    assert len(t.fetch(14, 1, 2)) == 2


def test_fetch_errors_never_contain_the_token():
    boom = requests.ConnectionError("pool: /2/14/1/2?access_token=SECRET refused")
    with pytest.raises(TileError) as exc:
        _tiles([boom]).fetch(14, 1, 2)
    assert "SECRET" not in str(exc.value)
    with pytest.raises(TileError, match="HTTP 403") as exc:
        _tiles([_Resp(403)]).fetch_raw(14, 1, 2)
    assert "SECRET" not in str(exc.value)


def test_tiles_need_a_token(monkeypatch):
    monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
    monkeypatch.setattr("data.tiles.load_dotenv", lambda: None)
    with pytest.raises(RuntimeError, match="MAPILLARY_TOKEN"):
        MapillaryTiles()
