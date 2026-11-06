"""Minimal Mapbox Vector Tile reader: layer names and feature *properties* only (no geometry).

Mapillary's coverage tiles are standard MVT files, which are Protocol Buffers messages. We only
need each image's properties (id, sequence_id, creator_id, is_pano, ...), so instead of adding a
dependency we read the few protobuf fields we need. The wire format in one paragraph:

    a message is a sequence of (key, value) pairs; key = (field_number << 3) | wire_type;
    wire type 0 = varint, 1 = 8 fixed bytes, 2 = length-prefixed bytes, 5 = 4 fixed bytes.

The MVT schema (https://github.com/mapbox/vector-tile-spec/tree/master/2.1) uses::

    Tile    { repeated Layer layers = 3; }
    Layer   { string name = 1; repeated Feature features = 2; repeated string keys = 3;
              repeated Value values = 4; uint32 extent = 5; uint32 version = 15; }
    Feature { uint64 id = 1; repeated uint32 tags = 2 (packed); GeomType type = 3;
              repeated uint32 geometry = 4 (packed); }
    Value   { string = 1; float = 2; double = 3; int64 = 4; uint64 = 5; sint64 = 6; bool = 7; }

A feature's ``tags`` are pairs (key index, value index) into the layer's ``keys`` and ``values``.
"""

from __future__ import annotations

import gzip
import struct
from collections.abc import Iterable, Iterator
from typing import Any


class MVTError(ValueError):
    """The bytes are not a valid vector tile."""


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise MVTError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise MVTError("varint too long")


def _fields(buf: bytes) -> Iterator[tuple[int, int, Any]]:
    """Yield (field number, wire type, value) for each field of one message.

    The value is an int for varints and ``bytes`` for the other wire types."""
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1 or wire == 5:
            size = 8 if wire == 1 else 4
            value, pos = buf[pos : pos + size], pos + size
            if len(value) != size:
                raise MVTError("truncated fixed-width field")
        elif wire == 2:
            size, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
            if len(value) != size:
                raise MVTError("truncated length-delimited field")
        else:
            raise MVTError(f"unsupported wire type {wire}")
        yield field, wire, value


def _value(buf: bytes) -> Any:
    """Decode a ``Value`` message (it holds exactly one field)."""
    for field, _, raw in _fields(buf):
        if field == 1:
            return raw.decode("utf-8")
        if field == 2:
            return struct.unpack("<f", raw)[0]
        if field == 3:
            return struct.unpack("<d", raw)[0]
        if field == 4:  # int64: two's complement stored as a 64-bit varint
            return raw - (1 << 64) if raw >= 1 << 63 else raw
        if field == 5:
            return raw
        if field == 6:  # sint64: zigzag
            return (raw >> 1) ^ -(raw & 1)
        if field == 7:
            return bool(raw)
    return None


def _layer_name(buf: bytes) -> str:
    for field, _, raw in _fields(buf):
        if field == 1:
            return raw.decode("utf-8")
    return ""


def _layer_features(buf: bytes) -> list[dict[str, Any]]:
    keys: list[str] = []
    values: list[Any] = []
    raw_features: list[bytes] = []
    for field, _, raw in _fields(buf):
        if field == 2:
            raw_features.append(raw)
        elif field == 3:
            keys.append(raw.decode("utf-8"))
        elif field == 4:
            values.append(_value(raw))

    features = []
    for raw in raw_features:
        feature_id: int | None = None
        tags: list[int] = []
        for field, wire, item in _fields(raw):
            if field == 1:
                feature_id = item
            elif field == 2:
                if wire == 2:  # packed (the normal case)
                    pos = 0
                    while pos < len(item):
                        tag, pos = _varint(item, pos)
                        tags.append(tag)
                else:
                    tags.append(item)
        try:
            props = {keys[k]: values[v] for k, v in zip(tags[0::2], tags[1::2], strict=False)}
        except IndexError:
            raise MVTError("feature tag points outside the layer's keys/values") from None
        features.append({"id": feature_id, "properties": props})
    return features


def decode_tile(
    content: bytes, layers: Iterable[str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{layer name: [{"id", "properties"}, ...]}``. ``layers`` limits which layers are
    decoded (the others are skipped without reading their features)."""
    if content[:2] == b"\x1f\x8b":  # gzip-compressed tile
        content = gzip.decompress(content)
    wanted = None if layers is None else set(layers)
    out: dict[str, list[dict[str, Any]]] = {}
    for field, _, raw in _fields(content):
        if field != 3:
            continue
        name = _layer_name(raw)
        if wanted is None or name in wanted:
            out.setdefault(name, []).extend(_layer_features(raw))
    return out
