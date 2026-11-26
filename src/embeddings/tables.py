"""Streaming Parquet output for embedding tables.

Vectors are stored as Arrow ``FixedSizeList<float32>[D]`` columns, so a table of N embeddings
is one contiguous ``N x D`` buffer on disk. Reading them back::

    table = pd.read_parquet("embeddings.parquet")
    matrix = np.stack(table["embedding"].to_numpy())      # [N, D] float32
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

VECTOR = "vector"  # column kind for 1-D float arrays of a fixed length


def shard_path(path: str | Path, index: int, count: int) -> Path:
    """``embeddings.parquet`` -> ``embeddings.shard-003-of-008.parquet``; unchanged for a single shard.

    The tasks of a Slurm job array each write their own shard file, so they never collide;
    ``mpn_al.embeddings.merge`` joins them afterwards.
    """
    path = Path(path)
    if count <= 1:
        return path
    return path.with_name(f"{path.stem}.shard-{index:03d}-of-{count:03d}{path.suffix}")


class ParquetStreamWriter:
    """Append rows to one Parquet file in chunks.

    Rows are written to ``<path>.partial`` and the file takes its final name only on a clean
    :meth:`close`, so an interrupted job never leaves a truncated table behind under the
    expected name.

    ``columns`` maps column name to ``"string"``, ``"int64"``, ``"float32"``, ``"float64"``,
    ``"bool"`` or :data:`VECTOR`. Missing values are ``None``.
    """

    def __init__(
        self,
        path: str | Path,
        columns: Mapping[str, str],
        *,
        flush_rows: int = 2000,
        vector_sizes: Mapping[str, int] | None = None,
    ):
        self.path = Path(path)
        self.columns = dict(columns)
        self.flush_rows = flush_rows
        # Known vector lengths; needed only when a table may stay empty, so that all files of a
        # dataset directory share one schema.
        self.vector_sizes = dict(vector_sizes or {})
        self._rows: list[Mapping[str, Any]] = []
        self._writer = None
        self._schema = None
        self._partial = self.path.with_name(self.path.name + ".partial")
        self.rows_written = 0

    def add(self, row: Mapping[str, Any]) -> None:
        missing = set(self.columns) - set(row)
        if missing:
            raise KeyError(f"row lacks columns {sorted(missing)}")
        self._rows.append(row)
        if len(self._rows) >= self.flush_rows:
            self._flush()

    def close(self) -> Path:
        """Write what is buffered and move the file to its final name. Safe to call on an empty table."""
        self._flush(final=True)
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._partial.replace(self.path)
        return self.path

    def __enter__(self) -> "ParquetStreamWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        elif self._writer is not None:
            self._writer.close()  # keep the .partial file for inspection, never promote it

    def _flush(self, final: bool = False) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not self._rows and not (final and self._writer is None):
            return
        if self._schema is None:
            self._schema = self._build_schema(pa)
        arrays = []
        for name, kind in self.columns.items():
            values = [row[name] for row in self._rows]
            field_type = self._schema.field(name).type
            if kind == VECTOR and pa.types.is_fixed_size_list(field_type):
                size = field_type.list_size
                flat = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in values])
                if flat.size != size * len(values):
                    raise ValueError(f"column {name!r}: every vector must have length {size}")
                arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), size))
            else:
                arrays.append(pa.array(values, type=field_type))
        table = pa.Table.from_arrays(arrays, schema=self._schema)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(str(self._partial), self._schema, compression="zstd")
        self._writer.write_table(table)
        self.rows_written += len(self._rows)
        self._rows = []

    def _build_schema(self, pa: Any) -> Any:
        scalar = {"string": pa.string(), "int64": pa.int64(), "float32": pa.float32(), "float64": pa.float64(), "bool": pa.bool_()}
        fields = []
        for name, kind in self.columns.items():
            if kind == VECTOR:
                first = next((row[name] for row in self._rows if row[name] is not None), None)
                if name in self.vector_sizes:
                    size = self.vector_sizes[name]
                else:  # an empty table has no vector to size the column by: plain list type
                    size = int(np.asarray(first).size) if first is not None else -1
                fields.append(pa.field(name, pa.list_(pa.float32(), size)))
            elif kind in scalar:
                fields.append(pa.field(name, scalar[kind]))
            else:
                raise ValueError(f"column {name!r}: unknown kind {kind!r}")
        return pa.schema(fields)
