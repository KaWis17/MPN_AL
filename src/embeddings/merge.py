"""Join the files written by the tasks of a Slurm job array.

``extract_embeddings.py --num-shards N`` writes ``embeddings.shard-000-of-00N.parquet`` and so on
(one file per task, so tasks never collide), and ``segment_cells.py`` writes
``tile_stats.shard-...jsonl``. This module merges them::

    python scripts/merge_shards.py embeddings data/02_active_learning/rounds/round_01/embeddings.parquet
    python scripts/merge_shards.py tile-stats data/03_predictions/round_01

The per-tile files of a segmentation run (``by_tile/*.geojson``, ``cells/*.parquet``) need no merging.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("mpn_al.merge")

_SHARD = re.compile(r"\.shard-(\d+)-of-(\d+)$")


def find_shards(final_path: str | Path, *, allow_missing: bool = False) -> list[Path]:
    """Shard files of ``final_path`` (``embeddings.parquet`` -> ``embeddings.shard-*-of-*.parquet``), in shard order.

    Raises when shards are missing, counted from the ``-of-N`` in their names, unless
    ``allow_missing``: a task that failed has to be resubmitted, and a task whose shard held no
    tiles (more tasks than tiles) writes no file, which is the one harmless reason for a gap.
    """
    final_path = Path(final_path)
    found: dict[int, Path] = {}
    counts: set[int] = set()
    for path in final_path.parent.glob(f"{final_path.stem}.shard-*-of-*{final_path.suffix}"):
        match = _SHARD.search(path.stem)
        if match:
            found[int(match[1])] = path
            counts.add(int(match[2]))
    if not found:
        raise FileNotFoundError(f"no shard files next to {final_path}")
    if len(counts) != 1:
        raise ValueError(f"the shards of {final_path} come from different arrays (sizes {sorted(counts)})")
    total = counts.pop()
    missing = sorted(set(range(total)) - set(found))
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"{final_path}: shards {missing} of {total} are missing. Resubmit those array tasks with "
            "--skip-existing, or pass --allow-missing if they held no tiles (more tasks than tiles)"
        )
    return [found[index] for index in sorted(found)]


def merge_embedding_shards(final_path: str | Path, *, key: str = "tile_id", allow_missing: bool = False) -> int:
    """Concatenate the shards of ``final_path`` into ``final_path``; returns the number of rows."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    final_path = Path(final_path)
    shards = find_shards(final_path, allow_missing=allow_missing)
    try:
        table = pa.concat_tables([pq.read_table(path) for path in shards])
    except pa.ArrowInvalid as exc:
        raise ValueError(f"the shards have different schemas (different models or settings?): {exc}") from exc
    if key in table.column_names and pc.count_distinct(table[key]).as_py() != table.num_rows:
        raise ValueError(f"column {key!r} has duplicates: the shards overlap")
    temporary = final_path.with_name(final_path.name + ".partial")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(final_path)  # readers never see a half-written file
    logger.info("merged %d shard files into %s (%d rows)", len(shards), final_path, table.num_rows)
    return table.num_rows


def merge_tile_stats(run_dir: str | Path) -> int:
    """Write ``tile_stats.jsonl`` from every ``tile_stats*.jsonl`` of a run directory; returns the tile count.

    A tile that appears more than once (it was redone) keeps its last record.
    """
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("tile_stats*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no tile_stats*.jsonl in {run_dir}")
    records: dict[str, dict] = {}
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["tile_id"]] = record
    out = run_dir / "tile_stats.jsonl"
    temporary = out.with_name(out.name + ".partial")
    temporary.write_text("".join(json.dumps(record) + "\n" for record in records.values()), encoding="utf-8")
    temporary.replace(out)
    logger.info("tile statistics of %d tiles merged into %s", len(records), out)
    return len(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge the shard files that the tasks of a Slurm job array wrote.")
    subcommands = parser.add_subparsers(dest="what", required=True)
    embeddings = subcommands.add_parser("embeddings", help="merge <name>.shard-*-of-*.parquet into <name>.parquet")
    embeddings.add_argument("out", type=Path, help="the final file, e.g. .../embeddings.parquet; its shards lie next to it")
    embeddings.add_argument("--key", default="tile_id", help="column that must be unique (window_id for window tables)")
    embeddings.add_argument("--allow-missing", action="store_true", help="merge even though some shards are absent")
    stats = subcommands.add_parser("tile-stats", help="merge the tile_stats*.jsonl files of a segmentation run")
    stats.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.what == "embeddings":
            merge_embedding_shards(args.out, key=args.key, allow_missing=args.allow_missing)
        else:
            merge_tile_stats(args.run_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0
