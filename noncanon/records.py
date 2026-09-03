"""The rollout record format shared by generation (GPU) and analysis (CPU)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("prompt_id", pa.string()),
        ("sample", pa.int32()),
        ("problem", pa.string()),
        ("answer", pa.string()),
        ("prompt_token_ids", pa.list_(pa.int32())),
        ("token_ids", pa.list_(pa.int32())),
        ("text", pa.string()),
        ("finish_reason", pa.string()),
        ("logprobs", pa.list_(pa.float32())),  # sampled token's raw logprob at each position
        ("topk_ids", pa.list_(pa.list_(pa.int32()))),  # top-k candidates (plus the sampled token if outside the top-k)
        ("topk_logprobs", pa.list_(pa.list_(pa.float32()))),
    ]
)


def write_records(rows: list[dict], path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression="zstd")


def iter_records(path: Path, batch_size: int = 16):
    """Stream rows as dicts without materializing the whole file."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()
