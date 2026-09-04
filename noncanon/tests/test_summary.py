"""The summary tables: per-family p-value bookkeeping, sparse marking, settings lookup."""

import json

import pytest

from noncanon import summary


def write_cell(tmp_path, name, arm, flags, correct, tokens=2000, temperature=1.0):
    d = tmp_path / name
    (d / "metrics").mkdir(parents=True)
    with (d / "metrics" / "analysis.jsonl").open("w") as f:
        for i, (flagged, ok) in enumerate(zip(flags, correct)):
            f.write(json.dumps({"file": f"{arm}.parquet", "finish_reason": "stop", "correct": ok, "n_tokens": tokens, "n_units": tokens,
                                "nc_events": int(flagged), "event_positions": [100] if flagged else []}) + "\n")
    (d / f"{arm}.meta.json").write_text(json.dumps({"sampling": {"temperature": temperature, "top_p": 1.0}}))
    return d


def test_ladder_p_values_are_per_family_and_sparse_tests_are_marked(tmp_path, capsys):
    a = write_cell(tmp_path, "a", "untruncated", [1] * 5 + [0] * 45, [True] * 50)
    b = write_cell(tmp_path, "b", "untruncated", [1] * 25 + [0] * 25, [True] * 50)
    c = write_cell(tmp_path, "c", "untruncated", [1] * 3 + [0] * 47, [True] * 3 + [False] * 47)  # 3 correct rollouts: sparse
    summary.ladder([f"X:s1={a}", f"Y:s1={c}", f"X:s2={b}"], "untruncated")
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| ")][1:]  # drop the header
    x1, y1, x2 = rows
    assert "| — | — |" in x1 and "| — | — |" in y1  # first stage of each family
    assert "2.0e-05" in x2 and y1.count("—") >= 2  # X:s2 is tested against X:s1 (5/50 vs 25/50), never against Y
    assert f"(3/3){summary.SPARSE}" in y1


def test_pairs_reads_settings_and_fails_clearly_without_meta(tmp_path, capsys):
    a = write_cell(tmp_path, "a", "untruncated", [1, 0, 0, 0], [True] * 4, temperature=1.0)
    b = write_cell(tmp_path, "b", "recommended", [0, 0, 0, 0], [True] * 4, temperature=0.6)
    summary.pairs([f"m={a}:untruncated,{b}:recommended"], ("u", "r"), None)
    out = capsys.readouterr().out
    assert "T=1.0, top-p=1.0 / T=0.6, top-p=1.0" in out and summary.SPARSE in out
    (b / "recommended.meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="recommended.meta.json"):
        summary.pairs([f"m={a}:untruncated,{b}:recommended"], ("u", "r"), None)


def test_p_str_thresholds():
    assert summary.fmt_p(1e-12) == "< 1e-10" and summary.fmt_p(0.00042) == "4.2e-04" and summary.fmt_p(0.0479) == "0.048"
    assert float(summary.omnibus([(5, 50), (25, 50), (3, 50)])) < 1e-6
    assert summary.omnibus([(0, 50), (0, 50)]) == "—"
