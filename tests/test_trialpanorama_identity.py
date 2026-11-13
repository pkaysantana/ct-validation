"""Tests for trialpanorama_identity, the read-only TrialPanorama name-DrugBank id diagnostic.

Two separate synthetic fixtures, one per mechanism of issue #7. The first three tests pin what
aggregate.py does today with them (documented, not endorsed: delete or update them if the
clustering policy changes). The rest check that the diagnostic tells the two apart:

  A (merge): a TrialPanorama arm named after one drug in another source carries a different
    drug's DrugBank id, so two previously separate components end up in one cluster.
  B (identifier only): an arm whose name no other source knows carries a DrugBank id, so its
    trials join that drug's cluster through the identifier alone. No components connect.

None of the fixtures says which mapping is chemically right.
"""

import hashlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import aggregate
import pandas as pd
import pytest
import trialpanorama_identity as tpi

SCRIPT = Path(tpi.__file__)


def _write(source_dir, name, rows):
    pd.DataFrame(rows).to_parquet(source_dir / f"{name}.parquet", index=False)


def _tp_rows(*arms):
    """TrialPanorama drug-indication rows: (study, arm name, DrugBank id, phase, [EFO ids])."""
    return [
        {
            "study_ids": study,
            "drug_name": name,
            "drugbank_id": drugbank_id,
            "phase": phase,
            "efo_id": efo,
            "is_concluded": True,
        }
        for study, name, drugbank_id, phase, efos in arms
        for efo in efos
    ]


def _drug_a(source_dir):
    """DrugA: a ChEMBL record with a target, and a STITCH record pairing its ChEMBL and DrugBank ids."""
    _write(
        source_dir,
        "chembl_gene_drug",
        [{"gene": "GENEA", "drug_name": "DrugA", "chembl_id": "CHEMBLA"}],
    )
    _write(
        source_dir,
        "stitch_gene_drug",
        [
            {
                "gene": "GENEA",
                "drug_name": None,
                "chembl_id": "CHEMBLA",
                "drugbank_id": "DBA",
                "stitch_id": "CIDmA",
            }
        ],
    )


@pytest.fixture
def merge_source_dir(tmp_path):
    """Mechanism A: arm "Compound-B" (ChEMBL records it as CHEMBLB) carries DrugA's id DBA."""
    source_dir = tmp_path / "merge"
    source_dir.mkdir()
    _drug_a(source_dir)
    _write(
        source_dir,
        "chembl_drug_indication",
        [{"drug_name": "Compound-B", "chembl_id": "CHEMBLB", "efo_id": "EFO:B", "phase": 1}],
    )
    _write(
        source_dir,
        "trialpanorama_drug_indication",
        _tp_rows(("NCT1", "Compound-B", "DBA", 1, ["EFO:B"])),
    )
    return source_dir


@pytest.fixture
def identifier_only_source_dir(tmp_path):
    """Mechanism B: arm "CODE-9", known to no other source, carries DrugA's id DBA.

    One arm in one study, entered under two mapped diseases: two rows, one study arm.
    """
    source_dir = tmp_path / "identifier_only"
    source_dir.mkdir()
    _drug_a(source_dir)
    _write(
        source_dir,
        "trialpanorama_drug_indication",
        _tp_rows(("NCT2", "CODE-9", "DBA", 4, ["EFO:X", "EFO:Y"])),
    )
    return source_dir


@pytest.fixture
def mixed_source_dir(tmp_path):
    """DrugA with one link of every class, plus two arms joined through a shared id.

    Salt-X and Salt-Y are separate ChEMBL records; TrialPanorama gives both arms DBH, an id
    no other source carries (a hub id such as sodium citrate's).
    """
    source_dir = tmp_path / "mixed"
    source_dir.mkdir()
    _drug_a(source_dir)
    _write(
        source_dir,
        "chembl_drug_indication",
        [
            {"drug_name": "Salt-X", "chembl_id": "CHEMBLX", "efo_id": "EFO:S", "phase": 2},
            {"drug_name": "Salt-Y", "chembl_id": "CHEMBLY", "efo_id": "EFO:S", "phase": 2},
        ],
    )
    _write(
        source_dir,
        "trialpanorama_drug_indication",
        _tp_rows(
            ("NCT3", "DrugA", "DBA", 2, ["EFO:Z"]),
            ("NCT4", "DrugA 100 mg", "DBA", 2, ["EFO:Z"]),
            ("NCT5", "Mystery", "DBZ", 3, ["EFO:M"]),
            ("NCT6", "Salt-X", "DBH", 2, ["EFO:S"]),
            ("NCT7", "Salt-Y", "DBH", 2, ["EFO:S"]),
        ),
    )
    return source_dir


def _stitch(source_dir, rows):
    _write(source_dir, "stitch_gene_drug", [{"drug_name": None, **r} for r in rows])


def _run(source_dir, tmp_path, **kwargs):
    out = tmp_path / "diagnostic"
    summary = tpi.run(source_dir, out, **kwargs)
    links = pd.read_csv(out / "links.tsv", sep="\t")
    return summary, links.set_index(["arm_name", "drugbank_id"]), out


def _aggregate(source_dir, out):
    aggregate.aggregate_clinical_trials(source_dir, out)
    return {
        name: pd.read_parquet(out / f"{name}.parquet")
        for name in ["drug_id_mapping", "gene_indication_max_phase"]
    }


def _without_trialpanorama(source_dir, tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    for path in source_dir.glob("*.parquet"):
        if not path.name.startswith("trialpanorama"):
            (bare / path.name).write_bytes(path.read_bytes())
    return bare


def _max_phases(tables):
    gim = tables["gene_indication_max_phase"]
    if gim.empty:  # no evidence source left: aggregate writes a table with no columns
        return {}
    return {(g, e): int(p) for g, e, p in zip(gim["gene"], gim["efo_id"], gim["max_phase"])}


# --- Current behaviour of aggregate.py (pinned, not endorsed) ---


def test_mechanism_a_puts_two_components_in_one_cluster_on_current_code(merge_source_dir, tmp_path):
    # One cluster, and MIN() keeps only one of the two ChEMBL ids on it.
    with_tp = _aggregate(merge_source_dir, tmp_path / "with")["drug_id_mapping"]
    assert len(with_tp) == 1
    assert with_tp["chembl_id"].tolist() == ["CHEMBLA"]

    bare = _without_trialpanorama(merge_source_dir, tmp_path)
    without_tp = _aggregate(bare, tmp_path / "without")["drug_id_mapping"]
    assert sorted(without_tp["chembl_id"]) == ["CHEMBLA", "CHEMBLB"]


def test_mechanism_a_gives_the_arm_disease_to_the_other_drugs_target(merge_source_dir, tmp_path):
    assert _max_phases(_aggregate(merge_source_dir, tmp_path / "a"))[("GENEA", "EFO:B")] == 1
    bare = _without_trialpanorama(merge_source_dir, tmp_path)
    assert ("GENEA", "EFO:B") not in _max_phases(_aggregate(bare, tmp_path / "b"))


def test_mechanism_b_attaches_the_arm_without_connecting_components(
    identifier_only_source_dir, tmp_path
):
    tables = _aggregate(identifier_only_source_dir, tmp_path)
    assert _max_phases(tables)[("GENEA", "EFO:X")] == 4
    # One cluster with one ChEMBL id: a check for clusters holding several drugs sees nothing.
    assert len(tables["drug_id_mapping"]) == 1
    assert tables["drug_id_mapping"]["chembl_id"].tolist() == ["CHEMBLA"]


# --- Classification ---


def test_diagnostic_classifies_mechanism_a_as_merge(merge_source_dir, tmp_path):
    summary, links, out = _run(merge_source_dir, tmp_path)
    link = links.loc[("compound-b", "DBA")]
    assert link["effect"] == "merge"
    assert bool(link["name_in_other_source"])
    assert link["name_side_chembl_ids"] == "CHEMBLB"
    assert link["id_side_chembl_ids"] == "CHEMBLA"
    assert link["components_reached"] == 2
    assert link["cluster_components"] == 2
    assert summary["published_clusters_joining_components"] == {
        "clusters": 1,
        "with_a_merge_link": 1,
    }
    clusters = pd.read_csv(out / "clusters.tsv", sep="\t")
    assert clusters["components"].tolist() == [2]
    assert clusters["merge_links"].tolist() == [1]


def test_diagnostic_classifies_mechanism_b_as_identifier_only(identifier_only_source_dir, tmp_path):
    summary, links, _ = _run(identifier_only_source_dir, tmp_path)
    link = links.loc[("code-9", "DBA")]
    assert link["effect"] == "identifier_only"
    assert not bool(link["name_in_other_source"])
    assert link["components_reached"] == 1
    assert link["cluster_components"] == 1
    assert summary["by_effect"]["merge"]["links"] == 0
    assert summary["published_clusters_joining_components"]["clusters"] == 0


def test_diagnostic_classes_on_mixed_links(mixed_source_dir, tmp_path):
    _, links, _ = _run(mixed_source_dir, tmp_path)
    assert links["effect"].to_dict() == {
        ("druga", "DBA"): "consistent",
        ("druga 100 mg", "DBA"): "identifier_only",
        ("mystery", "DBZ"): "unanchored_id",
        ("salt-x", "DBH"): "merge",
        ("salt-y", "DBH"): "merge",
    }


def test_name_rule_is_exact_on_the_normalised_name(tmp_path):
    """The rule behind issue #7's "match no curated name" count, on synthetic names.

    Case and surrounding spaces are normalised away as aggregate.py does; a spelling variant, a
    dose suffix or a different code are not matched. A match in any other source counts,
    including a DGIdb-only name with no ChEMBL id.
    """
    source_dir = tmp_path / "rule"
    source_dir.mkdir()
    _drug_a(source_dir)
    _write(
        source_dir,
        "chembl_drug_indication",
        [{"drug_name": "BMS-1", "chembl_id": "CHEMBL1", "efo_id": "EFO:1", "phase": 1}],
    )
    _write(source_dir, "dgidb_gene_drug", [{"gene": "GENED", "drug_name": "BMS-4"}])
    _write(
        source_dir,
        "trialpanorama_drug_indication",
        _tp_rows(
            ("N1", " bms-1 ", "DBA", 1, ["EFO:1"]),
            ("N2", "BMS-2", "DBA", 1, ["EFO:1"]),
            ("N3", "BMS-3 100 mg", "DBA", 1, ["EFO:1"]),
            ("N4", "BMS-4", "DBA", 1, ["EFO:1"]),
            ("N5", "BMS 1", "DBA", 1, ["EFO:1"]),
            ("N6", "dasatinib", "DBA", 1, ["EFO:1"]),
        ),
    )
    _, links, _ = _run(source_dir, tmp_path)
    bms = links[links.index.get_level_values("arm_name").str.startswith("bms")]
    assert len(bms) == 5
    unmatched = bms[~bms["name_in_other_source"]]
    assert sorted(unmatched.index.get_level_values("arm_name")) == [
        "bms 1",
        "bms-2",
        "bms-3 100 mg",
    ]
    assert set(unmatched["effect"]) == {"identifier_only"}
    assert set(bms[bms["name_in_other_source"]]["effect"]) == {"merge"}


def test_merge_beyond_one_step_shows_in_cluster_components(tmp_path):
    """DrugA -[DBH]- U1 -[DBK]- U2 -[DBB]- DrugB: no single link reaches two components.

    DBH and DBK are carried by TrialPanorama only. The documented one-step limit leaves the
    links unflagged as merges, but the published cluster's component count still shows it.
    """
    source_dir = tmp_path / "chain"
    source_dir.mkdir()
    _stitch(
        source_dir,
        [
            {"gene": "GENEA", "chembl_id": "CHEMBLA", "drugbank_id": "DBA", "stitch_id": "CA"},
            {"gene": "GENEB", "chembl_id": "CHEMBLB", "drugbank_id": "DBB", "stitch_id": "CB"},
        ],
    )
    _write(
        source_dir,
        "chembl_gene_drug",
        [{"gene": "GENEA", "drug_name": "DrugA", "chembl_id": "CHEMBLA"}],
    )
    _write(
        source_dir,
        "trialpanorama_drug_indication",
        _tp_rows(
            ("NCT1", "DrugA", "DBH", 2, ["EFO:1"]),
            ("NCT2", "U1", "DBH", 2, ["EFO:2"]),
            ("NCT3", "U1", "DBK", 2, ["EFO:2"]),
            ("NCT4", "U2", "DBK", 2, ["EFO:3"]),
            ("NCT5", "U2", "DBB", 2, ["EFO:3"]),
        ),
    )
    summary, links, _ = _run(source_dir, tmp_path)
    assert "merge" not in set(links["effect"])
    assert set(links["cluster_components"]) == {2}
    assert summary["published_clusters_joining_components"] == {
        "clusters": 1,
        "with_a_merge_link": 0,
    }


def test_an_arm_carrying_two_drugs_ids_is_a_merge(tmp_path):
    source_dir = tmp_path / "combination"
    source_dir.mkdir()
    _stitch(
        source_dir,
        [
            {"gene": "GENEA", "chembl_id": "CHEMBLA", "drugbank_id": "DBA", "stitch_id": "CA"},
            {"gene": "GENEB", "chembl_id": "CHEMBLB", "drugbank_id": "DBB", "stitch_id": "CB"},
        ],
    )
    _write(
        source_dir,
        "trialpanorama_drug_indication",
        _tp_rows(
            ("NCT1", "A plus B", "DBA", 3, ["EFO:1"]),
            ("NCT1", "A plus B", "DBB", 3, ["EFO:1"]),
        ),
    )
    _, links, _ = _run(source_dir, tmp_path)
    assert set(links["effect"]) == {"merge"}
    assert links.loc[("a plus b", "DBA"), "study_arms"] == 1


def test_dgidb_synonyms_give_the_arm_name_a_component(identifier_only_source_dir, tmp_path):
    """With DGIdb mapping code-9 to druga, the arm name belongs to DrugA's component."""
    drugs = tmp_path / "drugs.tsv"
    drugs.write_text("drug_claim_name\tdrug_name\tnomenclature\nCODE-9\tDRUGA\tPrimary Drug Name\n")
    _write(
        identifier_only_source_dir,
        "chembl_gene_drug",
        [{"gene": "GENEA", "drug_name": "DrugA", "chembl_id": "CHEMBLA"}],
    )
    summary, with_synonyms, _ = _run(identifier_only_source_dir, tmp_path / "s", dgidb_drugs=drugs)
    _, without, _ = _run(identifier_only_source_dir, tmp_path / "n")
    assert with_synonyms.loc[("code-9", "DBA"), "effect"] == "consistent"
    assert with_synonyms.loc[("code-9", "DBA"), "name_dgidb_canonical"] == "druga"
    assert without.loc[("code-9", "DBA"), "effect"] == "identifier_only"
    assert summary["checks"]["dgidb_synonyms"] == 1


# --- Counting units and provenance ---


def test_counting_units_deduplicate_disease_expanded_rows(identifier_only_source_dir, tmp_path):
    summary, links, out = _run(identifier_only_source_dir, tmp_path)
    assert links.loc[("code-9", "DBA"), ["rows", "rows_with_efo", "study_arms"]].tolist() == [
        2,
        2,
        1,
    ]
    counts = summary["by_effect"]["identifier_only"]
    assert (counts["links"], counts["rows"], counts["study_arms"], counts["studies"]) == (
        1,
        2,
        1,
        1,
    )

    records = pd.read_parquet(out / "records.parquet")
    assert records[["file", "row", "study_ids", "efo_id"]].values.tolist() == [
        ["trialpanorama_drug_indication.parquet", 0, "NCT2", "EFO:X"],
        ["trialpanorama_drug_indication.parquet", 1, "NCT2", "EFO:Y"],
    ]


def test_rows_without_an_efo_id_are_counted_apart(identifier_only_source_dir, tmp_path):
    _write(
        identifier_only_source_dir,
        "trialpanorama_drug_indication",
        _tp_rows(("NCT2", "CODE-9", "DBA", 4, ["EFO:X", None])),
    )
    _, links, _ = _run(identifier_only_source_dir, tmp_path)
    assert links.loc[("code-9", "DBA"), ["rows", "rows_with_efo"]].tolist() == [2, 1]


def test_trialpanorama_gene_drug_records_are_links_too(identifier_only_source_dir, tmp_path):
    _write(
        identifier_only_source_dir,
        "trialpanorama_gene_drug",
        [{"gene": "GENEQ", "drug_name": "CODE-9", "drugbank_id": "DBA"}],
    )
    _, links, _ = _run(identifier_only_source_dir, tmp_path)
    assert links.loc[("code-9", "DBA"), ["rows", "gene_drug_rows"]].tolist() == [2, 1]


def test_a_trialpanorama_file_without_drugbank_ids_has_no_links(
    identifier_only_source_dir, tmp_path
):
    _write(
        identifier_only_source_dir,
        "trialpanorama_gene_drug",
        [{"gene": "GENEQ", "drug_name": "CODE-9"}],
    )
    summary, links, _ = _run(identifier_only_source_dir, tmp_path)
    assert links.loc[("code-9", "DBA"), "gene_drug_rows"] == 0
    assert summary["by_effect"]["identifier_only"]["links"] == 1


def test_blank_drugbank_ids_are_counted(identifier_only_source_dir, tmp_path):
    _write(
        identifier_only_source_dir,
        "trialpanorama_drug_indication",
        _tp_rows(("NCT2", "CODE-9", "DBA", 4, ["EFO:X"]), ("NCT3", "Other", " ", 2, ["EFO:Q"])),
    )
    summary, _, _ = _run(identifier_only_source_dir, tmp_path)
    assert summary["blank_drugbank_id_records"] == {"trialpanorama": 1, "other_sources": 0}


# --- Runtime checks, inputs and CLI ---


def test_rebuilt_partition_matches_aggregate(mixed_source_dir, tmp_path):
    summary, _, _ = _run(mixed_source_dir, tmp_path)
    assert summary["checks"]["partition"]["matches_aggregate_rebuild"]


def test_partition_check_detects_a_mismatch(merge_source_dir):
    records = tpi.cluster_records(tpi.load_records(merge_source_dir), None)
    _, membership = tpi.published_clusters(merge_source_dir, None)
    assert tpi.check_partition(records, membership)[0]["matches_aggregate_rebuild"]
    split = membership.copy()
    split.loc[split["id_value"] == "CHEMBLB", "our_drug_id"] = 99
    assert not tpi.check_partition(records, split)[0]["matches_aggregate_rebuild"]


def test_compares_with_the_published_drug_id_mapping(merge_source_dir, tmp_path):
    summary, _, _ = _run(merge_source_dir, tmp_path / "a")
    assert summary["checks"]["published_drug_id_mapping"] == "not_found"
    aggregate.aggregate_clinical_trials(merge_source_dir, merge_source_dir / "aggregated")
    summary, _, _ = _run(merge_source_dir, tmp_path / "b")
    assert summary["checks"]["published_drug_id_mapping"] == "identical"
    mapping = merge_source_dir / "aggregated" / "drug_id_mapping.parquet"
    pd.read_parquet(mapping).assign(drug_name="changed").to_parquet(mapping, index=False)
    summary, _, _ = _run(merge_source_dir, tmp_path / "c")
    assert summary["checks"]["published_drug_id_mapping"] == "differs"


def test_inputs_are_left_unchanged(mixed_source_dir, tmp_path):
    def digest():
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(mixed_source_dir.iterdir())
        }

    before = digest()
    _run(mixed_source_dir, tmp_path)
    assert digest() == before


def test_missing_dgidb_file_fails_before_any_output(merge_source_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        _run(merge_source_dir, tmp_path, dgidb_drugs=tmp_path / "absent.tsv")
    assert not (tmp_path / "diagnostic").exists()


def test_source_dir_must_be_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        _run(tmp_path / "absent", tmp_path)


def test_output_dir_cannot_be_the_source_dir(merge_source_dir):
    before = sorted(p.name for p in merge_source_dir.iterdir())
    with pytest.raises(ValueError, match="read-only"):
        tpi.run(merge_source_dir, merge_source_dir)
    assert sorted(p.name for p in merge_source_dir.iterdir()) == before


def test_no_links_is_an_error_not_an_empty_report(identifier_only_source_dir, tmp_path):
    _write(
        identifier_only_source_dir,
        "trialpanorama_drug_indication",
        [{"study_ids": "NCT2", "drug_name": "CODE-9", "phase": 4, "efo_id": "EFO:X"}],
    )
    with pytest.raises(ValueError, match="DrugBank id"):
        _run(identifier_only_source_dir, tmp_path)


def test_summary_schema(mixed_source_dir, tmp_path):
    summary, _, out = _run(mixed_source_dir, tmp_path)
    assert set(summary) == {
        "by_effect",
        "published_clusters_joining_components",
        "blank_drugbank_id_records",
        "checks",
        "environment",
        "inputs",
    }
    assert set(summary["by_effect"]) == set(tpi.EFFECTS)
    assert len(summary["environment"]["aggregate_py_git_blob"]) == 40
    assert "chembl_gene_drug.parquet" in summary["inputs"]
    assert sorted(p.name for p in out.iterdir()) == [
        "clusters.tsv",
        "links.tsv",
        "records.parquet",
        "report.md",
        "summary.json",
    ]
    assert list(pd.read_csv(out / "links.tsv", sep="\t").columns) == tpi.LINK_COLUMNS


def test_cli_with_explicit_paths_from_another_working_directory(mixed_source_dir, tmp_path):
    """Absolute paths with a space, and relative paths, both resolve; nothing else is written."""
    work = tmp_path / "elsewhere with space"
    work.mkdir()

    def cli(*args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args], cwd=work, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    cli("--source-dir", str(mixed_source_dir), "--output-dir", str(work / "abs out"))
    shutil.copytree(mixed_source_dir, work / "inputs")
    cli("--source-dir", "inputs", "--output-dir", "rel_out")

    assert sorted(p.name for p in work.iterdir()) == ["abs out", "inputs", "rel_out"]
    for name in ["links.tsv", "records.parquet", "clusters.tsv", "summary.json", "report.md"]:
        assert (work / "abs out" / name).read_bytes() == (work / "rel_out" / name).read_bytes()
    links = pd.read_csv(work / "abs out" / "links.tsv", sep="\t")
    assert sorted(links["effect"]) == [
        "consistent",
        "identifier_only",
        "merge",
        "merge",
        "unanchored_id",
    ]


def test_a_missing_private_aggregate_helper_is_named():
    tpi._require_aggregate_helpers(aggregate)
    stand_in = types.SimpleNamespace(_load_sources=None, _drug_name_key=None)
    with pytest.raises(ImportError, match="_build_drug_id_mapping"):
        tpi._require_aggregate_helpers(stand_in)
