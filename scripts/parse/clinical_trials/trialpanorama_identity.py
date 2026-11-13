#!/usr/bin/env python3
"""Classify how TrialPanorama's DrugBank ids attach trial arms to drug clusters (issue #7).

A read-only diagnostic. It reads the per-source parquets aggregate.py reads, writes only to
--output-dir, and changes no clustering, parsing or benchmark output. It makes no judgement about
chemical identity: every class below describes the clustering graph, not whether a mapping is
right.

Terms
-----
link            a distinct (normalized arm name, DrugBank id) pair in the TrialPanorama files.
                One arm appears in many rows (one per mapped disease); the link is what
                create_drug_clusters turns into graph edges.
base partition  the partition create_drug_clusters builds from the same records with every
                TrialPanorama DrugBank id withheld.
component       a cluster of the base partition holding at least one record from a source other
                than TrialPanorama (ChEMBL, Open Targets, DGIdb or STITCH).

An arm name belongs to a component when its normalized form (TRIM(LOWER(drug_name)), blank
becoming NULL: the key aggregate.py joins on) equals the normalized drug_name of another
source's record, or when --dgidb-drugs maps it to one by a DGIdb synonym. Nothing fuzzier is
applied, so "dasatinib 100 mg" does not match "dasatinib".

Each end of a link looks one step out. The arm name reaches its own component and the
components its other DrugBank ids reach. The id reaches the component of the other source that
carries it or, for an id only TrialPanorama carries, the components of the other arm names
sharing it. A link is judged by what its own arm name reaches, so a hub id such as one shared by
hundreds of names is not charged to each name.

  consistent       the id reaches only the component the arm name belongs to.
  merge            the link connects previously separate components.
  identifier_only  the arm name belongs to no component, so the arm joins the id's cluster
                   through the identifier alone; within one step no two components connect.
  unanchored_id    apart from this arm name, nothing the id reaches belongs to a component.

Neither `merge` nor `identifier_only` says a mapping is incorrect. A merge also joins a salt
form to its free base, or one molecule that two sources keep apart; identifier_only also holds
arm names that are a drug name plus a dose or formulation. Longer chains (names that share ids
that are shared again) are not assigned to a link; `cluster_components` counts the components
in each published cluster, so they still show.

Outputs in --output-dir
-----------------------
links.tsv       one row per link
                  arm_name, drugbank_id, effect
                  name_in_other_source    arm name equals another source's normalized drug_name
                  name_dgidb_canonical    DGIdb canonical name for the arm name, if any
                  components_reached      components the arm name and id reach together
                  cluster_components      components in the link's published cluster
                  name_side_names, name_side_chembl_ids, id_side_names, id_side_chembl_ids
                                          up to 5 names / ChEMBL ids per side, to inspect
                  our_drug_id             cluster id in this build (not stable across builds)
                  rows, rows_with_efo     drug-indication rows (one per mapped disease), and
                                          those with an EFO id, which aggregate.py keeps
                  study_arms              distinct (study, arm name, DrugBank id)
                  gene_drug_rows, max_phase, example_study
records.parquet every TrialPanorama record of a link that is not `consistent`:
                  file, row (0-based within that file), study_ids, drug_name, drugbank_id,
                  phase, efo_id, effect
clusters.tsv    one row per published cluster that a merge or identifier_only link reaches:
                  our_drug_id, published_name, merge_links, identifier_only_links, rows,
                  study_arms, drugbank_ids, components, chembl_ids
summary.json    counts per effect in every unit below, the checks, environment, input hashes
report.md       the same counts, readable

Counting units: `rows` are TrialPanorama drug-indication rows, `study_arms` distinct (study,
arm name, DrugBank id) so an arm carrying two ids counts once per id, `studies` distinct studies,
`links` and `arm_names` as named. The same study can appear under several effects, so counts
across effects do not add up to distinct studies or clusters.

Checks at runtime
-----------------
  * The partition rebuilt from the records must equal the one aggregate.py builds from the same
    files, identifier by identifier, or the run stops.
  * summary.json reports whether the rebuilt drug_id_mapping equals
    <source-dir>/aggregated/drug_id_mapping.parquet when that file exists (`identical`,
    `differs`, `not_found`). A difference means that aggregate came from other inputs or
    settings, such as another DGIdb file.
  * Blank DrugBank ids are counted: aggregate.py unions a blank string like any other id.
  * --output-dir may not be --source-dir, and a missing input path stops the run before any
    output is written.

Assumptions and limitations
---------------------------
  * Inputs are the parsed per-source parquets of one build, as aggregate.py reads them
    (<source>_gene_drug.parquet, <source>_drug_indication.parquet), plus the DGIdb drugs.tsv
    given to aggregate.py. Without --dgidb-drugs the clusters differ from such a run and a
    warning says so. TrialPanorama files need drug_name and drugbank_id; a drug-indication file
    also study_ids, phase and efo_id. A file without drugbank_id has no links.
  * Versions: written against aggregate.py at git blob 696f94e, unchanged from v0.4.3 through
    v0.5.1 (main df1a477); summary.json records the blob it ran with, and pandas and duckdb.
    Tested on the v0.4.3 data release. Later data releases were not available.
  * It depends on three private aggregate.py helpers: _load_sources, _drug_name_key and
    _build_drug_id_mapping. They keep the record set, name key and cluster ids from drifting
    from production. A missing helper stops the run with a message naming it; a changed one
    fails the partition check.
  * aggregate.py interpolates source paths into SQL, so a source directory whose path contains
    an apostrophe fails inside it. This script reads files by parameter and is not affected
    itself.
  * Only TrialPanorama's DrugBank ids are examined. A merge made by another source, for
    example a DGIdb synonym, is already in the base partition and is not reported.
  * Name matching is exact on the normalized string. Spelling variants, trade names, salt
    forms and dose text are not matched, so identifier_only includes names that a chemist
    would match.
  * No chemical identity is adjudicated and no correction, exclusion or clustering rule is
    proposed. Whether a link is wrong needs external evidence this script does not use.

Usage:
    python scripts/parse/clinical_trials/trialpanorama_identity.py \\
        --source-dir data/clinical_trials --dgidb-drugs data/sources/dgidb/drugs.tsv.gz \\
        --output-dir data/clinical_trials/trialpanorama_identity
"""

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import aggregate
import duckdb
import pandas as pd

log = logging.getLogger(__name__)

TRIALPANORAMA = "trialpanorama"
EFFECTS = ["consistent", "merge", "identifier_only", "unanchored_id"]
FLAGGED_EFFECTS = ["merge", "identifier_only"]
LINK_KEY = ["drug_name_normalized", "drugbank_id"]
ARM_KEY = ["study_ids", *LINK_KEY]
# Private helpers this script cannot do without; see "Assumptions and limitations".
PRIVATE_AGGREGATE_HELPERS = ("_load_sources", "_drug_name_key", "_build_drug_id_mapping")

LINK_COLUMNS = [
    "arm_name",
    "drugbank_id",
    "effect",
    "name_in_other_source",
    "name_dgidb_canonical",
    "components_reached",
    "cluster_components",
    "name_side_names",
    "name_side_chembl_ids",
    "id_side_names",
    "id_side_chembl_ids",
    "our_drug_id",
    "rows",
    "rows_with_efo",
    "study_arms",
    "gene_drug_rows",
    "max_phase",
    "example_study",
]


def _require_aggregate_helpers(module) -> None:
    """Stop with a clear message if aggregate.py no longer has a helper this script uses."""
    if missing := [name for name in PRIVATE_AGGREGATE_HELPERS if not hasattr(module, name)]:
        raise ImportError(
            f"trialpanorama_identity.py relies on private aggregate.py helpers {missing}, "
            "which this aggregate.py does not define; the script needs updating"
        )


_require_aggregate_helpers(aggregate)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    """Git's blob id of a file with LF endings, comparable to `git rev-parse <rev>:<path>`."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def load_records(source_dir: Path) -> pd.DataFrame:
    """Every drug record create_drug_clusters sees, with its source file and 0-based row.

    Mirrors aggregate._build_drug_id_mapping: the same files, identifier columns and name key.
    check_partition verifies the mirror.
    """
    gene_drug_paths, drug_indication_paths = aggregate._load_sources(source_dir)
    conn = duckdb.connect(":memory:")
    parts = []
    for source, path in [*gene_drug_paths.items(), *drug_indication_paths.items()]:
        cols = conn.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(path)]).df().columns
        wanted = ["chembl_id", "drugbank_id", "stitch_id"]
        if source == TRIALPANORAMA:
            wanted += ["study_ids", "phase", "efo_id"]
        select = "".join(f"{c}, " for c in wanted if c in cols)
        df = conn.execute(
            f"""
            SELECT drug_name, {select}{aggregate._drug_name_key()} AS drug_name_normalized,
                file_row_number AS row
            FROM read_parquet(?, file_row_number = true)
            ORDER BY file_row_number
            """,
            [str(path)],
        ).df()
        # Absent and all-null columns are restored after the concat, which would otherwise
        # warn that all-NA columns stop being ignored when choosing dtypes.
        parts.append(df.dropna(axis=1, how="all").assign(source=source, file=path.name))
    conn.close()
    columns = ["drug_name", *aggregate.ID_COLUMNS, "study_ids", "phase", "efo_id", "row"]
    records = pd.concat(parts, ignore_index=True).reindex(columns=[*columns, "source", "file"])
    for col in aggregate.ID_COLUMNS:
        records[col] = records[col].astype(object).where(records[col].notna(), None)
    return records


def published_clusters(
    source_dir: Path,
    drug_synonyms: dict[str, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """aggregate.py's drug_id_mapping and identifier membership, rebuilt from these inputs."""
    gene_drug_paths, drug_indication_paths = aggregate._load_sources(source_dir)
    conn = duckdb.connect(":memory:")
    try:
        return aggregate._build_drug_id_mapping(
            conn, gene_drug_paths, drug_indication_paths, drug_synonyms
        )
    finally:
        conn.close()


def compare_published(source_dir: Path, drug_id_mapping: pd.DataFrame) -> str:
    """Whether the rebuilt drug_id_mapping equals the one aggregate.py wrote next to the inputs.

    `differs` means that aggregate came from other inputs or settings (for example another
    --dgidb-drugs), so cluster ids here would not match it.
    """
    path = source_dir / "aggregated" / "drug_id_mapping.parquet"
    if not path.exists():
        return "not_found"
    published = pd.read_parquet(path)
    if list(published.columns) != list(drug_id_mapping.columns):
        return "differs"
    key = list(published.columns)
    a = published.astype(str).sort_values(key).reset_index(drop=True)
    b = drug_id_mapping.astype(str).sort_values(key).reset_index(drop=True)
    return "identical" if a.equals(b) else "differs"


def _membership(records: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """(id_value, cluster) for every non-empty identifier, as aggregate._cluster_membership."""
    parts = [
        records[[col, cluster_col]].dropna().set_axis(["id_value", "cluster"], axis=1)
        for col in aggregate.ID_COLUMNS
    ]
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    return out[out["id_value"].astype(str).str.strip() != ""]


def check_partition(records: pd.DataFrame, membership: pd.DataFrame) -> tuple[dict, dict]:
    """Compare the rebuilt full partition with aggregate.py's, identifier by identifier.

    Returns the check and, when the partitions agree, rebuilt cluster -> our_drug_id.
    """
    joined = _membership(records, "full_cluster").merge(
        membership[["id_value", "our_drug_id"]], on="id_value", how="outer"
    )
    pairs = joined.dropna().drop_duplicates(["cluster", "our_drug_id"])
    check = {
        "identifiers": int(joined["id_value"].nunique()),
        "identifiers_only_in_aggregate": int(joined["cluster"].isna().sum()),
        "identifiers_only_in_rebuild": int(joined["our_drug_id"].isna().sum()),
        "rebuilt_clusters_split_by_aggregate": int(pairs["cluster"].duplicated().sum()),
        "aggregate_clusters_split_by_rebuild": int(pairs["our_drug_id"].duplicated().sum()),
    }
    check["matches_aggregate_rebuild"] = not any(v for k, v in check.items() if k != "identifiers")
    return check, dict(zip(pairs["cluster"].astype(int), pairs["our_drug_id"].astype(int)))


def cluster_records(records: pd.DataFrame, drug_synonyms: dict[str, str] | None) -> pd.DataFrame:
    """Add the full partition (aggregate.py's) and the base partition (no TrialPanorama ids)."""
    records = records.copy()
    full = aggregate.create_drug_clusters(records.copy(), drug_synonyms=drug_synonyms)
    records["full_cluster"] = full["drug_cluster"]
    base = records.copy()
    base.loc[base["source"].eq(TRIALPANORAMA), "drugbank_id"] = None
    records["base_cluster"] = aggregate.create_drug_clusters(base, drug_synonyms=drug_synonyms)[
        "drug_cluster"
    ]
    return records


def _joined(values, limit: int = 5) -> str | None:
    values = sorted(values)
    return "|".join(values[:limit]) + ("|..." if len(values) > limit else "") if values else None


def classify_links(
    records: pd.DataFrame,
    drug_synonyms: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify every TrialPanorama (arm name, DrugBank id) link against the base partition.

    `records` must carry full_cluster and base_cluster (cluster_records). Returns (links,
    tp_records): one row per link, and the TrialPanorama records that carry a link, each with
    its link's effect. The classes are defined in the module docstring.
    """
    is_tp = records["source"].eq(TRIALPANORAMA)
    other = records.loc[~is_tp]
    source_components = set(other["base_cluster"])
    with_id = other[other["drugbank_id"].notna()]
    id_cluster = dict(zip(with_id["drugbank_id"], with_id["base_cluster"]))

    tp = records.loc[is_tp & records[LINK_KEY].notna().all(axis=1)].copy()
    if tp.empty:
        raise ValueError("No TrialPanorama record carries both a drug name and a DrugBank id")
    links = tp.drop_duplicates(LINK_KEY)[[*LINK_KEY, "base_cluster", "full_cluster"]]
    links = links.rename(columns={"base_cluster": "name_cluster"}).reset_index(drop=True)

    # A name's records all share one base cluster, so `own` holds at most one component.
    names, ids = links["drug_name_normalized"].tolist(), links["drugbank_id"].tolist()
    own = {n: {int(c)} & source_components for n, c in zip(names, links["name_cluster"])}
    names_by_id: dict[str, set] = defaultdict(set)
    ids_by_name: dict[str, set] = defaultdict(set)
    for n, d in zip(names, ids):
        names_by_id[d].add(n)
        ids_by_name[n].add(d)
    # For an id no other source carries: how many of its arm names belong to each component.
    hub_counts = {
        d: Counter(c for m in ms for c in own[m])
        for d, ms in names_by_id.items()
        if d not in id_cluster
    }

    def id_reach(d: str, name: str) -> set:
        """Components id `d` reaches, leaving out what `name` itself brings."""
        if d in id_cluster:
            return {int(id_cluster[d])}
        return {c for c, k in hub_counts[d].items() if k - (c in own[name]) > 0}

    effects, name_ends, id_ends = [], [], []
    for n, d in zip(names, ids):
        id_end = id_reach(d, n)
        via = set().union(*(id_reach(x, n) for x in ids_by_name[n] if x != d))
        if not id_end:
            effect = "unanchored_id"
        elif own[n]:
            effect = "consistent" if id_end <= own[n] else "merge"
        else:
            effect = "identifier_only" if len(id_end | via) == 1 or not via else "merge"
        effects.append(effect)
        name_ends.append(own[n] | via)
        id_ends.append(id_end)
    links["effect"] = effects
    links["components_reached"] = [len(a | b) for a, b in zip(name_ends, id_ends)]

    named = other[other["drug_name_normalized"].notna()]
    names_of = named.groupby("base_cluster")["drug_name_normalized"].agg(set).to_dict()
    chembl = other[other["chembl_id"].notna()]
    chembl_of = chembl.groupby("base_cluster")["chembl_id"].agg(set).to_dict()

    def gather(clusters, lookup) -> set:
        return set().union(*(lookup.get(c, set()) for c in clusters))

    links["name_in_other_source"] = links["drug_name_normalized"].isin(
        set(named["drug_name_normalized"])
    )
    links["name_dgidb_canonical"] = links["drug_name_normalized"].map(drug_synonyms or {})
    for side, clusters in [("name_side", name_ends), ("id_side", id_ends)]:
        links[f"{side}_names"] = [_joined(gather(c, names_of)) for c in clusters]
        links[f"{side}_chembl_ids"] = [_joined(gather(c, chembl_of)) for c in clusters]

    tp = tp.merge(links[[*LINK_KEY, "effect"]], on=LINK_KEY, how="left")
    return links, tp


def count_links(links: pd.DataFrame, tp: pd.DataFrame) -> pd.DataFrame:
    """Add rows, study arms and gene-drug rows per link."""
    di = tp[tp["file"].str.endswith("_drug_indication.parquet")]
    counts = (
        di.groupby(LINK_KEY)
        .agg(
            rows=("row", "size"),
            rows_with_efo=("efo_id", "count"),
            study_arms=("study_ids", "nunique"),
            max_phase=("phase", "max"),
            example_study=("study_ids", "min"),
        )
        .reset_index()
    )
    gene_drug = tp[tp["file"].str.endswith("_gene_drug.parquet")].groupby(LINK_KEY).size()
    counts = counts.merge(
        gene_drug.rename("gene_drug_rows").reset_index(), on=LINK_KEY, how="outer"
    )
    out = links.merge(counts, on=LINK_KEY, how="left")
    for col in ["rows", "rows_with_efo", "study_arms", "gene_drug_rows"]:
        out[col] = out[col].fillna(0).astype(int)
    out["max_phase"] = out["max_phase"].astype("Int64")
    return out


def cluster_table(
    records: pd.DataFrame,
    links: pd.DataFrame,
    drug_id_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """One row per published cluster a merge or identifier_only link reaches.

    `components` counts the base-partition components in the published cluster: 2 or more means
    TrialPanorama links connected previously separate components.
    """
    flagged = links[links["effect"].isin(FLAGGED_EFFECTS)]
    if flagged.empty:
        return pd.DataFrame(columns=["our_drug_id"])
    other = records[records["source"].ne(TRIALPANORAMA)]
    components = other.groupby("our_drug_id").agg(
        components=("base_cluster", "nunique"), chembl_ids=("chembl_id", "nunique")
    )
    by_effect = pd.crosstab(flagged["our_drug_id"], flagged["effect"]).reindex(
        columns=FLAGGED_EFFECTS, fill_value=0
    )
    by_effect.columns = [f"{c}_links" for c in by_effect.columns]
    sums = flagged.groupby("our_drug_id")[["rows", "study_arms"]].sum()
    ids = flagged.groupby("our_drug_id")["drugbank_id"].agg(lambda s: "|".join(sorted(set(s))))
    table = pd.concat([by_effect, sums, ids.rename("drugbank_ids")], axis=1)
    table = table.join(components).reset_index()
    names = drug_id_mapping[["our_drug_id", "drug_name"]].rename(
        columns={"drug_name": "published_name"}
    )
    table = table.merge(names, on="our_drug_id", how="left")
    columns = ["our_drug_id", "published_name", *by_effect.columns, "rows", "study_arms"]
    columns += ["drugbank_ids", "components", "chembl_ids"]
    return table[columns].sort_values(["rows", "our_drug_id"], ascending=[False, True])


def summarise(links: pd.DataFrame, tp: pd.DataFrame) -> dict:
    """Counts per effect in every counting unit, so no unit is read as another."""
    di = tp[tp["file"].str.endswith("_drug_indication.parquet")]
    by_effect = {}
    for effect in EFFECTS:
        lk = links[links["effect"] == effect]
        d = di[di["effect"] == effect]
        by_effect[effect] = {
            "links": int(len(lk)),
            "arm_names": int(lk["arm_name"].nunique()),
            "drugbank_ids": int(lk["drugbank_id"].nunique()),
            "rows": int(len(d)),
            "rows_with_efo": int(d["efo_id"].notna().sum()),
            "study_arms": int(len(d.drop_duplicates(ARM_KEY))),
            "studies": int(d["study_ids"].nunique()),
            "published_clusters": int(lk["our_drug_id"].nunique()),
        }
    return {"by_effect": by_effect}


def _markdown_table(df: pd.DataFrame) -> list[str]:
    def cell(v) -> str:
        return "" if pd.isna(v) else f"{v:,}" if isinstance(v, int) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return lines


def write_report(summary: dict, clusters: pd.DataFrame, path: Path) -> None:
    by_effect = pd.DataFrame(summary["by_effect"]).T.rename_axis("effect").reset_index()
    lines = [
        "# TrialPanorama arm name - DrugBank id links",
        "",
        "Classes compare each link's ends in the partition built without TrialPanorama's",
        "DrugBank ids. They describe the clustering graph; none says a mapping is incorrect.",
        "",
        *_markdown_table(by_effect),
        "",
    ]
    if not clusters.empty:
        lines += ["## Published clusters reached by merge or identifier_only links", ""]
        lines += ["Top 20 by rows; the full list is clusters.tsv.", ""]
        lines += [*_markdown_table(clusters.head(20)), ""]
    joined = summary["published_clusters_joining_components"]
    lines += [
        f"Published clusters holding two or more components: {joined['clusters']:,}, "
        f"{joined['with_a_merge_link']:,} of them with a merge link.",
        "",
        f"Checks: {json.dumps(summary['checks'])}",
        f"Blank DrugBank id records: {json.dumps(summary['blank_drugbank_id_records'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    source_dir: Path,
    output_dir: Path,
    dgidb_drugs: Path | None = None,
) -> dict:
    """Classify every link and write the outputs listed in the module docstring."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"--source-dir is not a directory: {source_dir}")
    if dgidb_drugs is not None and not dgidb_drugs.exists():
        raise FileNotFoundError(dgidb_drugs)
    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("--output-dir must differ from --source-dir: inputs are read-only")
    if dgidb_drugs is None:
        log.warning("No --dgidb-drugs: clusters differ from an aggregate run that used them")
    drug_synonyms = aggregate.load_dgidb_synonyms(dgidb_drugs) if dgidb_drugs else None

    records = cluster_records(load_records(source_dir), drug_synonyms)
    drug_id_mapping, membership = published_clusters(source_dir, drug_synonyms)
    partition, to_published = check_partition(records, membership)
    if not partition["matches_aggregate_rebuild"]:
        raise RuntimeError(f"Rebuilt partition differs from aggregate.py's: {partition}")
    records["our_drug_id"] = records["full_cluster"].map(to_published).astype("Int64")

    links, tp = classify_links(records, drug_synonyms)
    links["our_drug_id"] = links["full_cluster"].map(to_published).astype("Int64")
    other = records[records["source"] != TRIALPANORAMA]
    per_cluster = other.groupby("our_drug_id")["base_cluster"].nunique()
    links["cluster_components"] = links["our_drug_id"].map(per_cluster).fillna(0).astype(int)
    links = count_links(links, tp)
    links = links.rename(columns={"drug_name_normalized": "arm_name"})
    links = links.sort_values(["effect", "rows", "arm_name"], ascending=[True, False, True])

    output_dir.mkdir(parents=True, exist_ok=True)
    links[LINK_COLUMNS].to_csv(output_dir / "links.tsv", sep="\t", index=False)
    tp.loc[
        tp["effect"] != "consistent",
        ["file", "row", "study_ids", "drug_name", "drugbank_id", "phase", "efo_id", "effect"],
    ].sort_values(["file", "row"]).to_parquet(output_dir / "records.parquet", index=False)
    clusters = cluster_table(records, links, drug_id_mapping)
    clusters.to_csv(output_dir / "clusters.tsv", sep="\t", index=False)

    summary = summarise(links, tp)
    # Only TrialPanorama ids are withheld from the base partition, so every published cluster
    # holding two components was connected through TrialPanorama links.
    joined = set(per_cluster.loc[per_cluster > 1].index)
    summary["published_clusters_joining_components"] = {
        "clusters": len(joined),
        "with_a_merge_link": len(
            joined & set(links.loc[links["effect"] == "merge", "our_drug_id"])
        ),
    }
    # aggregate.py unions a blank string like any other id, so blank ids would join unrelated drugs.
    is_tp = records["source"] == TRIALPANORAMA
    blank = records["drugbank_id"].astype(str).str.strip().eq("") & records["drugbank_id"].notna()
    summary["blank_drugbank_id_records"] = {
        "trialpanorama": int((blank & is_tp).sum()),
        "other_sources": int((blank & ~is_tp).sum()),
    }
    summary["checks"] = {
        "partition": partition,
        "dgidb_synonyms": len(drug_synonyms or {}),
        "published_drug_id_mapping": compare_published(source_dir, drug_id_mapping),
    }
    summary["environment"] = {
        "aggregate_py_git_blob": _git_blob_sha1(Path(aggregate.__file__)),
        "pandas": pd.__version__,
        "duckdb": duckdb.__version__,
    }
    inputs = sorted(source_dir.glob("*.parquet")) + ([dgidb_drugs] if dgidb_drugs else [])
    summary["inputs"] = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in inputs}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, clusters, output_dir / "report.md")
    log.info(f"Wrote {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Per-source parquets")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dgidb-drugs", type=Path, help="DGIdb drugs.tsv, as given to aggregate")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args.source_dir, args.output_dir, args.dgidb_drugs)


if __name__ == "__main__":
    main()
