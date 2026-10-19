"""Resolve corpus objects and tissue spellings without changing diagnoses."""
from pathlib import Path
import csv


def tissue_key(name):
    return (name or "").strip().casefold()


def object_directory(dataset, sample):
    root = Path(dataset)
    matches = sorted(p for p in root.glob(f"*/*/*/{sample}")
                     if p.is_dir() and "rawdata" not in p.parts[-3].casefold())
    if not matches:
        raise FileNotFoundError(f"{sample}: no object under {root}")
    if len({p.resolve() for p in matches}) != 1 or len({tissue_key(p.parent.name) for p in matches}) != 1:
        raise ValueError(f"{sample}: ambiguous corpus objects: {matches}")
    return next((p for p in matches if p.relative_to(root).parts[0] == "training_set"), matches[0])


def tissue_of_object(dataset, sample, fold_case=True):
    group = object_directory(dataset, sample).parent.name
    return tissue_key(group) if fold_case else group


EXCLUSION_FIELDS = ('donor_or_block_id',)


def reference_exclusions(dataset, sample, bank_objects, fields=EXCLUSION_FIELDS):
    """Leave shared donor/block groups out when fitting a new split's beta."""
    excluded = {sample}
    groups_path = Path(dataset)/'metadata/shared_origin_groups.csv'
    if groups_path.is_file():
        with groups_path.open() as f:
            groups = [set(row['samples'].split(';')) for row in csv.DictReader(f)
                      if row.get('field') in fields]
        while True:
            expanded = excluded | set().union(*(g for g in groups if g & excluded))
            if expanded == excluded:
                break
            excluded = expanded
    return sorted(excluded & set(bank_objects))
