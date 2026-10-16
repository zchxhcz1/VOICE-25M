"""Select tissue and panel-specific gene fusion weights."""
from .tissue import tissue_key
import hashlib
import json


def validate_gate_genes(gate, genes):
    """A transferred integer-ID beta must use the same ordered head as its fit."""
    expected = gate.get('fit_signature', {}).get('genes')
    if expected:
        actual = hashlib.sha256(json.dumps(list(genes)).encode()).hexdigest()
        if actual != expected:
            raise ValueError('Gate beta gene IDs belong to a different gene head; refit the gate')


def select_gate_source(gate, tissue, n_genes):
    width = "wide" if n_genes >= int(gate.get("panel_cut", 1000)) else "narrow"
    fold = gate.get("tissue_casefold", False)
    normalise = tissue_key if fold else lambda t: t or ""
    strata = gate["strata"]
    minimum = int(gate.get("min_refs", 2))
    eligible = {k: v for k, v in strata.items()
                if v.get("n_refs", minimum) >= minimum}
    keys = {}
    for key in eligible:
        t, w = key.rsplit("|", 1)
        if w not in ("narrow", "wide"):
            raise ValueError(f"Invalid gate stratum: {key}")
        folded = f"{normalise(t)}|{w}"
        if folded in keys:
            raise ValueError(f"Case-equivalent strata require refitting: {keys[folded]}, {key}")
        keys[folded] = key
    own = f"{normalise(tissue)}|{width}"
    if own in keys:
        return keys[own]
    order = gate.get("strata_order", list(strata))
    return next((k for k in order if k in eligible and
                 normalise(k.rsplit("|", 1)[0]) == normalise(tissue)), "global")


def select_gate(gate, tissue, n_genes):
    source = select_gate_source(gate, tissue, n_genes)
    beta = (gate["strata"][source] if source != "global" else gate["global"])["beta"]
    return source, {int(k): float(v) for k, v in beta.items()}
