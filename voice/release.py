"""Portable release paths and identities. No training corpus is required at inference."""
from pathlib import Path
import json
from voice.retrieval import sha
from voice.training_contract import gene_hash


class Release:
    def __init__(self, directory, retrieval=True):
        self.root = Path(directory).resolve()
        self.config = json.loads((self.root / 'release.json').read_text())
        if self.config.get('schema') != 'voice25m-release-v1':
            raise ValueError('Unsupported release.json schema')
        self.paths = {}
        fields = ['uni2h', 'stage1', 'stage2', 'gene_space']
        if retrieval:
            fields += ['bank', 'gate']
        for key in fields:
            self.paths[key] = (self.root / self.config[key]).resolve()
            if not self.paths[key].exists():
                raise FileNotFoundError(f'Missing release artifact {key}: {self.paths[key]}')
        self.hashes = {}
        for key in ['stage1', 'stage2', 'gene_space']:
            self.hashes[key] = sha(self.paths[key])
            expected = self.config.get('sha256', {}).get(key)
            if expected and expected != self.hashes[key]:
                raise ValueError(f'Release checksum mismatch: {key}')
        if retrieval:
            self.bank = json.loads((self.paths['bank'] / 'model.json').read_text())
            self.gate = json.loads(self.paths['gate'].read_text())
            self.gate_meta = json.loads(self.paths['gate'].with_suffix('.meta.json').read_text())
            checks = [(self.bank['checkpoint_sha256'], self.hashes['stage1']),
                      (self.bank['gene_space_sha256'], self.hashes['gene_space']),
                      (self.gate_meta['checkpoint_sha256'], self.hashes['stage1']),
                      (self.gate_meta['stage2_checkpoint_sha256'], self.hashes['stage2']),
                      (self.gate_meta['gene_space_sha256'], self.hashes['gene_space']),
                      (self.gate_meta['cache_fingerprint'], self.bank['cache_fingerprint']),
                      (self.gate_meta['gate_sha256'], sha(self.paths['gate']))]
            if any(a != b for a, b in checks):
                raise ValueError('Model checkpoints, genes, reference bank and fusion weights do not match')
            if (self.gate_meta.get('K') != self.bank['retrieval_K'] or
                    self.gate_meta.get('tau') != self.bank['tau']):
                raise ValueError('Fusion weights and reference bank retrieval settings differ')
