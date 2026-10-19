"""Bin, sample and pad scGPT cell tokens using the VOICE-25M extraction settings."""
import torch
from .preprocess import binning


class DataCollator:
    def __init__(self, pad_token_id, pad_value):
        self.pad_token_id, self.pad_value = pad_token_id, pad_value

    def __call__(self, examples):
        length = min(1200, max(len(example['genes']) for example in examples))
        all_genes, all_values = [], []
        for example in examples:
            genes, values = example['genes'], example['expressions']
            values[1:] = binning(values[1:], n_bins=51)
            if len(genes) > length:
                indices = torch.randperm(len(genes)-1, device=genes.device)[:length-1]
                indices = torch.cat([torch.arange(1), indices+1])
                genes, values = genes[indices], values[indices]
            elif len(genes) < length:
                padding = length-len(genes)
                genes = torch.cat([genes, torch.full((padding,), self.pad_token_id, dtype=genes.dtype, device=genes.device)])
                values = torch.cat([values, torch.full((padding,), self.pad_value, dtype=values.dtype, device=values.device)])
            all_genes.append(genes)
            all_values.append(values)
        return {'gene':torch.stack(all_genes), 'expr':torch.stack(all_values)}
