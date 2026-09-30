"""Export an explicitly held-out original waveform excerpt for actual platform smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wfdb

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'training' / 'mitbih_cnn_v1')
    parser.add_argument('--record', default='106')
    parser.add_argument('--seconds', type=int, default=120)
    args = parser.parse_args()
    split = json.loads((args.output / 'cache' / 'split.json').read_text(encoding='utf-8'))
    if args.record not in split['test']['record_ids']:
        raise ValueError('The platform validation example must be from a frozen held-out subject.')
    inventory = json.loads((ROOT / 'training' / 'dataset_inventory.json').read_text(encoding='utf-8'))
    entry = next(entry for entry in inventory['records'] if entry['record'] == args.record)
    source = wfdb.rdrecord(str(Path(inventory['dataset_path']) / args.record), sampto=int(args.seconds * entry['sample_rate']))
    if any(unit != 'mV' for unit in source.units):
        raise ValueError('Unexpected original units; cannot silently convert validation sample.')
    payload = {'sample_rate': source.fs, 'units': 'mV',
               'leads': {name: source.p_signal[:, index].tolist() for index, name in enumerate(source.sig_name)}}
    path = args.output / f'heldout-{args.record}-{args.seconds}s.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Exported {path}: frozen test subject {entry["subject_group"]}, physical leads {source.sig_name}, {args.seconds}s.')


if __name__ == '__main__':
    main()
