"""CLI helper to inspect saved stats (either .npy or flattened .csv with .shape.json).
Usage:
  python tools/inspect_stats.py path/to/file.{npy|csv}

If a CSV is provided, the script looks for a sibling .shape.json with key 'orig_shape'.
Prints reconstructed array shape and useful summaries (overall mean, per-ensemble, per-query, per-model).
"""
from pathlib import Path
import json
import numpy as np
import argparse

def load_and_reconstruct(path: Path):
    suf = path.suffix.lower()
    if suf == '.npy':
        arr = np.load(path)
        return arr
    elif suf == '.csv' or suf == '.txt':
        meta_p = path.with_suffix('.shape.json')
        if not meta_p.exists():
            raise FileNotFoundError(f"Missing shape metadata {meta_p}")
        meta = json.load(open(meta_p, 'r'))
        orig_shape = tuple(meta['orig_shape'])
        flat = np.loadtxt(path, delimiter=',')
        if flat.ndim == 1:
            flat = flat.reshape(1, -1)
        arr = flat.reshape(orig_shape)
        return arr
    else:
        raise ValueError('Unsupported file type: ' + suf)

def summarize(arr: np.ndarray):
    # ensure float
    arr = arr.astype(float)
    print('reconstructed shape:', arr.shape)
    overall = float(np.nanmean(arr))
    print('overall mean:', overall)

    ndim = arr.ndim
    if ndim == 1:
        print('1D array: treat as flat values')
    elif ndim == 2:
        print('2D array: interpreting as (rows, cols)')
        print('mean per row:', np.nanmean(arr, axis=1))
        print('mean per col:', np.nanmean(arr, axis=0))
    elif ndim == 3:
        E, Q, M = arr.shape
        print(f'interpreting as (ensembles, queries, models) = (E={E}, Q={Q}, M={M})')
        print('mean per-ensemble (E):', np.nanmean(arr, axis=(1,2)).tolist())
        print('mean per-query (Q):', np.nanmean(arr, axis=(0,2)).tolist())
        print('mean per-model (M):', np.nanmean(arr, axis=(0,1)).tolist())
    else:
        print(f'{ndim}D array: reporting overall mean and per-first-axis mean')
        print('mean per-first-axis:', np.nanmean(arr, axis=tuple(range(1, arr.ndim))).tolist())

def main():
    p = argparse.ArgumentParser(description='Inspect .npy or flattened .csv stats files')
    p.add_argument('path', help='relative path to .npy or .csv file')
    args = p.parse_args()
    path = Path(args.path)
    if not path.exists():
        print('file not found:', path)
        return
    try:
        arr = load_and_reconstruct(path)
    except Exception as e:
        print('error loading file:', e)
        return
    summarize(arr)

if __name__ == '__main__':
    main()
