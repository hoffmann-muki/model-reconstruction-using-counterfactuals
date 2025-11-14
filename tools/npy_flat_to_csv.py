#!/usr/bin/env python3
"""
Flatten a NumPy .npy array to 2D and save as CSV, plus a small JSON file containing the original shape.
Usage:
    python tools/npy_flat_to_csv.py path/to/file.npy [--out out.csv]

If the array is already 1D or 2D, it will be saved directly (1D saved as a single column).
If >2D, the result will have shape (orig_dim0, prod(orig_dim1:)).
"""
import argparse
import numpy as np
import os
import json

def main():
    p = argparse.ArgumentParser(description='Flatten .npy to 2D CSV and write shape metadata')
    p.add_argument('npy', help='Path to input .npy file (relative or absolute)')
    p.add_argument('--out', '-o', help='Output CSV path (default: same name with .csv)', default=None)
    p.add_argument('--fmt', help='Numeric format for savetxt (default: %.6e)', default='%.6e')
    p.add_argument('--delimiter', help='CSV delimiter (default: ,)', default=',')
    p.add_argument('--allow-pickle', action='store_true', help='Allow loading pickled object arrays')
    args = p.parse_args()

    in_path = args.npy
    if not os.path.exists(in_path):
        raise SystemExit(f"Input file not found: {in_path}")

    arr = np.load(in_path, allow_pickle=args.allow_pickle)
    orig_shape = getattr(arr, 'shape', None)

    # Ensure we have a numpy array
    arr = np.asarray(arr)

    if arr.ndim == 0:
        # scalar -> write single value CSV
        flat = arr.reshape(1, 1)
    elif arr.ndim == 1:
        flat = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        flat = arr
    else:
        # flatten trailing dimensions
        flat = arr.reshape(arr.shape[0], -1)

    out_csv = args.out or os.path.splitext(in_path)[0] + '.csv'
    meta_path = os.path.splitext(out_csv)[0] + '.shape.json'

    # Save CSV
    try:
        np.savetxt(out_csv, flat, delimiter=args.delimiter, fmt=args.fmt)
    except Exception as e:
        raise SystemExit(f"Error saving CSV: {e}")

    # Save metadata
    with open(meta_path, 'w') as mf:
        json.dump({'orig_shape': orig_shape}, mf)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote shape metadata: {meta_path}")

if __name__ == '__main__':
    main()
