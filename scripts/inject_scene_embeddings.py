"""Utility to inject or write precomputed scene graph embeddings into scene pickles.

Usage examples:

# Inject embeddings from a numpy file (N, D) into pickles in a directory (sorted by name)
python scripts/inject_scene_embeddings.py --scene-pkl-dir data/scene_pickles --embeddings-npy data/embs.npy

# Inject the same embedding for all scenes (random test embedding)
python scripts/inject_scene_embeddings.py --scene-pkl-dir data/scene_pickles --random

The script will try to set a `graph_embedding` field in each pickle (dict or object attribute).
"""
import argparse
import os
import pickle
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-pkl-dir", required=True, help="Directory containing scene .pkl files")
    p.add_argument("--embeddings-npy", default=None, help=".npy file with shape (N,D) or (D,) to use")
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--random", action="store_true", help="Generate random embeddings instead of loading .npy")
    p.add_argument("--out-dir", default=None, help="If set, writes modified pickles to this directory instead of overwriting")
    args = p.parse_args()

    files = sorted([f for f in os.listdir(args.scene_pkl_dir) if f.endswith('.pkl')])
    if len(files) == 0:
        print('No .pkl files found in', args.scene_pkl_dir)
        return

    if args.embeddings_npy and not args.random:
        embs = np.load(args.embeddings_npy)
        if embs.ndim == 1:
            embs = np.tile(embs.reshape(1, -1), (len(files), 1))
        if embs.shape[0] < len(files):
            # repeat or pad
            reps = int(np.ceil(len(files) / embs.shape[0]))
            embs = np.tile(embs, (reps, 1))[: len(files)]
    elif args.random:
        rng = np.random.RandomState(0)
        embs = rng.randn(len(files), args.embed_dim).astype(np.float32)
    else:
        print('Either --embeddings-npy or --random must be provided')
        return

    out_dir = args.out_dir or args.scene_pkl_dir
    os.makedirs(out_dir, exist_ok=True)

    for i, fn in enumerate(files):
        pth = os.path.join(args.scene_pkl_dir, fn)
        try:
            with open(pth, 'rb') as fh:
                data = pickle.load(fh)
        except Exception as e:
            print('failed to load', pth, e)
            continue

        emb = embs[i]
        try:
            if isinstance(data, dict):
                data['graph_embedding'] = emb
            else:
                setattr(data, 'graph_embedding', emb)
        except Exception:
            print('failed to set embedding for', pth)
            continue

        out_p = os.path.join(out_dir, fn)
        try:
            with open(out_p, 'wb') as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('failed to write', out_p, e)
            continue

    print('done. wrote embeddings for', len(files), 'files to', out_dir)


if __name__ == '__main__':
    main()
