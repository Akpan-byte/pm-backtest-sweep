#!/usr/bin/env python3
"""Split a file list into overlapping chunks for parallel GHA jobs."""
import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="file listing (one path per line)")
    ap.add_argument("--chunks", type=int, default=5, help="number of chunks")
    ap.add_argument("--overlap", type=int, default=100, help="overlap files between adjacent chunks")
    ap.add_argument("--outdir", default=".", help="output directory for chunk files")
    args = ap.parse_args()

    with open(args.files) as f:
        files = [line.strip() for line in f if line.strip()]

    total = len(files)
    chunk_size = total // args.chunks
    remainder = total % args.chunks

    print(f"Total files: {total}, chunks: {args.chunks}, overlap: {args.overlap}")
    print(f"Base chunk size: {chunk_size}, remainder: {remainder}")

    os.makedirs(args.outdir, exist_ok=True)

    for i in range(args.chunks):
        start = i * chunk_size + min(i, remainder)
        end = (i + 1) * chunk_size + min(i + 1, remainder)
        # Add overlap: extend start backwards (except first chunk)
        if i > 0:
            start = max(0, start - args.overlap)
        # Add overlap: extend end forwards (except last chunk)
        if i < args.chunks - 1:
            end = min(total, end + args.overlap)

        chunk_files = files[start:end]
        chunk_path = os.path.join(args.outdir, f"chunk_{i}.txt")
        with open(chunk_path, "w") as f:
            f.write("\n".join(chunk_files) + "\n")

        print(f"  Chunk {i}: {len(chunk_files)} files (idx {start}-{end-1}) -> {chunk_path}")

    print("Done")

if __name__ == "__main__":
    main()
