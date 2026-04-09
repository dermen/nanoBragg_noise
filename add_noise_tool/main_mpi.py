"""
MPI version of braggaddnoise.

Usage:
    srun -n 64 braggaddnoise-mpi --dirname trials/trial779 --run 1
    srun -n 64 braggaddnoise-mpi --dirname trials/trial779 --run 1 --outdir /tmp/trial779_noisy

Each MPI rank processes a subset of H5 files. Rank 0 handles
outdir setup and file copying before broadcasting to all ranks.
"""

import argparse
import glob
import os

import numpy as np
import h5py
from mpi4py import MPI

from .main import add_noise, process_f


def get_args():
    parser = argparse.ArgumentParser(
        description="MPI version: add background and noise to simulated H5 images.",
    )

    core_group = parser.add_argument_group("Core Arguments")
    core_group.add_argument(
        "--dirname", type=str, required=True,
        help="Path to the directory containing the simulated H5 files.",
    )
    core_group.add_argument(
        "--run", type=int, required=True,
        help="The run number (e.g., 1). Files are shots_R_*h5 and background_R.h5.",
    )

    noise_group = parser.add_argument_group("Noise Parameters")
    noise_group.add_argument("--calib", type=float, default=0.03)
    noise_group.add_argument("--flicker", type=float, default=0)
    noise_group.add_argument("--readout", type=float, default=0)

    detector_group = parser.add_argument_group("Detector Parameters")
    detector_group.add_argument("--gain", type=float, default=1)

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--outdir", type=str, default=None,
        help="Write noisified H5 files here instead of overwriting originals.",
    )

    args = parser.parse_args()
    return args


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = get_args()

    # Rank 0: setup outdir and build file list
    if rank == 0:
        bgname = os.path.join(args.dirname, f"background_{args.run}.h5")
        bg = h5py.File(bgname, 'r')['background'][()]
        fnames = sorted(glob.glob(f"{args.dirname}/shots_{args.run}_*h5"))

        outdir = args.outdir
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            import shutil
            for extra in [bgname,
                          os.path.join(args.dirname, f"geom_run{args.run}.expt"),
                          os.path.join(args.dirname, f"run_{args.run}_master.h5")]:
                if os.path.exists(extra):
                    shutil.copy2(extra, outdir)

        print(f"Processing {len(fnames)} H5 files with {size} MPI ranks")
    else:
        bg = None
        fnames = None
        outdir = args.outdir

    # Broadcast shared data
    bg = comm.bcast(bg, root=0)
    fnames = comm.bcast(fnames, root=0)

    # Each rank processes its share of files
    my_files = [f for i, f in enumerate(fnames) if i % size == rank]

    for f in my_files:
        print(f"Rank {rank}: noisifying {os.path.basename(f)}")
        process_f(f, bg, args, outdir=outdir)

    comm.Barrier()
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
