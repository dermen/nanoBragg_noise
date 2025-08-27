
import argparse
import glob
import os

import numpy as np
from joblib import Parallel, delayed
import h5py


from add_noise_tool import add_noise


def get_args():
    parser = argparse.ArgumentParser(
        description="Add background and noise to a series of simulated H5 images."
    )

    # Define the command-line arguments
    parser.add_argument(
        "--dirname",
        type=str,
        required=True,
        help="Path to the directory containing the simulated H5 files."
    )
    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="The run number to identify the shot and background files (e.g., '1' or '3')."
    )
    parser.add_argument(
        "--calib",
        type=float,
        default=0.03,
        help="The calibration noise value to be passed to the 'add_noise' function. (default=0.03)"
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=10,
        help="The number of parallel processes (default=10)."
    )

    args = parser.parse_args()
    return args


def process_f(f, bg, args):

    with h5py.File(f, 'r+') as h:
        shots = h['sim_image'].keys()
        for s in shots:
            ds_name = f'sim_image/{s}'
            img = h[ds_name][()] + bg
            noise = add_noise.add_noise(img, calibration_noise=args.calib)
            del h[ds_name]
            h.create_dataset(ds_name, data=noise.astype(np.float32))


def worker_main(fnames, args, bg, njobs, jobid):
    for i_f, f in enumerate(fnames):
        if i_f % njobs != jobid:
            continue

        print(f"Worker {jobid} noisifying shots in file {f} ({i_f+1}/{len(fnames)})")
        process_f(f, bg, args)


def main():
    args = get_args()
    bgname = os.path.join(args.dirname, f"background_{args.run}.h5")
    bg = h5py.File(bgname, 'r')['background'][()]
    fnames = glob.glob(f"{args.dirname}/shots_{args.run}_*h5")
    Parallel(n_jobs=args.nj)(delayed(worker_main)(fnames, args, bg, args.nj, j) for j in range(args.nj))


if __name__ == "__main__":
    main()
