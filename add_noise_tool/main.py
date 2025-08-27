
import argparse
import glob
import os

import numpy as np
from joblib import Parallel, delayed
import h5py


def get_args():
    parser = argparse.ArgumentParser(
        description="Add background and noise to a series of simulated H5 images to make the data more portable",
    )

    # Core Arguments
    core_group = parser.add_argument_group("Core Arguments")
    core_group.add_argument(
        "--dirname",
        type=str,
        required=True,
        help="Path to the directory containing the simulated H5 files."
    )
    core_group.add_argument(
        "--run",
        type=int,
        required=True,
        help="The run number to identify the shot and background files (e.g., '1' or '3'). Files are stored as shots_R_*h5 and background_R.h5 ."
    )
    core_group.add_argument(
        "--nj",
        type=int,
        default=10,
        help="The number of parallel processes (default=10)."
    )

    # Noise Parameters
    noise_group = parser.add_argument_group("Noise Parameters")
    noise_group.add_argument(
        "--calib",
        type=float,
        default=0.03,
        help="The calibration noise value to be passed to the 'add_noise' function. (default=0.03)"
    )
    noise_group.add_argument(
        "--flicker",
        type=float,
        default=0,
        help="The flicker noise level in the source. (default=0)"
    )
    noise_group.add_argument(
        "--readout",
        type=float,
        default=0,
        help="The readout noise level in detector ADUs. (default=0)"
    )

    # Detector Parameters
    detector_group = parser.add_argument_group("Detector Parameters")
    detector_group.add_argument(
        "--gain",
        type=float,
        default=1,
        help="The quantum gain to convert photons to detector ADUs. (default=1)"
    )

    args = parser.parse_args()

    return args


def add_noise(img, poiss_seed=8675309, gauss_seed=8675309, calib_seed=8675309,
              flicker_noise=0, calib_noise=0, readout_noise=0,
              quantum_gain=1, adc_offset=0):
    """
    functional port of nanoBragg add noise
    see add_noise in
        https://github.com/cctbx/cctbx_project/blob/master/simtbx/nanoBragg/nanoBragg.cpp

    :param img: expected number of photons in each pixel
    :param poiss_seed: seed for rng
    :param gauss_seed: seed for rng
    :param flicker_noise: noise level in the source (should be on range 0-1)
    :param calib_noise: noise level in the pixel response (should be on range 0-1)
    :param readout_noise: noise level in detector readout (should be in ADUs)
    :param quantum_gain: convert from photons to area detector units
    :param adc_offset: baseline offset
    :return: img in ADU
    """
    poiss_rng = np.random.default_rng(poiss_seed)  # different for all images
    gauss_rng = np.random.default_rng(gauss_seed)  # different for all images
    calib_rng = np.random.default_rng(calib_seed)  # same for all images

    def gauss_noise(img, noise_level, how="mult"):
        if noise_level > 0:
            deviates = gauss_rng.normal(loc=0, scale=1, size=img.shape)
            if how=="mult":
                img = img * (1 + noise_level*deviates)
            else:
                img = img + noise_level*deviates
        return img

    # negative values in the simulated data mark detector gaps and bad pixels, so ignore those
    finite = img >= 0
    img_f = img[finite]

    # simulate photons arriving at detector
    img_f = gauss_noise(img_f, flicker_noise, how="mult")
    img_f = poiss_rng.poisson(lam=img_f)

    # simulate detector calibration noise
    img_f = gauss_noise(img_f, calib_noise, how="mult")

    # convert to detector units and add readout noise
    img_f = quantum_gain*img_f + adc_offset
    img_f = gauss_noise(img_f, readout_noise, how="add")
    img[finite] = img_f

    return img


def process_f(f, bg, args):

    with h5py.File(f, 'r+') as h:
        shots = h['sim_image'].keys()
        for s in shots:
            ds_name = f'sim_image/{s}'
            shot_num = int(s.split("_")[-1])
            img = h[ds_name][()] + bg
            noise = add_noise(img, calib_noise=args.calib, poiss_seed=shot_num, gauss_seed=shot_num,
                              flicker_noise=args.flicker, readout_noise=args.readout, quantum_gain=args.gain)
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
