# nanoBragg_noise

This tool allows synthetic diffraction data to be stored on disk as Bragg peaks alone, with individual background files for maximum compression. It is meant to facilitate the sharing of synthetic diffraction data to drive development of novel data reduction algorithms. It adds noise and background to simulated diffraction data, specifically designed for data generated using the `datsimx.sim` script with the `--noNoise` option (see [mx_simulate.py](https://github.com/pixel-modelers/datsimx/blob/main/datsimx/mx_simulate.py)), which is a thin wrapper to a nanoBragg-style forward model.

## Usage Example

To add noise to a noiseless dataset, use the `braggaddnoise` command with the following options:

```bash
braggaddnoise --dirname simulated_scan --run 3 --nj 20

--dirname: The path to the folder containing the simulated data (e.g., simulated_scan).

--run: The run number used in the datsimx simulation (e.g., 3).

--nj: The number of parallel processes to use for faster processing.
```

The above command will find all HDF5 files matching `shots_3_*h5` in the `simulated_scan` folder, add the background to them that's stored in the file `background_3.h5`, and add noise to the result, finally overwriting the existing `sim_image` datasets.

