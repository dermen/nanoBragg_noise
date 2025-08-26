import numpy as np


# A simplified representation of nanoBragg add_noise
class Deviates:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def poidev(self, expected_photons):
        return self.rng.poisson(lam=expected_photons)

    def gaussdev(self, size=None):
        return self.rng.normal(loc=0.0, scale=1.0, size=size)


def add_noise_no_mask(
    raw_pixels,
    flicker_noise=0.0,
    calibration_noise=0.0,
    readout_noise=0.0,
    quantum_gain=1.0,
    adc_offset=0.0,
    seed=8675309,
    calib_seed=8675309,
    verbose=False
):
    """
    Applies various types of noise to the entire raw image, conceptually based on the provided C++ code.

    Args:
        raw_pixels (np.ndarray): The input image data (expected photons/pixel).
        flicker_noise (float): Multiplicative source noise (1/f noise).
        calibration_noise (float): Multiplicative detector calibration noise.
        readout_noise (float): Additive read-out noise.
        quantum_gain (float): Gain factor converting photons to ADU.
        adc_offset (float): ADU offset.
        seed (int): Seed for random number generation.
        calib_seed (int): Seed for calibration noise generation.
        verbose (bool): If True, prints status messages.

    Returns:
        np.ndarray: The final noisy image in ADU units.
    """

    # --- Step 0: Initial setup ---
    image = raw_pixels.astype(np.float64).copy()

    # Create the seeds for reproducibility
    seed = seed
    calib_seed = calib_seed

    # Use different RNGs for different noise types
    image_deviates_rng = Deviates(seed=seed)
    pixel_deviates_rng = Deviates(seed=calib_seed)

    # --- Step 1: Simulate 1/f and Poisson noise ---
    if verbose:
        print(f"Applying calibration at {calibration_noise * 100:.2f}%, flicker noise at {flicker_noise * 100:.2f}%")

    # Apply flicker noise to the entire image
    if flicker_noise > 0.0:
        flicker_deviates = image_deviates_rng.gaussdev(size=image.shape)
        image *= (1.0 + flicker_noise * flicker_deviates)

    # Negative photons should be invalid
    image[image < 0] = 0

    # Apply Poisson noise to the entire image
    image = image_deviates_rng.poidev(image)

    if verbose:
        total_photons_initial = image.sum()
        max_I_initial = image.max()
        print(f"{total_photons_initial:.0f} photons generated on noise image, max= {max_I_initial:.2f}")

    # --- Step 2: Apply calibration noise ---
    if calibration_noise > 0.0:
        if verbose:
            print("Applying calibration noise...")
        # Calibration noise is applied pixel by pixel
        calib_deviates = pixel_deviates_rng.gaussdev(size=image.shape)
        image = image*(1.0 + calibration_noise * calib_deviates)

    if verbose:
        total_photons_calibrated = image.sum()
        max_I_calibrated = image.max()
        print(f"{total_photons_calibrated:.0f} photons after calibration error, max= {max_I_calibrated:.2f}")

    # --- Step 3: PSF (Point Spread Function) application (Placeholder) ---
    # The C++ code calls apply_psf here.
    #

    # --- Step 4: Convert to ADU and add readout noise ---
    if verbose:
        print(f"ADU = {quantum_gain:.2f} * observed_photons + {adc_offset:.2f} + {readout_noise:.2f}")

    # Convert photons to ADU for the entire image
    adu_image = image * quantum_gain + adc_offset

    # Apply readout noise
    if readout_noise > 0.0:
        readout_deviates = image_deviates_rng.gaussdev(size=adu_image.shape)
        adu_image += readout_noise * readout_deviates

    if verbose:
        total_adu = adu_image.sum()
        max_adu = adu_image.max()
        net_adu = total_adu - adc_offset * adu_image.size
        print(f"{net_adu:.0f} net adu generated on final image, max= {max_adu:.2f}")

    return adu_image
