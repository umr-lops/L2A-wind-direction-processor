"""
A Grouazel
Oct 2023
<div class="alert alert-block alert-warning">
<b>WARNING:</b>
As the bindings are not properly set up on datarmor-jupyter-hub, it is not possible to process files on
 /home/datawork-cersat-project using the singularity servers available on datarmor-jupyterhub. <br>
<b>Use the validation server</b> to be able to see the files on /home/datawork-cersat-project.
</div>

inspired from generate_L1_wind_product_v2.ipynb (R Marquart)
/raid/singularity-images/tensorflow-22.04.sif

"""
import pdb

import numpy as np
import xarray as xr

# /!\ --------- /!\ ---------- /!\ #
# /!\ ----- M64RN4 class ----!\ #
# !\ --------- /!\ ---------- /!\ #

from l2awinddirection.M64RN4 import M64RN4_distribution


def generate_wind_distribution_product(tiles, m64rn4, nb_classes=36, shape=(44, 44, 1)):
    """
    Generate wind direction from previously extracted tiles and store them in an xarray.Dataset.
    Parameters:
        tiles (xarray.Dataset): L1b path of the file to process.
        model (M64RN4_distribution): list of the M64RN4 models used for prediction.
        shape (tuple of int): shape of the inputs of the M64RN4 models.
    Returns:
        xarray.Dataset: dataset containing the tiles with associated wind direction.
    """
    try:
        tiles_stacked = tiles.stack(all_tiles=["burst", "tile_line", "tile_sample"])
    except:
        tiles_stacked = tiles.stack(all_tiles=["tile_line", "tile_sample"])
    # mask0 = ~np.any(
    #     np.isnan(tiles_stacked.sigma0), axis=(0, 1)
    # )  # original mask from Robin, sigma0 never contains NaN contrarily to sigma0_filt
    mask = tiles_stacked.land_flag == False  # test AG
    mask_indices = np.where(mask)
    tiles_stacked_no_nan = tiles_stacked.where(mask, drop=True)

    if "sigma0_filt" in tiles_stacked:
        X = tiles_stacked_no_nan.sigma0_filt.transpose(
            "all_tiles", "azimuth", "range"
        ).values
    else:
        X = tiles_stacked_no_nan.sigma0.transpose(
            "all_tiles", "azimuth", "range"
        ).values

    X_normalized = np.array(
        [((x - np.average(x)) / np.std(x)).reshape(shape) for x in X]
    )  # Normalize data

    heading_angle = np.deg2rad(tiles_stacked["ground_heading"].values)
    # Assign new coordinates 'bin_centers' to the dataset
    dx = 180 / nb_classes
    angles = np.arange(0 + dx / 2, 180 + dx / 2, dx)
    predictions = np.ones((tiles_stacked.longitude.size, angles.size)) * np.nan
    tiles_stacked_no_nan = tiles_stacked_no_nan.assign_coords(
        bin_centers=("bin_centers", angles)
    )

    # Predict wind direction distributions and add them to the dataset
    # predictions = np.squeeze(m64rn4.model.predict(X_normalized)) # original Robin
    if X_normalized.size > 0:
        predictions_usable = m64rn4.model.predict(X_normalized)
        predictions[mask_indices, :] = predictions_usable

    tiles_stacked = tiles_stacked.assign(
        wind_direction_distribution=(["all_tiles", "bin_centers"], predictions)
    )

    # Compute mean wind direction, most likely wind direction and associated standard deviation
    mean_wdir = xr.apply_ufunc(
        compute_mean_direction,
        tiles_stacked.wind_direction_distribution,
        tiles_stacked.bin_centers,
        input_core_dims=[["bin_centers"], ["bin_centers"]],
        vectorize=True,
    )
    mean_wdir = mean_wdir + heading_angle
    most_likely_wdir = xr.apply_ufunc(
        compute_most_likely_direction,
        tiles_stacked.wind_direction_distribution,
        tiles_stacked.bin_centers,
        input_core_dims=[["bin_centers"], ["bin_centers"]],
        vectorize=True,
    )
    most_likely_wdir = xr.where(tiles_stacked['land_flag'],np.nan,most_likely_wdir) # apply the land_mask on the wdin direction prediction
    most_likely_wdir = most_likely_wdir + heading_angle
    std_wdir = xr.apply_ufunc(
        compute_standard_deviation,
        tiles_stacked.wind_direction_distribution,
        tiles_stacked.bin_centers,
        input_core_dims=[["bin_centers"], ["bin_centers"]],
        vectorize=True,
    )

    # Format dataset
    tiles_stacked = tiles_stacked.assign(mean_wdir=(["all_tiles"], mean_wdir.data))
    tiles_stacked["mean_wdir"].attrs[
        "definition"
    ] = "180° ambiguous mean wind direction."
    tiles_stacked["mean_wdir"].attrs["long_name"] = "Mean wind direction"
    tiles_stacked["mean_wdir"].attrs[
        "convention"
    ] = "Clockwise, relative to geographic North."
    tiles_stacked["mean_wdir"].attrs["units"] = "degree"
    tiles_stacked["mean_wdir"].attrs["vmin"] = 0
    tiles_stacked["mean_wdir"].attrs["vmax"] = 180

    tiles_stacked = tiles_stacked.assign(
        most_likely_wdir=(["all_tiles"], most_likely_wdir.data)
    )
    tiles_stacked["most_likely_wdir"].attrs[
        "definition"
    ] = "180° ambiguous most likely wind direction, defined with a bin precision of 2.5°."
    tiles_stacked["most_likely_wdir"].attrs["long_name"] = "Most likely wind direction"
    tiles_stacked["most_likely_wdir"].attrs[
        "convention"
    ] = "Clockwise, relative to geographic North."
    tiles_stacked["most_likely_wdir"].attrs["units"] = "degree"
    tiles_stacked["most_likely_wdir"].attrs["vmin"] = 0
    tiles_stacked["most_likely_wdir"].attrs["vmax"] = 180

    tiles_stacked = tiles_stacked.assign(std_wdir=(["all_tiles"], std_wdir.data))
    tiles_stacked["std_wdir"].attrs[
        "definition"
    ] = "Standard deviation associated with wind direction."
    tiles_stacked["std_wdir"].attrs[
        "long_name"
    ] = "Standard deviation of the wind direction"
    tiles_stacked["std_wdir"].attrs["units"] = "degree"
    tiles_stacked["std_wdir"].attrs["vmin"] = 0
    tiles_stacked["std_wdir"].attrs["vmax"] = 180

    return tiles_stacked.unstack()


def compute_mean_direction(weights, angles):
    """
    Compute mean direction given an array of weights and angles taking into account the circularity between 0 and 180° of the angles.
    Parameters:
        weights (numpy.ndarray): probability corresponding to each angle.
        angles (numpy.ndarray): array of angles in degrees corresponding to the middle of the bins used for training (e.g.: [0,5] -> 2.5).
    Returns:
        float: Computed mean direction.
    """
    angles_rad = (
        angles * (2 * np.pi) / 180
    )  # Recast angles as radians that range between 0 and 2 pi
    s_a = np.sum(weights * np.sin(angles_rad))
    c_a = np.sum(weights * np.cos(angles_rad))

    mean_direction = np.arctan2(s_a, c_a)

    return (
        mean_direction * 180 / (2 * np.pi)
    )  # Convert back the result to the considered range


def compute_standard_deviation(weights, angles):
    """
    Compute standard deviation given an array of weights and angles taking into account the circularity between 0 and 180° of the angles.
    Parameters:
        weights (numpy.ndarray): probability corresponding to each angle.
        angles (numpy.ndarray): array of angles in degrees corresponding to the middle of the bins used for training (e.g.: [0,5] -> 2.5).
    Returns:
        float: Computed standard deviation.
    """
    mean_dir = compute_mean_direction(weights, angles)

    A = np.abs(angles - mean_dir)
    B = np.abs(180 - np.abs(angles - mean_dir))
    # Select the min difference between the two terms defined above to deal with 180° discontinuity
    delta = np.min(np.vstack([A, B]), axis=0)  # 180° ambiguity
    variance = np.sum(weights * delta ** 2)

    return np.sqrt(variance)


def compute_most_likely_direction(weights, angles):
    """
    Get most likely direction given an array of weights and angles.
    Parameters:
        weights (numpy.ndarray): probability corresponding to each angle.
        angles (numpy.ndarray): array of angles in degrees corresponding to the middle of the bins used for training (e.g.: [0,5] -> 2.5).
    Returns:
        float: Most likely direction corresponding to the highest probability.
    """
    return angles[np.argmax(weights)]
