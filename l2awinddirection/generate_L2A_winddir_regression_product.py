"""
inspired from predict_wind_direction.py (project_rmarquart)
January 2024
"""
import pdb
import numpy as np
import scipy as sp
from scipy import stats


# --------------------- #
# ----- Functions ----- #
# --------------------- #


def generate_wind_product(tiles, model_regs, shape=(44, 44, 1)):
    """
    Generate wind direction from previously extracted tiles and store them in an xarray.Dataset.
    Parameters:
        tiles (xarray.Dataset): L1b path of the file to process.
        model_regs (list of M64RN4): list of the M64RN4 models used for prediction.
        shape (tuple of int): shape of the inputs of the M64RN4 models.
    Returns:
        (xarray.Dataset): dataset containing the tiles with associated wind direction.
    """
    try:
        tiles_stacked = tiles.stack(all_tiles=["burst", "tile_line", "tile_sample"])
    except:
        tiles_stacked = tiles.stack(all_tiles=["tile_line", "tile_sample"])

    mask = tiles_stacked.land_flag == False
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
    predictions = np.ones((tiles_stacked.longitude.size,len(model_regs))) * np.NaN
    # Predict wind directions using all models and obtain mean and std values
    if X_normalized.size > 0:
        predictions_usable = launch_prediction(X_normalized, model_regs)
        predictions[mask_indices,:] = predictions_usable

    mean_prediction = sp.stats.circmean(predictions, np.pi, axis=1)
    std_prediction = sp.stats.circstd(predictions, np.pi, axis=1)

    predicted_wdir = mean_prediction + heading_angle

    # Format dataset
    tiles_stacked = tiles_stacked.assign(
        wind_direction=(["all_tiles"], np.rad2deg(predicted_wdir) % 180)
    )
    tiles_stacked["wind_direction"].attrs[
        "definition"
    ] = "180° ambiguous wind direction, clockwise, relative to geographic North."
    tiles_stacked["wind_direction"].attrs["units"] = "degree"
    tiles_stacked["wind_direction"].attrs["vmin"] = 0
    tiles_stacked["wind_direction"].attrs["vmax"] = 180

    tiles_stacked = tiles_stacked.assign(
        wind_std=(["all_tiles"], np.rad2deg(std_prediction) % 180)
    )
    tiles_stacked["wind_std"].attrs[
        "definition"
    ] = "Standard deviation associated with wind direction. Calculated by computing the spread accross the "
    tiles_stacked["wind_std"].attrs["units"] = "degree"
    tiles_stacked["wind_std"].attrs["vmin"] = 0
    tiles_stacked["wind_std"].attrs["vmax"] = 180
    return tiles_stacked.unstack()


def launch_prediction(X_normalized, model_regs):
    """
    Predicts wind direction of the given vector.
    Parameters:
        X_normalized (numpy.array): normalized array used for prediction.
        model_regs (list of M64RN4): list of the M64RN4 models used for prediction.
    Returns:
        predictions (numpy.array): array of predictions.
    """
    predictions = np.zeros((X_normalized.shape[0], len(model_regs)))

    for i, m64rn4 in enumerate(model_regs):
        predictions[:, i] = np.squeeze(m64rn4.model.predict(X_normalized))

    return predictions
