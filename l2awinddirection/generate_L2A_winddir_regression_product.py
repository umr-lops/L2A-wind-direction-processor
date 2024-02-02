"""
inspired from predict_wind_direction.py (project_rmarquart)
January 2024
"""
import numpy as np
import scipy as sp
import xarray as xr
from scipy import stats
import glob
import os, sys
from tqdm import tqdm
from l2awinddirection.M64RN4 import M64RN4_regression

# --------------------- #
# ----- Functions ----- #
# --------------------- #

def generate_wind_product(tiles, model_regs, shape = (44, 44, 1)):
    """
    Generate wind direction from previously extracted tiles and store them in an xarray.Dataset.
    Parameters:
        tiles (xarray.Dataset): L1b path of the file to process.
        model_regs (list of M64RN4): list of the M64RN4 models used for prediction.
        shape (tuple of int): shape of the inputs of the M64RN4 models.
    Returns:
        (xarray.Dataset): dataset containing the tiles with associated wind direction.
    """
    tiles_stacked = tiles.stack(all_tiles = ['burst', 'tile_line','tile_sample'])
    tiles_stacked_no_nan = tiles_stacked.where(~np.any(np.isnan(tiles_stacked.sigma0), axis = (0, 1)), drop = True)
    X = tiles_stacked_no_nan.sigma0.transpose('all_tiles', 'azimuth', 'range').values
    X_normalized = np.array([((x - np.average(x))/np.std(x)).reshape(shape) for x in X]) # Normalize data
    
    heading_angle = np.deg2rad(tiles_stacked_no_nan['ground_heading'].values) 
    
    #Predict wind directions using all models and obtain mean and std values
    predictions = launch_prediction(X_normalized, model_regs)
    mean_prediction = sp.stats.circmean(predictions, np.pi, axis = 1)
    std_prediction = sp.stats.circstd(predictions, np.pi, axis = 1)
    
    predicted_wdir = mean_prediction + heading_angle
    
    # Format dataset
    tiles_stacked_no_nan = tiles_stacked_no_nan.assign(wind_direction = (['all_tiles'],  np.rad2deg(predicted_wdir)%180))
    tiles_stacked_no_nan['wind_direction'].attrs['definition'] = '180° ambiguous wind direction, clockwise, relative to geographic North.'
    tiles_stacked_no_nan['wind_direction'].attrs['units'] = 'degree'
    tiles_stacked_no_nan['wind_direction'].attrs['vmin'] = 0
    tiles_stacked_no_nan['wind_direction'].attrs['vmax'] = 180
    
    tiles_stacked_no_nan = tiles_stacked_no_nan.assign(wind_std = (['all_tiles'],  np.rad2deg(std_prediction)%180))
    tiles_stacked_no_nan['wind_std'].attrs['definition'] = 'Standard deviation associated with wind direction. Calculated by computing the spread accross the '
    tiles_stacked_no_nan['wind_std'].attrs['units'] = 'degree'
    tiles_stacked_no_nan['wind_std'].attrs['vmin'] = 0
    tiles_stacked_no_nan['wind_std'].attrs['vmax'] = 180

    return tiles_stacked_no_nan.unstack()
    

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


if __name__ == "__main__":

    #Parameters to instantiante the models
    input_shape = (44, 44, 1)
    data_augmentation = True
    learning_rate = 1e-3

    model_regs = []

    #Modify path_best_models depending of the acquisition mode (iw, ew, wv)
    path_best_models = glob.glob('.../analysis/s1_data_analysis/project_rmarquar/wsat/trained_models/iw/*.hdf5')
    for path in path_best_models:

        m64rn4_reg = M64RN4_regression(input_shape, data_augmentation)
        m64rn4_reg.create_and_compile(learning_rate)

        m64rn4_reg.model.load_weights(path)

        model_regs.append(m64rn4_reg)


    #Get listing of files from which to generate the wind direction
    files = glob.glob('.../analysis/s1_data_analysis/project_rmarquar/slc_wind_product/1.4k/*/tiles*1.4k.nc')
    print("Number of files to process: %d" % len(files))

    for file in tqdm(files):
        tiles = xr.open_dataset(file)
        if len(tiles.variables.keys()) == 0:
            continue
        res = generate_wind_product(tiles, model_regs)
        res.to_netcdf(file.replace('.nc', '_wind.nc'))
        # if you want to remove the file containing the tiles used for inference (data kept in final product)
        del tiles
        os.remove(file)