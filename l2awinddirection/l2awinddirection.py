"""Main module.
Step 1: move tiles nc files in a local directory for GPU processing (currently prediction is not working from another volume)
Step 2: do the prediction -> generate a .nc file containing the wind direction
Step 3: move the generated file to the archive storage

"""
import glob
import os
import shutil
import xarray as xr
from tqdm import tqdm
import logging
import time
import numpy as np

from l2awinddirection.generate_L2A_winddir_pdf_product import (
    generate_wind_distribution_product,
)
from l2awinddirection.generate_L2A_winddir_regression_product import (
    generate_wind_product,
)
from l2awinddirection.utils import get_conf
from l2awinddirection.M64RN4 import M64RN4_distribution, M64RN4_regression

conf = get_conf()


def get_memory_usage():
    try:
        import resource

        memory_used_go = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0 / 1000.0
        )
    except:  # on windows resource is not usable
        import psutil

        memory_used_go = psutil.virtual_memory().used / 1000 / 1000 / 1000.0
    str_mem = "RAM usage: %1.1f Go" % memory_used_go
    return str_mem


def main():
    import argparse

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description="winddirection_prediction->L2Awinddir")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite the existing outputs [default=False]",
        required=False,
    )
    parser.add_argument(
        "--mode",
        choices=["regression", "pdf"],
        help="must must choose a NN model or regression or pdf",
    )
    parser.add_argument(
        "--l2awindirtilessafe",
        required=True,
        help="Level-2A wind direction SAFE where are stored the extracted DN tiles",
    )
    # for now the output files will be in the same directory than the input SAFE
    parser.add_argument(
        "--outputdir",
        required=True,
        help="directory where to move and store output netCDF files",
    )
    parser.add_argument(
        "--workspace",
        help="specify a workspace for tiles I/O [default is the path in localconfig.yml or config.yml]",
        required=False,
    )
    parser.add_argument(
        "--modelpdf",
        help="specify the path of the NN model to predict PDF result [default is localconfig.yml or config.yml]",
        required=False,
    )
    parser.add_argument(
        "--modelregression",
        help="specify the path of the NN model to predict float (regression) result [default is localconfig.yml or config.yml]",
        required=False,
    )
    parser.add_argument(
        "--remove-tiles",
        help="remove the tiles files in the workspace directory [default=True]",
        required=False,
        default=True,
        action="store_false",
    )
    args = parser.parse_args()
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    t0 = time.time()
    logging.info("outputdir will be: %s", args.outputdir)

    # Parameters to instantiante the models
    input_shape = (44, 44, 1)
    data_augmentation = True
    learning_rate = 1e-3
    logging.info("the mode (model) chosen is: %s", args.mode)
    if args.modelpdf is None:
        path_model_pdf = conf["model_path_pdf"]
    else:
        path_model_pdf = args.modelpdf
    if args.modelregression is None:
        dirmodelsreg = conf["models_path_regression"]
    else:
        dirmodelsreg = args.modelregression
    if args.mode == "pdf":
        n_classes = 36
        model_m64rn4 = M64RN4_distribution(input_shape, data_augmentation, n_classes)
        model_m64rn4.create_and_compile(learning_rate)

        model_m64rn4.model.load_weights(path_model_pdf)
    elif args.mode == "regression":
        model_m64rn4 = []
        # path_best_models = glob.glob(
        #     ".../analysis/s1_data_analysis/project_rmarquar/wsat/trained_models/iw/*.hdf5"
        # )
        path_best_models = glob.glob(os.path.join(dirmodelsreg, "*.hdf5"))
        for path in path_best_models:

            m64rn4_reg = M64RN4_regression(input_shape, data_augmentation)
            m64rn4_reg.create_and_compile(learning_rate)

            m64rn4_reg.model.load_weights(path)

            model_m64rn4.append(m64rn4_reg)
    else:
        raise Exception("not handled case")
    # files = glob.glob("/raid/localscratch/agrouaze/tiles_iw_4_wdir/3.1/*SAFE/*.nc")
    if args.workspace is None:
        workspace = conf["workspace_prediction"]
    else:
        workspace = args.workspace
    logging.info(
        "workspace where the tiles will be temporarily moved is: %s",
        workspace,
    )
    safefile = os.path.join(workspace, os.path.basename(args.l2awindirtilessafe))
    logging.info(" step 1: move %s -> %s", args.l2awindirtilessafe, safefile)
    shutil.move(args.l2awindirtilessafe, safefile)
    files = glob.glob(os.path.join(safefile, "*.nc"))
    logging.info("Number of files to process: %d" % len(files))
    logging.info("step 2: predictions")
    for ii in tqdm(range(len(files))):
        file = files[ii]
        #     for file in files:
        outputfile = file.replace(
            "_wind.nc", "_winddirection.nc"
        )  # TODO: change to _tiles.nc -> _winddirection.nc
        tiles = xr.open_dataset(file)
        if len(tiles.variables.keys()) == 0:
            continue
        if args.mode == "pdf":
            res = generate_wind_distribution_product(
                tiles, model_m64rn4, nb_classes=36, shape=(44, 44, 1)
            )
            res.to_netcdf(outputfile)
        elif args.mode == "regression":
            res = generate_wind_product(tiles, model_m64rn4)
            res.to_netcdf(file.replace(".nc", "_wind.nc"))
        # if you want to remove the file containing the tiles used for inference (data kept in final product)
        del tiles
        if args.remove_tiles:
            logging.info("remove temporary tiles file in the workspace: %s", file)
            os.remove(file)
    final_safe_path = os.path.join(
        args.outpurdir, os.path.basename(args.l2awindirtilessafe)
    )
    if not os.path.exists(os.path.dirname(final_safe_path)):
        logging.info("mkdir %s", os.path.dirname(final_safe_path))
        os.makedirs(os.path.dirname(final_safe_path))
    logging.info("step 3: move %s -> %s", safefile, final_safe_path)
    shutil.move(safefile, final_safe_path)
    logging.info("Ifremer Level-2A wind direction SAFE path: %s", final_safe_path)
    logging.info("successful SAFE processing")
    logging.info("peak memory usage: %s ", get_memory_usage())
    logging.info("done in %1.3f min", (time.time() - t0) / 60.0)
