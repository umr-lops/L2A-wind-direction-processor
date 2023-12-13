"""Main module."""
import glob
import os
from l2awinddirection.M64RN4 import M64RN4_distribution
import xarray as xr
from tqdm import tqdm
import logging
import time
import numpy as np
from l2awinddirection.generate_L2A_winddir_pdf_product import generate_wind_distribution_product
from l2awinddirection.utils import get_conf
conf = get_conf()
def get_memory_usage():
    try:
        import resource
        memory_used_go = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000. / 1000.
    except:  # on windows resource is not usable
        import psutil
        memory_used_go = psutil.virtual_memory().used / 1000 / 1000 / 1000.
    str_mem = 'RAM usage: %1.1f Go' % memory_used_go
    return str_mem

def main():
    import argparse
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description="winddirection_pdf_prediction->L2Awinddir")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite the existing outputs [default=False]",
        required=False,
    )
    parser.add_argument(
        "--l2awindirtilessafe", required=True, help="Level-2A wind direction SAFE where are stored the extracted DN tiles"
    )
    # for now the output files will be in the same directory than the input SAFE
    # parser.add_argument(
    #     "--outputdir",
    #     required=True,
    #     help="directory where to store output netCDF files",
    # )
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
    logging.info("product version to produce: %s", args.version)
    logging.info("outputdir will be: %s", args.outputdir)



    # Parameters to instantiante the models
    input_shape = (44, 44, 1)
    data_augmentation = True
    learning_rate = 1e-3
    n_classes = 36

    model_m64rn4 = M64RN4_distribution(input_shape, data_augmentation, n_classes)
    model_m64rn4.create_and_compile(learning_rate)
    path_model = conf['model_path']
    model_m64rn4.model.load_weights(path_model)
    # files = glob.glob("/raid/localscratch/agrouaze/tiles_iw_4_wdir/3.1/*SAFE/*.nc")
    files = glob.glob(os.path.join(args.l2awindirtilessafe,'*.nc'))
    logging.info("Number of files to process: %d" % len(files))
    for file in tqdm(files):
        outputfile = file.replace("_wind.nc", "_winddirection.nc") # TODO: change to _tiles.nc -> _winddirection.nc
        tiles = xr.open_dataset(file)
        if len(tiles.variables.keys()) == 0:
            continue
        res = generate_wind_distribution_product(
            tiles, model_m64rn4, nb_classes=36, shape=(44, 44, 1)
        )
        res.to_netcdf(outputfile)
        # if you want to remove the file containing the tiles used for inference (data kept in final product)
        del tiles
        os.remove(file)

    logging.info("Ifremer Level-2A wind direction SAFE path: %s", args.l2awindirtilessafe)
    logging.info("successful SAFE processing")
    logging.info("peak memory usage: %s ", get_memory_usage())
    logging.info("done in %1.3f min", (time.time() - t0) / 60.0)
