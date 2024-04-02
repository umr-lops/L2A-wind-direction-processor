
import l2awinddirection
from yaml import load
import logging
import os
from yaml import CLoader as Loader
import datetime
def get_conf():
    """

    Returns:
        conf: dict
    """
    local_config_pontential_path = os.path.join(os.path.dirname(l2awinddirection.__file__), 'localconfig.yaml')

    if os.path.exists(local_config_pontential_path):
        config_path = local_config_pontential_path
    else:
        config_path = os.path.join(os.path.dirname(l2awinddirection.__file__), 'config.yaml')
    logging.info('config path: %s', config_path)
    print('config path:', config_path)
    stream = open(config_path, 'r')
    conf = load(stream, Loader=Loader)
    return conf


def get_l2_filepath(vignette_fullpath, version, outputdir)->str:
    """

    Args:
        vignette_fullpath: str .nc full path of the sigma0 extracted on the level-1b tiles
        version : str (eg. 1.2 or D01)
        outputdir: str directory where the l2 wdr will be stored
    Returns:
        l2_full_path: str l2 wdr path
    """
    safe_file = os.path.basename(os.path.dirname(vignette_fullpath))
    pathout_root = outputdir

    safe_start_date = datetime.datetime.strptime(safe_file.split('_')[5],'%Y%m%dT%H%M%S')
    pathout = os.path.join(pathout_root, safe_start_date.strftime('%Y'),safe_start_date.strftime('%j'))

    # SAFE part
    safe_file = safe_file.replace("L1B", "L2").replace('XSP', 'WDR').replace('TIL','WDR')
    safe_file = safe_file.replace('_1S','_2S')
    if (
        len(safe_file.split("_")) == 10
    ):  # classical ESA SLC naming #:TODO once xsarslc will be updated this case could be removed
        safe_file = safe_file.replace(".SAFE", "_" + version.upper() + ".SAFE")
    else:  # there is already a product ID in the L1B SAFE name
        lastpart = safe_file.split("_")[-1]
        safe_file = safe_file.replace(lastpart, version.upper() + ".SAFE")
    # if '1SDV' in safe_file:
    #     pola_str = 'dv'
    # elif '1SDH' in safe_file:
    #     pola_str = 'dh'
    # elif '1SSV' in safe_file:
    #     pola_str = 'sv'
    # elif '1SSH' in safe_file:
    #     pola_str = 'sh'
    # else:
    #     raise Exception('safe file polarization is not defined as expected')

    # measurement part
    base_measu = os.path.basename(vignette_fullpath)


    # lastpiece = base_measu.split("-")[-1]
    if 'IFR_' in base_measu: # there is the old string _L1B_xspec_IFR_3.7.6_wind.nc
        start_IFR = base_measu.index('_L1B_xspec_IFR')
        piece_to_remove = base_measu[start_IFR:]
        base_measu = base_measu.replace(piece_to_remove,'.nc')
    if base_measu.split('-')[-1][0] == 'c' or base_measu.split('-')[-1][0] == 'i': #there is already a version product ID
        base_measu = base_measu.replace(base_measu.split('-')[-1],version.lower() + ".nc")
    else:
        base_measu = base_measu.replace('.nc','-'+version.lower() + ".nc")

    # handle the begining of the measurement basename
    if base_measu[0:3]=='l1b':
        base_measu = base_measu.replace('l1b-','l2-')
    elif base_measu[0:3]=='til':
        base_measu = base_measu.replace('tiles_', 'l2-')
    else:
        base_measu = 'l2-'+base_measu
    # base_measu = base_measu.replace(base_measu.split('-')[4], pola_str) # replace -vv- by -dv- or -sv- depending on SAFE information
    base_measu = base_measu.replace('-xsp-','-wdr-')
    base_measu = base_measu.replace('-til-','-wdr-')
    base_measu = base_measu.replace('-slc-', '-wdr-')


    l2_full_path = os.path.join(pathout, safe_file,base_measu)
    logging.debug("File out: %s ", l2_full_path)
    if not os.path.exists(os.path.dirname(l2_full_path)):
        os.makedirs(os.path.dirname(l2_full_path), 0o0775)
    return l2_full_path
