import pytest
from l2awinddirection.utils import get_l2_filepath

inputs_tiles_files = [
   "/tests/iw/slc/l2a/winddirection/3.7.6/tiles/S1A_IW_XSP__1SDV_20231128T035702_20231128T035729_051412_063451_3781.SAFE/tiles_s1a-iw2-slc-vv-20231128t035702-20231128t035727-051412-063451-002_L1B_xspec_IFR_3.7.6_wind.nc",
   "/tests/iw/slc/l2a/winddirection/3.7.6/tiles/S1A_IW_XSP__1SDV_20231128T035702_20231128T035729_051412_063451_3781.SAFE/s1a-iw2-slc-vv-20231128t035702-20231128t035727-051412-063451-002_L1B_xspec_IFR_3.7.6_wind.nc",
   "/tests/iw/slc/l2a/winddirection/3.7.6/tiles/S1A_IW_XSP__1SDV_20231128T035702_20231128T035729_051412_063451_3781.SAFE/l1b-s1a-iw2-slc-vv-20231128t035702-20231128t035727-051412-063451-002-c02.nc",
   "/tests/iw/slc/l2a/winddirection/3.7.6/tiles/S1A_IW_XSP__1SDV_20231128T035702_20231128T035729_051412_063451_3781.SAFE/s1a-iw2-slc-vv-20231128t035702-20231128t035727-051412-063451-002_L1B_xspec_IFR_3.8-wind.nc",
    # "/tmp/data/products/tests/iw/slc/l1b/2021/110/S1B_IW_XSP__1SDV_20210420T094117_20210420T094144_026549_032B99_2058_B03.SAFE/s1b-iw1-xsp-vv-20210420t094118-20210420t094144-026549-032b99-004_b03.nc",
    # "/tmp/data/products/tests/iw/slc/l1b/2021/110/S1B_IW_XSP__1SDV_20210420T094117_20210420T094144_026549_032B99_2058_B03.SAFE/l1c-s1b-iw1-xsp-vv-20210420t094118-20210420t094144-026549-032b99-004_b03.nc",
    "/tmp/patchestest/2016/304/S1A_IW_TIL__1SDV_20161030T162958_20161030T163028_013722_01603F_7E41_I05.SAFE/l1b-s1a-iw2-til-vv-20161030t162959-20161030t163027-013722-01603f-002-i05.nc",
    "/tmp/patchestest/2016/304/S1A_IW_TIL__1SDV_20161030T162958_20161030T163028_013722_01603F_7E41_I05.SAFE/l1b-s1a-iw2-til-vv-20161030t162959-20161030t163027-013722-01603f-i05.nc",
]
expected_l2 = [
    '/tmp/2023/332/S1A_IW_WDR__2SDV_20231128T035702_20231128T035729_051412_063451_3781_D02.SAFE/l2-s1a-iw2-wdr-vv-20231128t035702-20231128t035727-051412-063451-002-d02.nc',
    '/tmp/2016/304/S1A_IW_WDR__2SDV_20161030T162958_20161030T163028_013722_01603F_7E41_D02.SAFE/l2-s1a-iw2-wdr-vv-20161030t162959-20161030t163027-013722-01603f-002-d02.nc',
    '/tmp/2016/304/S1A_IW_WDR__2SDV_20161030T162958_20161030T163028_013722_01603F_7E41_D02.SAFE/l2-s1a-iw2-wdr-vv-20161030t162959-20161030t163027-013722-01603f-d02.nc',
]


@pytest.mark.parametrize(
    ["vignette_input", "expected_l2"],
    (
        pytest.param(inputs_tiles_files[0], expected_l2[0]),
        pytest.param(inputs_tiles_files[1], expected_l2[0]),
        pytest.param(inputs_tiles_files[2], expected_l2[0]),
        pytest.param(inputs_tiles_files[3], expected_l2[0]),
        pytest.param(inputs_tiles_files[4], expected_l2[1]),
        pytest.param(inputs_tiles_files[5], expected_l2[2]),
    ),
)
def test_outputfile_path(vignette_input, expected_l2):
    version = "D02"
    outputdir = "/tmp/"
    actual_l2_path = get_l2_filepath(
        vignette_input, version=version, outputdir=outputdir)

    print(actual_l2_path)
    assert actual_l2_path == expected_l2
