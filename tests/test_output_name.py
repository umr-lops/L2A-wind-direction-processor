import pytest
from slcwindinversion.utils import get_l2_filepath

inputs_l1c = [
    "/tmp/data/products/tests/iw/slc/l1c/4.0.0/S1B_IW_XSP__1SDV_20210420T094117_20210420T094144_026549_032B99_2058.SAFE/s1b-iw1-xsp-vv-20210420t094118-20210420t094144-026549-032b99-004_L1B_xspec_IFR_4.0.0.nc",
    "/tmp/data/products/tests/iw/slc/l1b/2021/110/S1B_IW_XSP__1SDV_20210420T094117_20210420T094144_026549_032B99_2058_B03.SAFE/s1b-iw1-xsp-vv-20210420t094118-20210420t094144-026549-032b99-004_b03.nc",
    "/tmp/data/products/tests/iw/slc/l1b/2021/110/S1B_IW_XSP__1SDV_20210420T094117_20210420T094144_026549_032B99_2058_B03.SAFE/l1c-s1b-iw1-xsp-vv-20210420t094118-20210420t094144-026549-032b99-004_b03.nc",
]
expected_l2 = [
    '/tmp/2021/110/S1B_IW_WSP__1SDV_20210420T094117_20210420T094144_026549_032B99_2058_C02.SAFE/l2-s1b-iw1-wsp-dv-20210420t094118-20210420t094144-026549-032b99-c02.nc'
]


@pytest.mark.parametrize(
    ["l1c_fullpath", "expected_l2"],
    (
        pytest.param(inputs_l1c[0], expected_l2[0]),
        pytest.param(inputs_l1c[1], expected_l2[0]),
        pytest.param(inputs_l1c[2], expected_l2[0]),
    ),
)
def test_outputfile_path(l1c_fullpath, expected_l2):
    version = "C02"
    outputdir = "/tmp/"
    actual_l2_path = get_l2_filepath(
        l1c_fullpath, version=version, outputdir=outputdir)

    print(actual_l2_path)
    assert actual_l2_path == expected_l2
