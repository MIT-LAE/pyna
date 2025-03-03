import pytest
import numpy as np
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.levels import (
    compute_spl,
    compute_oaspl,
    compute_noy,
    compute_pnl,
    compute_tonal_corrections,
    compute_pnlt
)

@pytest.mark.parametrize(
    "msap, rho_0, c_0, spl_expected",
    [
        (np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]),
         1.225, 
         340.,
         np.array([53.02187846, 53.02187846, 53.02187846, 53.02187846, 53.02187846,
                   53.02187846, 53.02187846, 53.02187846, 53.02187846, 53.02187846,
                   53.02187846, 53.02187846, 53.02187846, 53.02187846, 53.02187846,
                   53.02187846, 53.02187846, 53.02187846, 53.02187846, 53.02187846,
                   53.02187846, 53.02187846, 53.02187846, 53.02187846])),
        (np.array([1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20]),
         1.225,
         340., 
         -96.978122*np.ones(24))
    ]
)
def test_compute_spl(msap, rho_0, c_0, spl_expected):

    spl = compute_spl(msap, rho_0, c_0)
    assert_quantity_almost_equal(spl, spl_expected, atol=1e-20)


@pytest.mark.parametrize(
    "spl, oaspl_expected",
    [
        (   
            np.array([150.4, 152.3, 153.0, 153.0, 152.2, 150.5, 148.2, 145.5, 143.2, 140.9, 138.7, 136.3, 133.9, 131.7, 129.4, 127.0, 124.8, 122.7, 120.4, 118.1, 115.9, 113.6, 111.2, 109.0]), 160.5
        )
    ]
)
def test_compute_oaspl(spl, oaspl_expected):

    oaspl = compute_oaspl(spl)
    assert_quantity_almost_equal(oaspl, oaspl_expected, atol=0.1)

@pytest.mark.parametrize(
    "spl, noy_expected",
    [
        (
            np.array([76.6, 76.8, 75.5, 71.3, 66.0, 77.6, 82.2, 83.2, 79.0, 75.6, 80.0, 73.9, 75.7, 74.1, 71.6, 70.2, 66.9, 64.6, 60.9, 56.8, 52.2, 45.8, 36.8, 25.2]),        # Lateral observer time 52.8 s
            [3.530391, 4.803701,  5.226422,  4.72071 ,  3.388754,  9.57983 , 15.136923, 17.387759, 13.645046, 11.794154, 16.000001, 10.483148, 11.876189, 10.629487, 10.267408, 12.149321, 11.107185, 10.879677, 9.030754,  6.805938,  4.625003,  2.775927,  1.313403,  0.178028]
        ),
        (
            np.array([77.5, 78.1, 77.4, 75.2, 70.6, 60.8, 62.5, 66.5, 66.9, 63.4, 54.1, 55.3, 51.7, 47.3, 42.2, 37.9, 32.5, 25.1, 18.1,  7.9, -4.1,-19.7,-42.3,-72.6]),            # Lateral observer time 67.8 s
            [3.863252, 5.423967, 6.140212, 6.571253, 4.927034, 2.670929, 3.548089, 5.261959, 5.805071, 5.063026, 2.657372, 2.887858, 2.250117, 1.658639, 1.337928, 1.308712, 1.035094, 0.674834, 0.416911, 0.142475, 0.      , 0.      , 0.      , 0.      ]
        )
    ]
)
def test_compute_noy(spl, noy_expected):
    
    noy = compute_noy(spl)
    assert_quantity_almost_equal(noy, noy_expected, atol=1e-6)

@pytest.mark.parametrize(
    "noy, pnl_expected",
    [
        (
            np.array([3.530391, 4.803701,  5.226422,  4.72071 ,  3.388754,  9.57983 , 15.136923, 17.387759, 13.645046, 11.794154, 16.000001, 10.483148, 11.876189, 10.629487, 10.267408, 12.149321, 11.107185, 10.879677, 9.030754,  6.805938,  4.625003,  2.775927,  1.313403,  0.178028]),
            95.17
        ),
        (
            np.array([3.863252, 5.423967, 6.140212, 6.571253, 4.927034, 2.670929, 3.548089, 5.261959, 5.805071, 5.063026, 2.657372, 2.887858, 2.250117, 1.658639, 1.337928, 1.308712, 1.035094, 0.674834, 0.416911, 0.142475, 0.      , 0.      , 0.      , 0.      ]),
            79.15
        )
    ]
)
def test_compute_pnl(noy, pnl_expected):

    pnl = compute_pnl(noy)
    assert_quantity_almost_equal(pnl, pnl_expected, atol=0.2)

@pytest.mark.parametrize(
    "spl, tonal_corrections_expected",
    [
        (
            np.array([0, 0, 70, 62, 70, 80, 82, 83, 76, 80, 80, 79, 78, 80, 78, 76, 79, 85, 79, 78, 71, 60, 54, 45]),
            [0, 0, 0, 0, 0, 0.29, 0.06, 0.61, 0, 0.17, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0.33, 0, 0, 0, 0]
        ),
    ]
)
def test_compute_tonal_corrections(spl, tonal_corrections_expected):
    
    tonal_corrections = compute_tonal_corrections(spl)
    assert_quantity_almost_equal(tonal_corrections, tonal_corrections_expected, atol=0.1)

@pytest.mark.parametrize(
    "spl, flag_tones_under_800Hz, pnlt_expected",
    [
        (
            np.array([76.6, 76.8, 75.5, 71.3, 66.0, 77.6, 82.2, 83.2, 79.0, 75.6, 80.0, 73.9, 75.7, 74.1, 71.6, 70.2, 66.9, 64.6, 60.9, 56.8, 52.2, 45.8, 36.8, 25.2]),        # Lateral observer time 52.8 s,
            False,
            95.17
        ),
        (
            np.array([77.5, 78.1, 77.4, 75.2, 70.6, 60.8, 62.5, 66.5, 66.9, 63.4, 54.1, 55.3, 51.7, 47.3, 42.2, 37.9, 32.5, 25.1, 18.1,  7.9, -4.1,-19.7,-42.3,-72.6]),        # Lateral observer time 67.8 s
            False,
            79.51
        )
    ]
)
def test_compute_pnlt(spl, flag_tones_under_800Hz, pnlt_expected):
    
    pnlt = compute_pnlt(spl, flag_tones_under_800Hz)
    assert_quantity_almost_equal(pnlt, pnlt_expected, atol=0.1)
    