import pytest
import numpy as np
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.source import (
    compute_fan_source_noise,
    compute_core_source_noise,
    compute_jet_mixing_source_noise,
    compute_jet_shock_source_noise,
)
from pyna.levels import compute_spl

@pytest.mark.parametrize(
    "dTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star, \
     blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing, \
     theta, M_0, c_0, T_0, rho_0, \
     n_harmonics, n_engines, \
     f, \
     method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment, \
     flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression, \
     spl_fan_expected", 
     [
         (
            0.17984723, 
            0.42268729, 
            0.41198041, 
            0.91000581, 
            1.1282707, 
            25, 
            48, 
            1.68, 
            300., 
            90.,
            0.31130, 
            343.75344, 
            298.15,
            1.1156405329761, 
            10, 
            3, 
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]), 
            "geae", 
            "alliedsignal", 
            "inlet",
            "takeoff",
            False,      # Broadband
            True,       # Tones
            False, 
            False, 
            False, 
            False,
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 130.8,  0.0,  0.0, 121.6,  0.0, 118.8]
         ),
         (
            0.17984723, 
            0.42268729, 
            0.41198041, 
            0.91000581, 
            1.1282707, 
            25, 
            48, 
            1.68, 
            300., 
            10.,
            0.31130, 
            343.75344, 
            298.15,
            1.1156405329761, 
            10, 
            3, 
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]), 
            "geae", 
            "alliedsignal", 
            "inlet",
            "takeoff",
            False,      # Broadband
            True,       # Tones
            False, 
            False, 
            False, 
            True,
            [0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 144.7,   0.0,   0.0, 135.5] 
         ),
         (
            0.17984723, 
            0.42268729, 
            0.41198041, 
            0.91000581, 
            1.1282707, 
            25, 
            48, 
            1.68, 
            300., 
            170.,
            0.31130, 
            343.75344, 
            298.15,
            1.1156405329761, 
            10, 
            3, 
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]), 
            "geae", 
            "alliedsignal", 
            "inlet",
            "takeoff",
            False,      # Broadband
            True,       # Tones
            False, 
            False, 
            False, 
            False,
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 98.2,  0.0,  0.0, 89.0,  0.0, 86.2, 83.2] 
         ),
     ]        
)
def test_compute_fan_tone_source_noise(dTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star,
                                  blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing,
                                  theta, M_0, c_0, T_0, rho_0,
                                  n_harmonics, n_engines,
                                  f,
                                  method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment,
                                  flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression,
                                  spl_fan_expected):

    # Test fan source noise module using 55t Depart-Standard at source time = 67.96 s
    msap_fan = compute_fan_source_noise(dTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star,
                                        blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing,
                                        theta, M_0, c_0, T_0, rho_0,
                                        n_harmonics, n_engines,
                                        f,
                                        method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment, 
                                        flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression)
    spl_fan = compute_spl(msap_fan, rho_0, c_0)

    assert_quantity_almost_equal(spl_fan, spl_fan_expected, atol=3.0)

@pytest.mark.parametrize(
    "dTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star, \
     blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing, \
     theta, M_0, c_0, T_0, rho_0, \
     n_harmonics, n_engines, \
     f, \
     method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment, \
     flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression, \
     spl_fan_expected", 
     [
         (
            0.17984723, 
            0.42268729, 
            0.41198041, 
            0.91000581, 
            1.1282707, 
            25, 
            48, 
            1.68, 
            300., 
            90.,
            0.31130, 
            343.75344, 
            298.15,
            1.1156405329761, 
            10, 
            3, 
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]), 
            "geae", 
            "alliedsignal", 
            "inlet",
            "takeoff",
            True,      # Broadband
            False,       # Tones
            False, 
            False, 
            False, 
            False,
            [28.1,  36.2,  44.0,  51.0,  57.7,  64.7,  70.6,  76.2,  81.6,  86.8,  91.3,  95.6,  99.6, 103.0, 106.1, 109.1, 111.4, 113.4, 115.1, 116.5, 117.4, 117.9, 118.1, 118.0]
         ),
         (
            0.17984723, 
            0.42268729, 
            0.41198041, 
            0.91000581, 
            1.1282707, 
            25, 
            48, 
            1.68, 
            300., 
            10.,
            0.31130, 
            343.75344, 
            298.15,
            1.1156405329761, 
            10, 
            3, 
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]), 
            "geae", 
            "alliedsignal", 
            "inlet",
            "takeoff",
            True,      # Broadband
            False,       # Tones
            False, 
            False, 
            False, 
            False,
            [ 35.0,  43.6,  52.1,  59.7,  66.9,  74.5,  81.0,  87.2,  93.2,  99.0, 104.1, 109.0, 113.6, 117.6, 121.2, 124.9, 127.8, 130.3, 132.6, 134.6, 136.1, 137.2, 138.0, 138.4]
         ),
         (
            0.17984723, 
            0.42268729, 
            0.41198041, 
            0.91000581, 
            1.1282707, 
            25, 
            48, 
            1.68, 
            300., 
            170.,
            0.31130, 
            343.75344, 
            298.15,
            1.1156405329761, 
            10, 
            3, 
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]), 
            "geae", 
            "alliedsignal", 
            "inlet",
            "takeoff",
            True,      # Broadband
            False,     # Tones
            False, 
            False, 
            False, 
            False,
            [-10.3, -2.7,  4.8, 11.3, 17.6, 24.1, 29.6, 34.7, 39.7, 44.5, 48.6, 52.4, 56.0, 59.0, 61.7, 64.2, 66.1, 67.7, 68.9, 69.9, 70.3, 70.5, 70.2, 69.6]
         ),
     ]        
)
def test_compute_fan_broadband_source_noise(dTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star,
                                  blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing,
                                  theta, M_0, c_0, T_0, rho_0,
                                  n_harmonics, n_engines,
                                  f,
                                  method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment,
                                  flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression,
                                  spl_fan_expected):

    # Test fan source noise module using 55t Depart-Standard at source time = 67.96 s
    msap_fan = compute_fan_source_noise(dTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star,
                                        blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing,
                                        theta, M_0, c_0, T_0, rho_0,
                                        n_harmonics, n_engines,
                                        f,
                                        method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment, 
                                        flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression)
    spl_fan = compute_spl(msap_fan, rho_0, c_0)

    assert_quantity_almost_equal(spl_fan, spl_fan_expected, atol=0.5)


@pytest.mark.parametrize(
    "mdot_i_c_star, Tt_i_c_star, Tt_j_c_star, Pt_i_c_star, DTt_design_c_star, theta, M_0, c_0, rho_0, f, n_engines, spl_core_expected",
    [
        (
            0.075252, 
            2.4374, 
            5.0731, 
            17.654, 
            2.7073,
            90., 
            0.31130, 
            343.75344, 
            1.1156405329761,
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),  
            3, 
            np.array([103.9, 107.9, 112.1, 115.3, 118.1, 121.2, 123.5, 125.5, 127.0, 128.1, 127.1, 125.6, 123.5, 121.2, 118.4, 115.3, 112.1, 108.2, 104.2, 100.6,  96.7,  92.2,  88.0,  83.2])
        ),
        (
            0.075252, 
            2.4374, 
            5.0731, 
            17.654, 
            2.7073,
            10., 
            0.31130, 
            343.75344, 
            1.1156405329761,
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            3, 
            np.array([98.0, 101.9, 105.7, 109.6, 113.2, 116.6, 119.4, 122.0, 124.3, 126.1, 127.4, 127.6, 126.4, 124.7, 122.7, 119.9, 117.1, 114.1, 110.5, 106.4, 102.7,  99.0,  94.6,  90.5])
        ),
        (
            0.075252, 
            2.4374, 
            5.0731, 
            17.654,
            2.7073, 
            170., 
            0.31130, 
            343.75344, 
            1.1156405329761,
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            3, 
            np.array([97.9, 101.8, 105.2, 108.0, 110.8, 113.3, 115.1, 116.5, 117.3, 116.2, 114.6, 112.6, 110.0, 107.2, 104.3, 100.7,  96.8,  93.0,  89.4,  85.2,  80.9,  76.7,  71.5,  66.7])
        )
    ]
)                               
def test_compute_core_source_noise(mdot_i_c_star, Tt_i_c_star, Tt_j_c_star, Pt_i_c_star, DTt_design_c_star, theta, M_0, c_0, rho_0, f, n_engines, spl_core_expected):

    # Test core source noise module using 55t Depart-Standard at source time = 67.96 s
    msap_core = compute_core_source_noise(mdot_i_c_star, Tt_i_c_star, Tt_j_c_star, Pt_i_c_star, DTt_design_c_star, theta, M_0, f, n_engines)
    spl_core = compute_spl(msap_core, rho_0, c_0)

    assert_quantity_almost_equal(spl_core, spl_core_expected, atol=5.2e-1)


@pytest.mark.parametrize(
    "V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, delta, M_0, c_0, rho_0, f, n_engines, spl_jet_mixing_expected",
    [
        (
            1.0585, 
            0.72454, 
            0.60222, 
            1.5929, 
            90., 
            0.,
            0.31130, 
            343.75344, 
            1.1156405329761,
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),  
            3, 
            np.array([135.6, 137.0, 138.1, 138.9, 139.5, 139.8, 139.9, 139.8, 139.6, 139.0, 138.5, 137.9, 137.3, 136.6, 135.7, 134.7, 133.7, 132.8, 131.8, 130.5, 129.3, 128.0, 126.7, 125.5])
        ),
        (
            1.0585, 
            0.72454, 
            0.60222, 
            1.5929, 
            10., 
            0.,
            0.31130, 
            343.75344, 
            1.1156405329761,
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            3, 
            np.array([132.5, 133.9, 135.0, 135.8, 136.4, 136.7, 136.8, 136.7, 136.5, 135.9, 135.4, 134.8, 134.2, 133.5, 132.6, 131.6, 130.6, 129.7, 128.7, 127.4, 126.2, 125.0, 123.7, 122.4])
        ),
        (
            1.0585, 
            0.72454, 
            0.60222, 
            1.5929, 
            170., 
            0.,
            0.31130, 
            343.75344, 
            1.1156405329761,
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            3, 
            np.array([140.5, 140.9,140.3,139.0,137.0,134.4,132.1,130.0,127.9,125.6,123.4,121.2,118.9,116.8,114.7,112.4,110.3,108.3,106.1,103.9,101.8, 99.6, 97.4, 95.3])
        )
    ]
)                                      
def test_compute_jet_mixing_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, delta, M_0, c_0, rho_0, f, n_engines, spl_jet_mixing_expected):

    # Test jet mixing source noise module using 55t Depart-Standard at source time = 67.96 s
    msap_jet_mixing = compute_jet_mixing_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, delta, M_0, c_0, f, n_engines)
    spl_jet_mixing = compute_spl(msap_jet_mixing, rho_0, c_0)

    assert_quantity_almost_equal(spl_jet_mixing, spl_jet_mixing_expected, atol=1e-1)


@pytest.mark.parametrize(
    "V_j_star, M_j_star, A_j_star, Tt_j_star, theta, delta, M_0, c_0, rho_0, f, n_engines, n_shock, msap_jet_shock_expected",
    [
        (1.0585, 
         1.2, 
         0.60222, 
         1.5929, 
         90., 
         0.,
         0.31130, 
         343.75344, 
         1.1156405329761,
         np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
         3,
         8,
         np.array([134.09207359, 137.19131432, 139.86477659, 141.91216342,
                   143.26672494, 144.70692682, 147.29713692, 152.83348526,
                   159.51614615, 161.50316897, 157.20886573, 157.31220895,
                   158.68293201, 156.09670451, 155.37145031, 154.41027222,
                   153.16400365, 151.88766964, 150.52031529, 149.35796059,
                   148.03946627, 146.75725837, 145.55435562, 144.35687282])
        )
    ]
)
def test_compute_jet_shock_source_noise(V_j_star, M_j_star, A_j_star, Tt_j_star, theta, delta, M_0, c_0, rho_0, f, n_engines, n_shock, msap_jet_shock_expected):

    msap_jet_shock = compute_jet_shock_source_noise(V_j_star, M_j_star, A_j_star, Tt_j_star, theta, delta, M_0, c_0, f, n_engines, n_shock)
    spl_jet_shock = compute_spl(msap_jet_shock, rho_0, c_0)

    assert_quantity_almost_equal(spl_jet_shock, msap_jet_shock_expected)
