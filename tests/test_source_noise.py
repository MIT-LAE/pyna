import pytest
import numpy as np
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.source import (
    compute_fan_source_noise,
    compute_core_source_noise,
    compute_jet_mixing_source_noise,
    compute_jet_shock_source_noise,
    compute_airframe_source_noise
)
from pyna.levels import compute_spl

# Fan noise modules
@pytest.mark.parametrize(
    "dTt_fan_star, mdot_fan_star, N_fan_star, A_fan_star, d_fan_star, \
     blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing, \
     theta, M_0, c_0, T_0, rho_0, \
     n_harmonics, n_engines, \
     frequency, \
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
def test_compute_fan_tone_source_noise(dTt_fan_star, mdot_fan_star, N_fan_star, A_fan_star, d_fan_star,
                                  blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing,
                                  theta, M_0, c_0, T_0, rho_0,
                                  n_harmonics, n_engines,
                                  frequency,
                                  method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment,
                                  flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression,
                                  spl_fan_expected):

    # Test fan source noise module using 55t NASA STCA Depart-Standard at source time = 67.96 s
    msap_fan = compute_fan_source_noise(dTt_fan_star, mdot_fan_star, N_fan_star, A_fan_star, d_fan_star,
                                        blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing,
                                        theta, M_0, c_0, T_0, rho_0,
                                        n_harmonics, n_engines,
                                        frequency,
                                        method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment, 
                                        flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression)
    spl_fan = compute_spl(msap_fan, rho_0, c_0)

    assert_quantity_almost_equal(spl_fan, spl_fan_expected, atol=3.0)

@pytest.mark.parametrize(
    "dTt_fan_star, mdot_fan_star, N_fan_star, A_fan_star, d_fan_star, \
     blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing, \
     theta, M_0, c_0, T_0, rho_0, \
     n_harmonics, n_engines, \
     frequency, \
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
def test_compute_fan_broadband_source_noise(dTt_fan_star, mdot_fan_star, N_fan_star, A_fan_star, d_fan_star,
                                  blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing,
                                  theta, M_0, c_0, T_0, rho_0,
                                  n_harmonics, n_engines,
                                  frequency,
                                  method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment,
                                  flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression,
                                  spl_fan_expected):

    # Test fan source noise module using 55t NASA STCA Depart-Standard at source time = 67.96 s
    msap_fan = compute_fan_source_noise(dTt_fan_star, mdot_fan_star, N_fan_star, A_fan_star, d_fan_star,
                                        blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing,
                                        theta, M_0, c_0, T_0, rho_0,
                                        n_harmonics, n_engines,
                                        frequency,
                                        method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment, 
                                        flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression)
    spl_fan = compute_spl(msap_fan, rho_0, c_0)

    assert_quantity_almost_equal(spl_fan, spl_fan_expected, atol=0.5)


# Core noise modules
@pytest.mark.parametrize(
    "mdot_combustor_inlet_star, Tt_combustor_inlet_star, Tt_combustor_outlet_star, Pt_combustor_inlet_star, dTt_combustor_design_star, theta, M_0, c_0, rho_0, frequency, n_engines, spl_core_expected",
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
def test_compute_core_source_noise(mdot_combustor_inlet_star, Tt_combustor_inlet_star, Tt_combustor_outlet_star, Pt_combustor_inlet_star, dTt_combustor_design_star, theta, M_0, c_0, rho_0, frequency, n_engines, spl_core_expected):

    # Test core source noise module using 55t NASA STCA Depart-Standard at source time = 67.96 s
    msap_core = compute_core_source_noise(mdot_combustor_inlet_star, Tt_combustor_inlet_star, Tt_combustor_outlet_star, Pt_combustor_inlet_star, dTt_combustor_design_star, theta, M_0, frequency, n_engines)
    spl_core = compute_spl(msap_core, rho_0, c_0)

    assert_quantity_almost_equal(spl_core, spl_core_expected, atol=5.2e-1)


# Jet noise modules
@pytest.mark.parametrize(
    "V_jet_star, rho_jet_star, A_jet_star, Tt_jet_star, theta, delta_jet, M_0, c_0, rho_0, frequency, n_engines, spl_jet_mixing_expected",
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
def test_compute_jet_mixing_source_noise(V_jet_star, rho_jet_star, A_jet_star, Tt_jet_star, theta, delta_jet, M_0, c_0, rho_0, frequency, n_engines, spl_jet_mixing_expected):

    # Test jet mixing source noise module using 55t NASA STCA Depart-Standard at source time = 67.96 s
    msap_jet_mixing = compute_jet_mixing_source_noise(V_jet_star, rho_jet_star, A_jet_star, Tt_jet_star, theta, delta_jet, M_0, c_0, frequency, n_engines)
    spl_jet_mixing = compute_spl(msap_jet_mixing, rho_0, c_0)

    assert_quantity_almost_equal(spl_jet_mixing, spl_jet_mixing_expected, atol=1e-1)

@pytest.mark.parametrize(
    "V_jet_star, M_j_star, A_jet_star, Tt_jet_star, theta, delta_jet, M_0, c_0, rho_0, frequency, n_engines, n_shock, msap_jet_shock_expected",
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
def test_compute_jet_shock_source_noise(V_jet_star, M_j_star, A_jet_star, Tt_jet_star, theta, delta_jet, M_0, c_0, rho_0, frequency, n_engines, n_shock, msap_jet_shock_expected):

    msap_jet_shock = compute_jet_shock_source_noise(V_jet_star, M_j_star, A_jet_star, Tt_jet_star, theta, delta_jet, M_0, c_0, frequency, n_engines, n_shock)
    spl_jet_shock = compute_spl(msap_jet_shock, rho_0, c_0)

    assert_quantity_almost_equal(spl_jet_shock, msap_jet_shock_expected)


# Airframe noise modules
@pytest.mark.parametrize(
    "noise_component_lst, \
    wing_span, wing_area, \
    horizontal_tail_span, horizontal_tail_area, \
    vertical_tail_span, vertical_tail_area, \
    theta_flaps, flaps_span, flaps_area, flaps_slot_number, \
    i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length, \
    M_0, c_0, rho_0, mu_0, theta, phi, frequency, \
    flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression, \
    spl_airframe_expected",
    [
        (
            ["landing_gear"], # noise_component_lst
            20.51304, # wing_span
            0, # wing_area
            0, # horizontal_tail_span
            0, # horizontal_tail_area
            0, # vertical_tail_span
            0, # vertical_tail_area
            0, # theta_flaps
            0, # flaps_span
            0, # flaps_area
            0, # flaps_slot_number
            True, # i_landing_gear
            2, # main_gear_number
            0.9144, # main_gear_tire_diameter
            2, # main_gear_wheel_number 
            2.286, # main_gear_length
            1, # nose_gear_number
            0.82296, # nose_gear_tire_diameter
            2, # nose_gear_wheel_number
            1.8288, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            90, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [115.25558766, 117.12999673, 118.93426941, 120.63204293, 122.17177455, 123.48474872, 124.48750492, 125.09216036, 125.22544579, 124.85053426, 123.97990113, 122.67073878, 121.00613314, 119.07350854, 116.9497252, 114.69501018, 112.35310628, 109.95423419, 107.51860247, 105.05944278, 102.58530238, 100.10165518, 97.61198769, 95.11851288]
        ),
        (
            ["landing_gear"], # noise_component_lst
            20.51304, # wing_span
            0, # wing_area
            0, # horizontal_tail_span
            0, # horizontal_tail_area
            0, # vertical_tail_span
            0, # vertical_tail_area
            0, # theta_flaps
            0, # flaps_span
            0, # flaps_area
            0, # flaps_slot_number
            True, # i_landing_gear
            2, # main_gear_number
            0.9144, # main_gear_tire_diameter
            2, # main_gear_wheel_number 
            2.286, # main_gear_length
            1, # nose_gear_number
            0.82296, # nose_gear_tire_diameter
            2, # nose_gear_wheel_number
            1.8288, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            10, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [102.51632681, 104.44275325, 106.32729581, 108.14711904, 109.86833667, 111.44248715, 112.80408018, 113.8719168,  114.55772377, 114.78376196, 114.50493456, 113.72424086, 112.49161315, 110.88700138, 108.99842546, 106.90544067, 104.67152568, 102.34334026,  99.95337461,  97.523466, 95.06795956,  92.59614126,  90.11396606,  87.62523008]
        ),
        (
            ["landing_gear"], # noise_component_lst
            20.51304, # wing_span
            0, # wing_area
            0, # horizontal_tail_span
            0, # horizontal_tail_area
            0, # vertical_tail_span
            0, # vertical_tail_area
            0, # theta_flaps
            0, # flaps_span
            0, # flaps_area
            0, # flaps_slot_number
            True, # i_landing_gear
            2, # main_gear_number,
            0.9144, # main_gear_tire_diameter
            2, # main_gear_wheel_number 
            2.286, # main_gear_length
            1, # nose_gear_number
            0.82296, # nose_gear_tire_diameter
            2, # nose_gear_wheel_number
            1.8288, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            170, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [ 98.07027439,  99.88033397, 101.58682073, 103.13932033, 104.47028142, 105.497026,   106.13145268, 106.2985756,  105.95850499, 105.12028998, 103.83856566, 102.19537221, 100.27841485,  98.16555859,  95.91821044, 93.58115963,  91.18543588,  88.75182622,  86.29395686,  83.82063647,  81.33750901,  78.84817038,  76.35490345,  73.85915421]
        ),
    ]
)
def test_compute_landing_gear_noise(noise_component_lst,
                                    wing_span, wing_area,
                                    horizontal_tail_span, horizontal_tail_area,
                                    vertical_tail_span, vertical_tail_area,
                                    theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                    i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                    M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                    flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression,
                                    spl_airframe_expected):

    # Test airframe source noise module using 55t NASA STCA Depart-Standard at source time = 24.87 s
    msap_airframe = compute_airframe_source_noise(noise_component_lst,
                                                  wing_span, wing_area,
                                                  horizontal_tail_span, horizontal_tail_area,
                                                  vertical_tail_span, vertical_tail_area,
                                                  theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                                  i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                                  M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                                  flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression)
    spl_airframe = compute_spl(msap_airframe, rho_0, c_0)

    print(spl_airframe)

    assert_quantity_almost_equal(spl_airframe, spl_airframe_expected, atol=0.1)

@pytest.mark.parametrize(
    "noise_component_lst, \
    wing_span, wing_area, \
    horizontal_tail_span, horizontal_tail_area, \
    vertical_tail_span, vertical_tail_area, \
    theta_flaps, flaps_span, flaps_area, flaps_slot_number, \
    i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length, \
    M_0, c_0, rho_0, mu_0, theta, phi, frequency, \
    flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression, \
    spl_airframe_expected",
    [
        (
            ["leading_edge_slats"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            90, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [115.9, 117.2, 118.1, 118.6, 118.7, 118.7, 118.8, 118.9, 119.1, 119.1, 118.9, 118.4, 117.5, 116.4, 115.1, 113.5, 111.9, 110.3, 108.4, 106.5, 104.7, 102.7, 100.7,  98.8]
        ),
        (
            ["leading_edge_slats"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            10, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [121.4, 123.3, 124.8, 125.7, 126.2, 126.4, 126.4, 126.5, 126.6, 126.8, 126.9, 126.7, 126.2, 125.4, 124.4, 123.0, 121.6, 120.0, 118.3, 116.4, 114.6, 112.7, 110.7, 108.9]
        ),
        (
            ["leading_edge_slats"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            170, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [95.2, 96.1, 96.6, 96.8, 96.8, 96.9, 97.0, 97.2, 97.2, 97.0, 96.5, 95.7, 94.6, 93.3, 91.8, 90.1, 88.4, 86.7, 84.8, 82.8, 81.0, 79.0, 77.0, 75.0]
        ),
    ]
)
def test_compute_leading_edge_slat_noise(noise_component_lst,
                                    wing_span, wing_area,
                                    horizontal_tail_span, horizontal_tail_area,
                                    vertical_tail_span, vertical_tail_area,
                                    theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                    i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                    M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                    flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression,
                                    spl_airframe_expected):
    
    # Test airframe source noise module using 55t NASA STCA Depart-Standard at source time = 24.87 s
    msap_airframe = compute_airframe_source_noise(noise_component_lst,
                                                  wing_span, wing_area,
                                                  horizontal_tail_span, horizontal_tail_area,
                                                  vertical_tail_span, vertical_tail_area,
                                                  theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                                  i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                                  M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                                  flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression)
    spl_airframe = compute_spl(msap_airframe, rho_0, c_0)

    assert_quantity_almost_equal(spl_airframe, spl_airframe_expected, atol=0.11)

@pytest.mark.parametrize(
    "noise_component_lst, \
    wing_span, wing_area, \
    horizontal_tail_span, horizontal_tail_area, \
    vertical_tail_span, vertical_tail_area, \
    theta_flaps, flaps_span, flaps_area, flaps_slot_number, \
    i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length, \
    M_0, c_0, rho_0, mu_0, theta, phi, frequency, \
    flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression, \
    spl_airframe_expected",
    [
        (
            ["trailing_edge_flaps"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            90, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [111.0, 112.0, 113.0, 113.5, 112.9, 112.3, 111.8, 111.3, 110.7, 110.1, 109.6, 109.1, 108.5, 107.1, 104.2, 100.9,  98.0,  95.1,  92.1,  89.0,  86.1,  83.1,  80.0,  77.1]
        ),
        (
            ["trailing_edge_flaps"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            10, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [105.4, 106.4, 107.4, 108.4, 109.1, 108.6, 108.0, 107.5, 106.9, 106.4, 105.8, 105.3, 104.7, 104.2, 103.3, 100.1,  97.2,  94.3,  91.2,  88.1,  85.2,  82.2,  79.1,  76.2]
        ),
        (
            ["trailing_edge_flaps"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            170, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [-216.75724839, -215.75724838, -215.21199616, -215.76199616, -216.31199616, -216.86199616, -217.41199616, -217.96199616, -218.51199616, -219.06199616, -219.61199616, -220.16199616, -221.4338336 , -224.4338336 , -227.4338336 , -230.4338336 , -233.4338336 , -236.4338336 , -239.4338336 , -242.4338336 , -245.4338336 , -248.4338336 , -251.4338336 , -254.4338336 ]
        )
    ]
)
def test_compute_trailing_edge_flap_noise(noise_component_lst,
                                          wing_span, wing_area,
                                          horizontal_tail_span, horizontal_tail_area,
                                          vertical_tail_span, vertical_tail_area,
                                          theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                          i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                          M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                          flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression,
                                          spl_airframe_expected):
    
    # Test airframe source noise module using 55t NASA STCA Depart-Standard at source time = 24.87 s
    msap_airframe = compute_airframe_source_noise(noise_component_lst,
                                                  wing_span, wing_area,
                                                  horizontal_tail_span, horizontal_tail_area,
                                                  vertical_tail_span, vertical_tail_area,
                                                  theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                                  i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                                  M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                                  flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression)
    spl_airframe = compute_spl(msap_airframe, rho_0, c_0)

    assert_quantity_almost_equal(spl_airframe, spl_airframe_expected, atol=0.2)

@pytest.mark.parametrize(
    "noise_component_lst, \
    wing_span, wing_area, \
    horizontal_tail_span, horizontal_tail_area, \
    vertical_tail_span, vertical_tail_area, \
    theta_flaps, flaps_span, flaps_area, flaps_slot_number, \
    i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length, \
    M_0, c_0, rho_0, mu_0, theta, phi, frequency, \
    flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression, \
    spl_airframe_expected",
    [
        (
            ["wing"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            90, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            True, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [106.0, 107.5, 108.5, 109.2, 109.4, 109.3, 109.0, 108.4, 107.6, 106.6, 105.6, 104.5, 103.2, 102.0, 100.7,  99.3,  98.0,  96.7,  95.3,  93.9,  92.6,  91.2,  89.7,  88.4]
        ),
        (
            ["wing"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            10, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            True, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [111.5, 113.4, 115.0, 116.1, 116.8, 117.1, 117.1, 116.8, 116.3, 115.5, 114.6, 113.5, 112.4, 111.2, 110.0, 108.6, 107.4, 106.1, 104.7, 103.3, 101.9, 100.6,  99.1,  97.8]
        ),
        (
            ["wing"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            170, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            True, # flag_delta_wing
            True, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [85.5, 86.5, 87.2, 87.5, 87.4, 87.1, 86.5, 85.8, 84.8, 83.7, 82.6, 81.4, 80.1, 78.9, 77.6, 76.2, 74.9, 73.5, 72.1, 70.7, 69.4, 68.0, 66.5, 65.2]
        )
    ]
)
def test_compute_trailing_edge_wing_noise(noise_component_lst,
                                          wing_span, wing_area,
                                          horizontal_tail_span, horizontal_tail_area,
                                          vertical_tail_span, vertical_tail_area,
                                          theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                          i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                          M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                          flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression,
                                          spl_airframe_expected):
    
    # Test airframe source noise module using 55t NASA STCA Depart-Standard at source time = 24.87 s
    msap_airframe = compute_airframe_source_noise(noise_component_lst,
                                                  wing_span, wing_area,
                                                  horizontal_tail_span, horizontal_tail_area,
                                                  vertical_tail_span, vertical_tail_area,
                                                  theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                                  i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                                  M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                                  flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression)
    spl_airframe = compute_spl(msap_airframe, rho_0, c_0)

    print(np.max(np.abs(spl_airframe-spl_airframe_expected)))

    assert_quantity_almost_equal(spl_airframe, spl_airframe_expected, atol=1.11)

@pytest.mark.parametrize(
    "noise_component_lst, \
    wing_span, wing_area, \
    horizontal_tail_span, horizontal_tail_area, \
    vertical_tail_span, vertical_tail_area, \
    theta_flaps, flaps_span, flaps_area, flaps_slot_number, \
    i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length, \
    M_0, c_0, rho_0, mu_0, theta, phi, frequency, \
    flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression, \
    spl_airframe_expected",
    [
        (
            ["horizontal_tail"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            90, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            False, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [102.3, 104.8, 106.9, 108.4, 109.5, 110.0, 110.1, 109.7, 108.9, 107.8, 106.6, 105.1, 103.4, 101.7, 100.0,  98.0,  96.1,  94.2,  92.3,  90.2,  88.3,  86.3,  84.3,  82.4]
        ),
        (
            ["horizontal_tail"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            10, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            False, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [106.7, 109.6, 112.2, 114.3, 115.9, 117.1, 117.7, 117.8, 117.5, 116.8, 115.8, 114.6, 113.0, 111.5, 109.8, 107.9, 106.1, 104.3, 102.3, 100.3,  98.4,  96.4,  94.4,  92.4]
        ),
        (
            ["horizontal_tail"], # noise_component_lst
            20.51304, # wing_span
            150.41, # wing_area
            5.6388, # horizontal_tail_span
            20.16, # horizontal_tail_area
            4.7244, # vertical_tail_span
            21.3677, # vertical_tail_area
            10., # theta_flaps    
            6.096, # flaps_span
            11.1484, # flaps_area
            1, # flaps_slot_number
            False, # i_landing_gear
            0, # main_gear_number
            0, # main_gear_tire_diameter
            0, # main_gear_wheel_number 
            0, # main_gear_length
            0, # nose_gear_number
            0, # nose_gear_tire_diameter
            0, # nose_gear_wheel_number
            0, # nose_gear_length
            0.243, # M_0
            346.16, # c_0
            1.18341, # rho_0
            1.83716554e-5, # mu_0
            170, # theta
            0., # phi
            np.array([50.11872336,     63.09573445,   79.43282347,   100.        ,
                      125.89254118,   158.48931925,  199.5262315 ,   251.18864315,
                      316.22776602,   398.10717055,  501.18723363,   630.95734448,
                      794.32823472,  1000.        , 1258.92541179,  1584.89319246,
                      1995.26231497, 2511.88643151, 3162.27766017,  3981.07170553,
                      5011.87233627, 6309.5734448 , 7943.28234724, 10000.        ]),
            False, # flag_delta_wing
            False, # flag_aerodynamically_clean_wing
            False, # flag_high_speed_research_suppression
            [82.7, 84.8, 86.5, 87.5, 88.1, 88.2, 87.8, 87.1, 86.1, 84.7, 83.3, 81.7, 79.9, 78.1, 76.3, 74.3, 72.4, 70.5, 68.5, 66.5, 64.6, 62.6, 60.5, 58.6]
        )
    ]
)
def test_compute_horizontal_tail_noise(noise_component_lst,
                                      wing_span, wing_area,
                                      horizontal_tail_span, horizontal_tail_area,
                                      vertical_tail_span, vertical_tail_area,
                                      theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                      i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                      M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                      flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression,
                                      spl_airframe_expected):
    
    # Test airframe source noise module using 55t NASA STCA Depart-Standard at source time = 24.87 s
    msap_airframe = compute_airframe_source_noise(noise_component_lst,
                                                  wing_span, wing_area,
                                                  horizontal_tail_span, horizontal_tail_area,
                                                  vertical_tail_span, vertical_tail_area,
                                                  theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                                  i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                                  M_0, c_0, rho_0, mu_0, theta, phi, frequency,
                                                  flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression)
    spl_airframe = compute_spl(msap_airframe, rho_0, c_0)

    assert_quantity_almost_equal(spl_airframe, spl_airframe_expected, atol=1.15)
