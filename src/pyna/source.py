import numpy as np
from pyna.noise_tables import (
    CoreNoiseTables,
    JetMixingNoiseTables,
    JetShockNoiseTables,
    FanNoiseTables
)

_R_SOURCE = 0.3048
_K_JET = 6.67e-5
_P_REF = 2e-5
_A_REF = 10.334 * ( 0.3048 ** 2 )
_RHO_SEALEVEL = 1.22514
_C_SEALEVEL = 340.29395

def compute_inlet_broadband_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_broadband, flag_inlet_distortions):
    """
    Compute the broadband component of the fan inlet mean-square acoustic pressure (msap).

    Parameters
    ----------
    temperature_term : float
    
    M_tip : float
    
    M_tip_rel_design : float
    
    rotor_stator_spacing : float
    
    theta : float 
    
    method_broadband : str
    
    flag_inlet_distortions : bool

    Returns
    -------
    float:
        
    """

    tables = FanNoiseTables()

    # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A):
    match method_broadband:
        case 'original':
            if M_tip_rel_design <= 1:
                if M_tip <= 0.9:
                    tipmach_term = 58.5
                else:
                    tipmach_term = 58.5 - 20 * np.log10(M_tip / 0.9)
            else:
                if M_tip <= 0.9:
                    tipmach_term = 58.5 + 20 * np.log10(M_tip_rel_design)
                else:
                    tipmach_term = 58.5 + 20 * np.log10(M_tip_rel_design) - 20 * np.log10(M_tip / 0.9)

        case 'alliedsignal':
            if M_tip_rel_design <= 1:
                if M_tip <= 0.9:
                    tipmach_term = 55.5
                else:
                    tipmach_term = 55.5 - 20 * np.log10(M_tip / 0.9)
            else:
                if M_tip <= 0.9:
                    tipmach_term = 55.5 + 20 * np.log10(M_tip_rel_design)
                else:
                    tipmach_term = 55.5 + 20 * np.log10(M_tip_rel_design) - 20 * np.log10(M_tip / 0.9)

        case 'geae':
            if M_tip_rel_design <= 1:
                if M_tip <= 0.9:
                    tipmach_term = 58.5
                else:
                    tipmach_term = 58.5 - 50 * np.log10(M_tip / 0.9)
            else:
                if M_tip <= 0.9:
                    tipmach_term = 58.5 + 20 * np.log10(M_tip_rel_design)
                else:
                    tipmach_term = 58.5 + 20 * np.log10(M_tip_rel_design) - 50 * np.log10(M_tip / 0.9)

        case 'kresja':
            if M_tip_rel_design <= 1:
                if M_tip < 0.72:
                    tipmach_term = 34 + 20 * np.log10(1. / 1.245)
                else:
                    tipmach_term = 34 - 43 * (M_tip - 0.72) + 20 * np.log10(1. / 1.245)
            else:
                if M_tip < 0.72:
                    tipmach_term = 34 + 20 * np.log10(M_tip_rel_design / 1.245)
                else:
                    tipmach_term = 34 - 43 * (M_tip - 0.72) + 20 * np.log10(M_tip_rel_design/ 1.245)

    # Rotor-stator correction term (F2 of Eqn 4, Figure 6B)
    match method_broadband:
        case 'original':
            if flag_inlet_distortions:
                if rotor_stator_spacing <= 100:
                    rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -5 * np.log10(100 / 300)
            else:
                rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)

        case 'alliedsignal':
            if flag_inlet_distortions:
                if rotor_stator_spacing <= 100:
                    rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -5 * np.log10(100. / 300)  
            else:
                rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)

        case 'geae':
            rotorstator_term = 0

        case 'kresja':
            if flag_inlet_distortions:
                if rotor_stator_spacing <= 100:
                    rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -5 * np.log10(100 / 300)
            else:
                rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)

    # Theta correction term (F3 of Eqn 4, Figure 7A):
    directivity = tables.get_inlet_broadband_directivity(theta, method_broadband)

    # Component value:
    return temperature_term + tipmach_term + rotorstator_term + directivity


def compute_discharge_broadband_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_broadband, flag_inlet_distortions, flag_inlet_guide_vanes):
    """
    
    Parameters
    ----------
    temperature_term : float

    M_tip : float

    M_tip_rel_design : float

    rotor_stator_spacing : float

    theta : float

    method_broadband : str

    flag_inlet_distortions : bool

    flag_inlet_guide_vanes : bool

    Returns
    -------
    float : 

    """

    tables = FanNoiseTables()

    # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B)
    match method_broadband:
        case 'original':
            if M_tip_rel_design <= 1:
                if M_tip <= 1:
                    tipmach_term = 60
                else:
                    tipmach_term = 60 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    tipmach_term = 60 + 20 * np.log10(M_tip_rel_design)
                else:
                    tipmach_term = 60 + 20 * np.log10(M_tip_rel_design) - 20 * np.log10(M_tip / 1)

        case 'alliedsignal':
            if M_tip_rel_design <= 1:
                if M_tip <= 1:
                    tipmach_term = 58
                else:
                    tipmach_term = 58 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    tipmach_term = 58 + 20 * np.log10(M_tip_rel_design)
                else:
                    tipmach_term = 58 + 20 * np.log10(M_tip_rel_design) - 20 * np.log10(M_tip / 1)

        case 'geae':
            if M_tip_rel_design <= 1:
                if M_tip <= 1:
                    tipmach_term = 63
                else:
                    tipmach_term = 63 - 30 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    tipmach_term = 63 + 20 * np.log10(M_tip_rel_design)
                else:
                    tipmach_term = 63 + 20 * np.log10(M_tip_rel_design) - 30 * np.log10(M_tip / 1)

        case 'kresja':
            if M_tip_rel_design <= 1:
                tipmach_term = 34 - 17 * (M_tip - 0.65) + 20 * np.log10(1 / 1.245)
            else:
                tipmach_term = 34 - 17 * (M_tip - 0.65) + 20 * np.log10(M_tip_rel_design / 1.245)

    # Rotor-stator correction term (F2 of Eqn 4, Figure 6B):
    match method_broadband:
        case 'original':
            if flag_inlet_distortions:
                if rotor_stator_spacing <= 100:
                    rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -5 * np.log10(100 / 300)
            else:
                rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)

        case 'alliedsignal':
            if flag_inlet_distortions:
                if rotor_stator_spacing <= 100:
                    rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -5 * np.log10(100 / 300)
            else:
                rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)

        case 'geae':
            rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)

        case 'kresja':
            # Rotor-stator spacing correction term (F2, of Eqn 10, Figure 6B):
            if flag_inlet_distortions:
                if rotor_stator_spacing <= 100:
                    rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -5 * np.log10(100 / 300)
            else:
                rotorstator_term = -5 * np.log10(rotor_stator_spacing / 300)

    # Theta correction term (F3 of Eqn 10, Figure 7B):
    directivity = tables.get_discharge_broadband_directivity(theta, method_broadband)

    # Added noise factor if there are inlet guide vanes present:
    if flag_inlet_guide_vanes:
        flag_inlet_guide_vanes_term = 3
    else:
        flag_inlet_guide_vanes_term = 0

    return temperature_term + tipmach_term + rotorstator_term + directivity + flag_inlet_guide_vanes_term


def compute_inlet_tone_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_rotor_stator_interactions, flag_inlet_distortions):
    """
    Compute the tone component of the fan inlet mean-square acoustic pressure (msap)

    Parameters
    ----------

    Returns
    -------
    """

    tables = FanNoiseTables()

    # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A):
    match method_rotor_stator_interactions:
        case 'original':
            if M_tip_rel_design <= 1:
                if M_tip <= 0.72:
                    tipmach_term = 60.5
                else:
                    F1TIA = 60.5 + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 59.5 + 80 * np.log10(1. / M_tip)
                    if F1TIA < F1TIB:
                        tipmach_term = F1TIA
                    else:
                        tipmach_term = F1TIB
            else:
                if M_tip <= 0.72:
                    tipmach_term = 60.5 + 20 * np.log10(M_tip_rel_design)
                else:
                    F1TIA = 60.5 + 20 * np.log10(M_tip_rel_design) + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 59.5 + 80 * np.log10(M_tip_rel_design / M_tip)
                    if F1TIA < F1TIB:
                        tipmach_term = F1TIA
                    else:
                        tipmach_term = F1TIB

        case 'alliedsignal':
            if M_tip_rel_design <= 1:
                if M_tip <= 0.72:
                    tipmach_term = 54.5
                else:
                    F1TIA = 54.5 + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 53.5 + 80 * np.log10(1. / M_tip)
                    if F1TIA < F1TIB:
                        tipmach_term = F1TIA
                    else:
                        tipmach_term = F1TIB
            else:
                if M_tip <= 0.72:
                    tipmach_term = 54.5 + 20 * np.log10(M_tip_rel_design)
                else:
                    F1TIA = 54.5 + 20 * np.log10(M_tip_rel_design) + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 53.5 + 80 * np.log10(M_tip_rel_design / M_tip)
                    if F1TIA < F1TIB:
                        tipmach_term = F1TIA
                    else:
                        tipmach_term = F1TIB

        case 'geae':
            if M_tip_rel_design <= 1:
                if M_tip <= 0.72:
                    tipmach_term = 60.5
                else:
                    F1TIA = 60.5 + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 64.5 + 80 * np.log10(1. / M_tip)
                    if F1TIA < F1TIB:
                        tipmach_term = F1TIA
                    else:
                        tipmach_term = F1TIB
            else:
                if M_tip <= 0.72:
                    tipmach_term = 60.5 + 20 * np.log10(M_tip_rel_design)
                else:
                    F1TIA = 60.5 + 20 * np.log10(M_tip_rel_design) + 50 * np.log10(M_tip / 0.72)
                    F1TIB = 64.5 + 80 * np.log10(M_tip_rel_design) - 80 * np.log10(M_tip)
                    if F1TIA < F1TIB:
                        tipmach_term = F1TIA
                    else:
                        tipmach_term = F1TIB

        case 'kresja':
            if M_tip_rel_design <= 1:
                tipmach_term = 42 - 20 * M_tip + 20 * np.log10(1. / 1.245)
            else:
                tipmach_term = 42 - 20 * M_tip + 20 * np.log10(M_tip_rel_design / 1.245)

            
    # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12):
    match method_rotor_stator_interactions:
        case 'original':
            if flag_inlet_distortions:
                if rotor_stator_spacing < 100:
                    rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -10 * np.log10(100 / 300)
            else:
                rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)

        case 'alliedsignal':
            if flag_inlet_distortions:
                if rotor_stator_spacing < 100:
                    rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -10 * np.log10(100./ 300)
            else:
                rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300) 

        case 'geae':
            rotorstator_term = 0.
        
        case 'kresja':
            if flag_inlet_distortions:
                if rotor_stator_spacing < 100:
                    rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -10 * np.log10(100 / 300)
            else:
                rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)

    # Theta correction term (F3 of Eqn 6, Figure 13A):
    directivity = tables.get_inlet_tones_directivity(theta, method_rotor_stator_interactions)

    # Component value:
    tonlv_I = temperature_term + tipmach_term + rotorstator_term + directivity

    return tonlv_I


def compute_discharge_tone_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_rotor_stator_interactions, flag_inlet_distortions, flag_inlet_guide_vanes):
    """
    Compute the tone component of the fan discharge mean-square acoustic pressure (msap)

    Parameters
    ----------
    temperature_term :float

    M_tip : float

    M_tip_rel_design : float

    theta : float

    rotor_stator_spacing : float

    theta : float

    method_rotor_stator_interactions : str

    flag_inlet_distortions : bool

    flag_inlet_guide_vanes : bool

    Returns
    -------
    float : 

    """

    tables = FanNoiseTables()

    # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B):
    match method_rotor_stator_interactions:
        case 'original':
            if M_tip_rel_design <= 1:
                if M_tip <= 1:
                    tipmach_term = 63
                else:
                    tipmach_term = 63 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    tipmach_term = 63 + 20 * np.log10(M_tip_rel_design)
                else:
                    tipmach_term = 63 + 20 * np.log10(M_tip_rel_design) - 20 * np.log10(M_tip / 1)

        case 'alliedsignal':
            # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by AlliedSignal):
            if M_tip_rel_design <= 1:
                if M_tip <= 1:
                    tipmach_term = 59
                else:
                    tipmach_term = 59 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    tipmach_term = 59 + 20 * np.log10(M_tip_rel_design)
                else:
                    tipmach_term = 59 + 20 * np.log10(M_tip_rel_design) - 20 * np.log10(M_tip / 1)

        case 'geae':
            # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by GE):
            if M_tip_rel_design <= 1:
                if M_tip <= 1:
                    tipmach_term = 63
                else:
                    tipmach_term = 63 - 20 * np.log10(M_tip / 1)
            else:
                if M_tip <= 1:
                    tipmach_term = 63 + 20 * np.log10(M_tip_rel_design)
                else:
                    tipmach_term = 63 + 20 * np.log10(M_tip_rel_design) - 20 * np.log10(M_tip / 1)

        case 'kresja':
            # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by Krejsa):
            if M_tip_rel_design <= 1:
                tipmach_term = 46 - 20 * M_tip + 20 * np.log10(1 / 1.245)
            else:
                tipmach_term = 46 - 20 * M_tip + 20 * np.log10(M_tip_rel_design / 1.245)
        
    # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12):
    match method_rotor_stator_interactions:
        case "original":
            if flag_inlet_distortions:
                if rotor_stator_spacing < 100:
                    rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -10 * np.log10(100 / 300)
            else:
                rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)
        
        case "alliedsignal":
            if flag_inlet_distortions:
                if rotor_stator_spacing < 100:
                    rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -10 * np.log10(100 / 300)
            else:
                rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)

        case "geae":
            rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)

        case "kresja":
            if flag_inlet_distortions:
                if rotor_stator_spacing < 100:
                    rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)
                else:
                    rotorstator_term = -10 * np.log10(100 / 300)
            else:
                rotorstator_term = -10 * np.log10(rotor_stator_spacing / 300)

    # Theta correction term (F3 of Eqn 6, Figure 13B):
    directivity = tables.get_discharge_tones_directivity(theta, method_rotor_stator_interactions)

    # Added noise factor if there are inlet guide vanes:
    if flag_inlet_guide_vanes:
        igv_term = 6
    else:
        igv_term = 0

    return temperature_term + tipmach_term + rotorstator_term + directivity + igv_term


def calculate_inlet_harmonics(inlet_tones, M_tip, M_tip_tangential, blade_number, vane_number, n_harmonics, theta, bpf, f, method_rotor_stator_interactions, flight_segment, flag_inlet_distortions, flag_inlet_guide_vanes):
    """
    Compute fan tone harmonics for inlet (dp) and discharge (dpx).

    Parameters
    ----------

    Returns
    -------


    """

    tables = FanNoiseTables()

    # Assign discrete interaction tones at bpf and harmonics to proper bins (see figures 8 and 9):
    # Initialize solution matrices
    dp = np.zeros(f.size)
    
    i_cutoff = get_cutoff(M_tip_tangential, blade_number, vane_number)

    for i_harmonic in np.arange(1, n_harmonics + 1):

        # Determine the tone fall-off rates per harmonic
        match method_rotor_stator_interactions:
            case 'original':
                if flag_inlet_guide_vanes:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                inlet_harmonic = 0
                            else:
                                inlet_harmonic = 3 * (i_harmonic + 1)
                        case 1:
                            if i_harmonic == 1:
                                inlet_harmonic = 8
                            else:
                                inlet_harmonic = 3 * (i_harmonic + 1)
                else:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                inlet_harmonic = 0
                            else:
                                inlet_harmonic = 3 * (i_harmonic - 1)
                        case 1:
                            if i_harmonic == 1:
                                inlet_harmonic = 8
                            else:
                                inlet_harmonic = 3 * (i_harmonic - 1)
                        
            case 'alliedsignal':
                if flag_inlet_guide_vanes:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                inlet_harmonic = 0
                            else:
                                inlet_harmonic = 3 * (i_harmonic + 1)
                        case 1:
                            if i_harmonic == 1:
                                inlet_harmonic = 8
                            else:
                                inlet_harmonic = 3 * (i_harmonic + 1)

                else:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                inlet_harmonic = 0
                            elif i_harmonic == 2:
                                inlet_harmonic = 9.2
                            else:
                                inlet_harmonic = 3 * i_harmonic + 1.8
                        case 1:
                            if i_harmonic == 1:
                                inlet_harmonic = 8
                            elif i_harmonic == 2:
                                inlet_harmonic = 9.2
                            else:
                                inlet_harmonic = 3 * i_harmonic + 1.8

            case 'geae':
                if flag_inlet_guide_vanes:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                inlet_harmonic = 0
                            else:
                                inlet_harmonic = 3 * (i_harmonic + 1)
                        case 1:
                            if i_harmonic == 1:
                                inlet_harmonic = 8
                            else:
                                inlet_harmonic = 3 * (i_harmonic + 1)
                
                else:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                inlet_harmonic = 0
                            else:
                                if M_tip < 1.15:
                                    inlet_harmonic = 6 * (i_harmonic - 1)
                                else:
                                    inlet_harmonic = 9 * (i_harmonic - 1)
                        case 1:
                            if i_harmonic == 1:
                                inlet_harmonic = 8
                            else:
                                if M_tip < 1.15:
                                    inlet_harmonic = 6 * (i_harmonic - 1)
                                else:
                                    inlet_harmonic = 9 * (i_harmonic - 1)

            case 'kresja':
                if flag_inlet_guide_vanes:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                inlet_harmonic = 0
                            else:
                                inlet_harmonic = 3 * (i_harmonic + 1)
                        case 1:
                            if i_harmonic == 1:
                                inlet_harmonic = 8
                            else:
                                inlet_harmonic = 3 * (i_harmonic + 1)

                else:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                inlet_harmonic = 0
                            else:
                                inlet_harmonic = 3 * (i_harmonic - 1)
                        case 1:
                            if i_harmonic == 1:
                                inlet_harmonic = 8
                            else:
                                inlet_harmonic = 3 * (i_harmonic - 1)

        # Calculate distortions
        if flag_inlet_distortions:
            turbulent_control_structures = tables.get_cleanup_turbulent_control_structures(method_rotor_stator_interactions, flight_segment, i_harmonic, theta)
            distortions = 10 ** (0.1 * (inlet_tones - turbulent_control_structures) - i_harmonic + 1)
        else:
            turbulent_control_structures = 0
            distortions = 0.

        # Calculate tone power
        tonpwr_i = 10 ** (0.1 * (inlet_tones - inlet_harmonic - turbulent_control_structures)) + distortions
        
        # Cycle through frequencies and assign tones to 1/3rd octave bins:
        f_1, f_2, f_3, f_4 = tables.get_filter_constants(filter_bandwidth=1.)

        nfi = 1
        for l in np.arange(nfi - 1, f.size):
            f_ratio = bpf * i_harmonic / f[l]
            FR = 1
            if f_ratio < f_1:
                break
            elif f_ratio > f_4:
                ll = l
                continue
            elif f_ratio > f_3:
                FR = (f_4 - f_ratio) / (f_4 - f_3)
            elif f_ratio < f_2:
                FR = (f_ratio - f_1) / (f_2 - f_1)
            dp[l] = dp[l] + tonpwr_i * FR
            nfi = ll
            continue

    return dp


def calculate_discharge_harmonics(discharge_tones, M_tip, M_tip_tangential, blade_number, vane_number, n_harmonics, bpf, f, method_rotor_stator_interactions, flag_inlet_guide_vanes):
    """
    Compute fan tone harmonics for inlet (dp) and discharge (dpx).

    Parameters
    ----------

    Returns
    -------


    """

    tables = FanNoiseTables()

    # Assign discrete interaction tones at bpf and harmonics to proper bins (see figures 8 and 9):
    # Initialize solution matrices
    dpx = np.zeros(f.size)

    i_cutoff = get_cutoff(M_tip_tangential, blade_number, vane_number)

    for i_harmonic in np.arange(1, n_harmonics + 1):

        # Determine the tone fall-off rates per harmonic
        match method_rotor_stator_interactions:
            case 'original':
                if flag_inlet_guide_vanes:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                discharge_harmonic = 0
                            else:
                                discharge_harmonic = 3 * (i_harmonic + 1)
                        case 1:
                            if i_harmonic == 1:
                                discharge_harmonic = 8
                            else:
                                discharge_harmonic = 3 * (i_harmonic + 1)
                else:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                discharge_harmonic = 0
                            else:
                                discharge_harmonic = 3 * (i_harmonic - 1)
                        case 1:
                            if i_harmonic == 1:
                                discharge_harmonic = 8
                            else:
                                discharge_harmonic = 3 * (i_harmonic - 1)
                        
            case 'alliedsignal':
                if flag_inlet_guide_vanes:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                discharge_harmonic = 0
                            else:
                                discharge_harmonic = 3 * (i_harmonic + 1)
                        case 1:
                            if i_harmonic == 1:
                                discharge_harmonic = 8
                            else:
                                discharge_harmonic = 3 * (i_harmonic + 1)

                else:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                discharge_harmonic = 0
                            elif i_harmonic == 2:
                                discharge_harmonic = 9.2
                            else:
                                discharge_harmonic = 3 * i_harmonic + 1.8
                        case 1:
                            if i_harmonic == 1:
                                discharge_harmonic = 8
                            elif i_harmonic == 2:
                                discharge_harmonic = 9.2
                            else:
                                discharge_harmonic = 3 * i_harmonic + 1.8

            case 'geae':
                if flag_inlet_guide_vanes:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                discharge_harmonic = 0
                            else:
                                discharge_harmonic = 3 * (i_harmonic + 1)
                        case 1:
                            if i_harmonic == 1:
                                discharge_harmonic = 8
                            else:
                                discharge_harmonic = 3 * (i_harmonic + 1)
                
                else:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                discharge_harmonic = 0
                            else:
                                if M_tip < 1.15:
                                    discharge_harmonic = 3 * (i_harmonic - 1)
                                else:
                                    discharge_harmonic = 3 * (i_harmonic - 1)
                        case 1:
                            if i_harmonic == 1:
                                discharge_harmonic = 8
                            else:
                                if M_tip < 1.15:
                                    discharge_harmonic = 3 * (i_harmonic - 1)
                                else:
                                    discharge_harmonic = 3 * (i_harmonic - 1)

            case 'kresja':
                if flag_inlet_guide_vanes:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                discharge_harmonic = 0
                            else:
                                discharge_harmonic = 3 * (i_harmonic + 1)
                        case 1:
                            if i_harmonic == 1:
                                discharge_harmonic = 8
                            else:
                                discharge_harmonic = 3 * (i_harmonic + 1)

                else:
                    match i_cutoff:
                        case 0:
                            if i_harmonic == 1:
                                discharge_harmonic = 0
                            else:
                                discharge_harmonic = 3 * (i_harmonic - 1)
                        case 1:
                            if i_harmonic == 1:
                                discharge_harmonic = 8
                            else:
                                discharge_harmonic = 3 * (i_harmonic - 1)

        # Calculate tone power
        tonpwr_x = 10 ** (0.1 * (discharge_tones - discharge_harmonic))

       # Cycle through frequencies and assign tones to 1/3rd octave bins:
        f_1, f_2, f_3, f_4 = tables.get_filter_constants(filter_bandwidth=1.)

        nfi = 1
        for l in np.arange(nfi - 1, f.size):
            f_ratio = bpf * i_harmonic / f[l]
            FR = 1
            if f_ratio < f_1:
                break
            elif f_ratio > f_4:
                ll = l
                continue
            elif f_ratio > f_3:
                FR = (f_4 - f_ratio) / (f_4 - f_3)
            elif f_ratio < f_2:
                FR = (f_ratio - f_1) / (f_2 - f_1)
            dpx[l] = dpx[l] + tonpwr_x * FR
            nfi = ll
            continue

    return dpx


def compute_combination_tone_level(temperature_term, M_tip, theta, bpf, f, method_rotor_stator_interactions, flag_inlet_guide_vanes):
    """
    Compute the combination tone component of the fan mean-square acoustic pressure (msap).

    Combination tone (multiple pure tone or buzzsaw) calculations.
    Note the original Heidmann reference states that MPTs should be computed if
    the tangential tip speed is supersonic, but the ANOPP implementation states MPTs
    should be computed if the relative tip speed is supersonic.  The ANOPP implementation
    is used here, i.e., if M_tip >= 1.0, MPTs are computed.

    Parameters
    ----------
    temperature_term : float

    M_tip : float

    theta : float

    bpf : float

    f : np.ndarray

    method_rotor_stator_interactions : str

    flag_inlet_guide_vanes : bool

    Returns
    -------
    np.ndarray : 

    """

    tables = FanNoiseTables()

    # Initialize solution matrices
    dcp = np.zeros(f.size)

    if M_tip >= 1:

        directivity = tables.get_combination_tones_directivity(theta, method_rotor_stator_interactions)

        # Noise adjustment (reduction) if there are inlet guide vanes, for all methods:
        if flag_inlet_guide_vanes:
            igv_term = -5
        else:
            igv_term = 0

        # Loop through the three sub-bpf terms (k = 1; 1/2 bpf term, k = 2; 1/4 bpf term, k = 3; 1/8 bpf term)
        for k in np.arange(1, 4):
            
            # Tip Mach-dependent term (F1 of Eqn 8 in Heidmann report, Figure 15A)
            tipmach_term = tables.get_combination_tones_tipmach(M_tip, k, method_rotor_stator_interactions)
            
            # Cycle through frequencies and make assignments:
            for j in np.arange(f.size):
                # Frequency-dependent term (F3 of Eqn 9, Figure 14):

                spectral_term = tables.get_combination_tones_spectral_distribution(f[j] / bpf, k, method_rotor_stator_interactions)

                # Be sure to add the three sub-bpf components together at each frequency:
                dcp[j] = dcp[j] + 10 ** (0.1 * (temperature_term + tipmach_term + directivity + igv_term + spectral_term))

    return dcp


def get_cutoff(M_tip_tangential, blade_number, vane_number):
    """
    Compute if the fan is in cut-off condition (0/1). If the cutoff parameter is less than 1.05 and the tip Mach is less than unity, the fan is cut off
    and fan noise does not propagate (i.e., the tones are reduced in magnitude).

    Parameters
    ----------
    M_tip_tangential : float

    blade_number : int

    vane_number : int

    Returns
    -------
    int : 

    """

    # Vane/blade ratio parameter:
    vane_blade_ratio = 1 - vane_number / blade_number
    
    # Fundamental tone cutoff parameter
    # Source: Zorumski report 1982 part 2. Chapter 8.1 Eq. 8
    delta_cutoff = abs(M_tip_tangential / vane_blade_ratio)
    
    if delta_cutoff < 1.05 and M_tip_tangential < 1:
        return 1
    else:
        return 0


def compute_fan_inlet_spectral_distribution(bpf, f, method_broadband):
    
    # Spectral distribution
    match method_broadband:
        case 'alliedsignal':
            spectral_distribution = 2.445096095 * (np.log(f / bpf / 2)) ** 2
            spectral_distribution[f/bpf > 2] = (13.97197769 * (np.log(f / bpf / 2)) ** 2)[f/bpf > 2]

        case 'kresja':
            spectral_distribution = 3.4929944 * (np.log(f / bpf / 4)) ** 2

        case _:
            spectral_distribution = 3.4929944 * (np.log(f / bpf / 2.5)) ** 2

    return spectral_distribution


def compute_fan_discharge_spectral_distribution(bpf, f, method_broadband):
    
    # Spectral distribution
    match method_broadband:
        case 'alliedsignal':
            spectral_distribution[f/bpf > 2] = (13.97197769 * (np.log(f / bpf / 2)) ** 2)[f/bpf > 2]
            
        case 'kresja':
            spectral_distribution = 3.4929944 * (np.log(f / bpf / 2.5)) ** 2

        case _:
            spectral_distribution = 3.4929944 * (np.log(f / bpf / 2.5)) ** 2

    return spectral_distribution


def compute_fan_source_noise(dTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star, 
                             blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing, 
                             theta, M_0, c_0, T_0, rho_0, 
                             n_harmonics, n_engines, 
                             f,
                             method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment,
                             flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression):
    """
    Calculates fan noise mean-square acoustic pressure (msap) using Berton's implementation of the fan noise method.

    Parameters
    ----------
    dTt_f_star : float

    mdot_f_star : float

    N_f_star : float

    A_f_star : float

    d_f_star : float

    blade_number : int
     
    vane_number : int
    
    M_tip_rel_design : float
    
    rotor_stator_spacing : float

    theta : float

    M_0 : float

    c_0 : float

    T_0 : float

    rho_0 : float

    n_harmonics : int

    n_engines : int

    f : np.ndarray

    method_broadband : str
    
    method_rotor_stator_interactions : str
    
    noise_direction : str

    flight_segment : str
    
    flag_broadband : bool

    flag_tones : bool

    flag_combination_tones : bool
    
    flag_inlet_distortions : bool
    
    flag_inlet_guide_vanes : bool
    
    flag_liner_suppression : bool
    
    Returns
    -------
    np.ndarray : 

    """

    tables = FanNoiseTables()

    ### Dimensionalize the inputs
    temperature_rise = dTt_f_star * T_0  # [K]
    rpm = N_f_star * 60 * c_0 / (d_f_star * np.sqrt(_A_REF))  # [rpm]
    M_tip_tangential = (d_f_star * np.sqrt(_A_REF) / 2) * rpm * 2 * np.pi / 60 / c_0
    mdot_fan = mdot_f_star * rho_0 * c_0 * _A_REF  # [kg/s]
    bpf = rpm * blade_number / 60. / (1 - M_0 * np.cos(theta * np.pi / 180))  # [Hz]
    M_flow = mdot_fan / (rho_0 * A_f_star * _A_REF * c_0)
    M_tip = (M_tip_tangential ** 2 + M_flow ** 2) ** 0.5

    # Temperature-flow power base term:
    match method_broadband:
        case 'kresja':
            temperature_term = 10 * np.log10((temperature_rise * 1.8) ** 4 * 2.20462 * mdot_fan / (1 - M_0 * np.cos(theta * np.pi / 180)) ** 4 / _R_SOURCE ** 2 / (_RHO_SEALEVEL ** 2 * _C_SEALEVEL ** 4))
        case _:
            temperature_term = 10 * np.log10((temperature_rise * 1.8) ** 2 * 2.20462 * mdot_fan / (1 - M_0 * np.cos(theta * np.pi / 180)) ** 4 / _R_SOURCE ** 2 / (_RHO_SEALEVEL ** 2 * _C_SEALEVEL ** 4))

    # Calculate individual noise components
    msap = 0.
    match noise_direction:
        case "inlet":
            if flag_broadband:
                broadband_level = compute_inlet_broadband_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_broadband, flag_inlet_distortions)
                spectral_distribution = compute_fan_inlet_spectral_distribution(bpf, f, method_broadband)
                msap += 10 ** (0.1 * (broadband_level - spectral_distribution))

            if flag_tones:
                tone_level = compute_inlet_tone_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_rotor_stator_interactions, flag_inlet_distortions)
                harmonics = calculate_inlet_harmonics(tone_level, M_tip, M_tip_tangential, blade_number, vane_number, n_harmonics, theta, bpf, f, method_rotor_stator_interactions, flight_segment, flag_inlet_distortions, flag_inlet_guide_vanes)
                msap += harmonics

        case "discharge":
            if flag_broadband:
                broadband_level = compute_discharge_broadband_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_broadband, flag_inlet_distortions, flag_inlet_guide_vanes)
                spectral_distribution = compute_fan_discharge_spectral_distribution(bpf, f, method_broadband)
                msap += 10 ** (0.1 * (broadband_level - spectral_distribution))
            
            if flag_tones:
                tone_level = compute_discharge_tone_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_rotor_stator_interactions, flag_inlet_distortions, flag_inlet_guide_vanes)
                harmonics = calculate_discharge_harmonics(tone_level, M_tip, M_tip_tangential, blade_number, vane_number, n_harmonics, bpf, f, method_rotor_stator_interactions, flag_inlet_guide_vanes)
                msap += harmonics
            
    # Compute combination tones
    if flag_combination_tones and M_tip > 0:
        combination_tone_level = compute_combination_tone_level(temperature_term, M_tip, theta, bpf, f, method_rotor_stator_interactions, flag_inlet_guide_vanes)
        msap += combination_tone_level
            
    # Fan liner suppression
    if flag_liner_suppression:
        msap *= tables.get_liner_suppression(f, theta, noise_direction)

    # Multiply for number of engines
    return msap * n_engines

    # TODO: implement shielding
    # Fan inlet shielding
    # if settings['shielding'] and component == 'fan_inlet':
    #     msap_j = msap_j / (10 ** (shield[i, :] / 10.))
    # msap_fan[i, :] = msap_j
    # return msap_fan


def compute_core_source_noise(mdot_i_c_star, Tt_i_c_star, Tt_j_c_star, Pt_i_c_star, DTt_design_c_star, theta, M_0, f, n_engines):
    """
	Compute core noise mean-square acoustic pressure (msap).

	Parameters
	----------
	mdot_i_c_star : float
    
    Tt_i_c_star : float
    
    Tt_j_c_star : float
    
    Pt_i_c_star : float
    
    DTt_design_c_star : float
    
    theta : float
    
    M_0 : float
    
    c_0 : float
    
    f : np.ndarray
    
    n_engines : int
	

	Returns
    -------
    np.ndarray
        Mean-square acoustic pressure of core source noise

	"""
    
    tables = CoreNoiseTables()
    
    r_s_star = _R_SOURCE / np.sqrt(_A_REF)    
    A_c_star = 1.   # Noise method is indepenent of A_c_star; only used in non-dimensionalization

    # Turbine transmission loss function
    # Source: Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    # if settings['core_turbine_attenuation_method'] == 'ge':
    g_TT = DTt_design_c_star ** (-4)
    # Source: Hultgren, 2012: A comparison of combustor models Equation 6
    # elif settings['core_turbine_attenuation_method'] == 'pw':
    #     zeta = (rho_te_c_star[i] * c_te_c_star[i]) / (rho_ti_c_star[i] * c_ti_c_star[i])
    #     g_TT = 0.8 * zeta / (1 + zeta) ** 2
    # else:
    #     raise ValueError('Invalid method to account for turbine attenuation effects of combustor noise. Specify GE/PW.')

    # Calculate acoustic power (Pi_star)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    Pi_star = 8.85e-7 * (mdot_i_c_star / A_c_star) * ((Tt_j_c_star - Tt_i_c_star) / Tt_i_c_star) ** 2 * Pt_i_c_star ** 2 * g_TT

    # Calculate directivity function (D)
    # Take the D function as SAE ARP876E Table 18 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table II
    directivity = 10 ** tables.get_directivity(theta)

    # Calculate the spectral function (S)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
    f_p = 400. / (1 - M_0 * np.cos(theta * np.pi / 180.))

    # Take the S function as SAE ARP876E Table 17 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table III
    spectral_distribution = 10 ** tables.get_spectral_distribution(np.log10(f/f_p))

    # Calculate mean-square acoustic pressure (msap)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
    msap = Pi_star * A_c_star / (4 * np.pi * r_s_star ** 2) * directivity * spectral_distribution / (1. - M_0 * np.cos(np.pi / 180. * theta)) ** 4

    # Multiply with number of engines
    msap = msap * n_engines

    # Normalize msap by reference pressure
    return msap/_P_REF**2


def compute_jet_mixing_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, delta, M_0, c_0, f, n_engines):
    """
    Compute jet mixing noise mean-square acoustic pressure (msap).

    Parameters
    ----------
    V_j_star : float

    rho_j_star : float

    A_j_star : float

    Tt_j_star : float

    theta : float

    delta : float

    M_0 : float

    c_0 : float

    f : np.ndarray

    n_engines: int

            
    Returns
    -------
    np.ndarray
        Mean-square acoustic pressure of jet mixing source noise

    """

    tables = JetMixingNoiseTables()

    r_s_star = _R_SOURCE / np.sqrt(_A_REF)

    # Calculate density exponent (omega)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table II
    omega = tables.get_density_exponent(np.log10(V_j_star))

    # Calculate power deviation factor (P)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table III
    p = 10**tables.get_power_deviation_factor(np.log10(V_j_star))
    
    # Calculate acoustic power (Pi_star)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 3
    Pi_star = _K_JET * rho_j_star ** omega * V_j_star ** 8 * p

    # Calculate directivity function (D)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
    directivity = 10**tables.get_directivity(theta, np.log10(V_j_star))

    # Calculate Strouhal frequency adjustment factor (xi)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
    # TODO: check xi = min(1, xi)
    xi = tables.get_strouhal_correction(V_j_star, theta)

    # Calculate Strouhal number (St)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Eq. 9
    D_j_star = np.sqrt(4 * A_j_star / np.pi)  # Jet diamater [-] (rel. to sqrt(A_e))
    f_star = f * np.sqrt(_A_REF) / c_0
    St = (f_star * D_j_star) / (xi * (V_j_star - M_0))

    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VI
    spectral_distribution = 10**(-tables.get_spectral_distribution(theta, Tt_j_star, np.log10(V_j_star), np.log10(St), f.size)/10)

    # Calculate forward velocity index (m_theta)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VII
    m_theta = tables.get_forward_velocity_index(theta)

    # Calculate mean-square acoustic pressure (msap)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 8
    msap = Pi_star * A_j_star / (4 * np.pi * r_s_star ** 2) * directivity * spectral_distribution / (1 - M_0 * np.cos(np.pi / 180. * (theta - delta))) * ((V_j_star - M_0) / V_j_star) ** m_theta

    # Multiply with number of engines
    # Normalize msap by reference pressure
    return msap * n_engines / _P_REF**2


def compute_jet_shock_source_noise(V_j_star, M_j, A_j_star, Tt_j_star, theta, delta, M_0, c_0, f, n_shock, n_engines):
                     
    """
    Compute jet mixing noise mean-square acoustic pressure (msap).

    Parameters
    ----------
    V_j_star : 
    
    M_j : 
    
    A_j_star : 
    
    Tt_j_star : 
    
    M_0 : 
    
    c_0 : 

    f : np.ndarray

    n_shock : int
    
    n_engines : int
    
    Returns
    -------
    np.ndarray : 

    """

    tables = JetShockNoiseTables()

    r_s_star = _R_SOURCE / np.sqrt(_A_REF)

    # Calculate msap for all frequencies
    # If the jet is supersonic: shock cell noise
    if M_j > 1:
        # Calculate beta function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 4
        beta = (M_j ** 2 - 1) ** 0.5

        # Calculate eta (exponent of the pressure ratio parameter)
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 5
        if beta > 1:
            if Tt_j_star < 1.1:
                eta = 1.
            else:
                eta = 2.
        else:
            eta = 4.

        # Calculate f_star
        # Source: Zorumski report 1982 part 2. Chapter 8.5 page 8-5-1 (symbols)
        f_star = f * np.sqrt(_A_REF) / c_0

        # Calculate sigma parameter
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 3
        sigma = 7.80 * beta * (1 - M_0 * np.cos(np.pi / 180 * theta)) * np.sqrt(A_j_star) * f_star

        # Calculate W function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 6-7
        b = 0.23077
        W = 0
        for k in np.arange(1, n_shock):
            sum_inner = 0
            for m in np.arange(n_shock - k):
                # Calculate q_km
                q_km = 1.70 * k / V_j_star * (1 - 0.06 * (m + (k + 1) / 2)) * (1 + 0.7 * V_j_star * np.cos(np.pi / 180 * theta))

                # Calculate inner sum (note: the factor b in the denominator below the sine should not be there: to get same graph as Figure 4)
                sum_inner = sum_inner + np.sin((b * sigma * q_km / 2)) / (sigma * q_km) * np.cos(sigma * q_km)

            # Compute the correlation coefficient spectrum C
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Table II
            C = tables.get_correlation_coefficient_spectrum(np.log10(sigma))

            # Add outer loop to the shock cell interference function
            W = W + (4. / (n_shock * b))* sum_inner * C ** (k ** 2)

        # Calculate the H function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Table III (+ linear extrapolation in logspace for log10sigma < 0; as given in SAEARP876)
        log10H = tables.get_group_source_strength_spectrum(np.log10(sigma))

        # Source: Zorumski report 1982 part 2. Chapter 8.5.4
        if Tt_j_star < 1.1:
            log10H = log10H - 0.2
        H = (10 ** log10H)

        # Calculate mean-square acoustic pressure (msap)
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 1
        msap = 1.92e-3 * A_j_star / (4 * np.pi * r_s_star ** 2) * (1 + W) / (1 - M_0 * np.cos(np.pi / 180. * (theta - delta))) ** 4 * beta ** eta * H
    
    else:
        msap = np.zeros(f.size) * M_j ** 0

    # Normalize msap by reference pressure
    return msap * n_engines / _P_REF**2


