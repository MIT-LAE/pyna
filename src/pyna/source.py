import numpy as np
from pyna.noise_tables import (
    CoreNoiseTables,
    JetMixingNoiseTables,
    JetShockNoiseTables,
    FanNoiseTables,
    AirframeNoiseTables
)

from pyna.constants import (
    _R_SOURCE,
    _A_REF,
    _RHO_SEALEVEL,
    _C_SEALEVEL,
    _P_REF,

)

_K_JET = 6.67e-5
_K_WING_CONVENTIONAL = 4.464e-5
_K_WING_AERODYNAMICALLY_CLEAN = 7.075e-6

# Fan noise modules
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

def calculate_inlet_harmonics(inlet_tones, M_tip, M_tip_tangential, blade_number, vane_number, n_harmonics, theta, blade_pass_frequency, frequency, method_rotor_stator_interactions, flight_segment, flag_inlet_distortions, flag_inlet_guide_vanes):
    """
    Compute fan tone harmonics for inlet (dp) and discharge (dpx).

    Parameters
    ----------

    Returns
    -------


    """

    tables = FanNoiseTables()

    # Assign discrete interaction tones at blade_pass_frequency and harmonics to proper bins (see figures 8 and 9):
    # Initialize solution matrices
    dp = np.zeros(frequency.size)
    
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
        for l in np.arange(nfi - 1, frequency.size):
            f_ratio = blade_pass_frequency * i_harmonic / frequency[l]
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

def calculate_discharge_harmonics(discharge_tones, M_tip, M_tip_tangential, blade_number, vane_number, n_harmonics, blade_pass_frequency, frequency, method_rotor_stator_interactions, flag_inlet_guide_vanes):
    """
    Compute fan tone harmonics for inlet (dp) and discharge (dpx).

    Parameters
    ----------

    Returns
    -------


    """

    tables = FanNoiseTables()

    # Assign discrete interaction tones at blade_pass_frequency and harmonics to proper bins (see figures 8 and 9):
    # Initialize solution matrices
    dpx = np.zeros(frequency.size)

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
        for l in np.arange(nfi - 1, frequency.size):
            f_ratio = blade_pass_frequency * i_harmonic / frequency[l]
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

def compute_combination_tone_level(temperature_term, M_tip, theta, blade_pass_frequency, frequency, method_rotor_stator_interactions, flag_inlet_guide_vanes):
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

    blade_pass_frequency : float

    frequency : np.ndarray

    method_rotor_stator_interactions : str

    flag_inlet_guide_vanes : bool

    Returns
    -------
    np.ndarray : 

    """

    tables = FanNoiseTables()

    # Initialize solution matrices
    dcp = np.zeros(frequency.size)

    if M_tip >= 1:

        directivity = tables.get_combination_tones_directivity(theta, method_rotor_stator_interactions)

        # Noise adjustment (reduction) if there are inlet guide vanes, for all methods:
        if flag_inlet_guide_vanes:
            igv_term = -5
        else:
            igv_term = 0

        # Loop through the three sub-blade_pass_frequency terms (k = 1; 1/2 blade_pass_frequency term, k = 2; 1/4 blade_pass_frequency term, k = 3; 1/8 blade_pass_frequency term)
        for k in np.arange(1, 4):
            
            # Tip Mach-dependent term (F1 of Eqn 8 in Heidmann report, Figure 15A)
            tipmach_term = tables.get_combination_tones_tipmach(M_tip, k, method_rotor_stator_interactions)
            
            # Cycle through frequencies and make assignments:
            for j in np.arange(frequency.size):
                # Frequency-dependent term (F3 of Eqn 9, Figure 14):

                spectral_term = tables.get_combination_tones_spectral_distribution(frequency[j] / blade_pass_frequency, k, method_rotor_stator_interactions)

                # Be sure to add the three sub-blade_pass_frequency components together at each frequency:
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

def compute_fan_inlet_spectral_distribution(blade_pass_frequency, frequency, method_broadband):
    
    # Spectral distribution
    match method_broadband:
        case 'alliedsignal':
            spectral_distribution = 2.445096095 * (np.log(frequency / blade_pass_frequency / 2)) ** 2
            spectral_distribution[frequency/blade_pass_frequency > 2] = (13.97197769 * (np.log(frequency / blade_pass_frequency / 2)) ** 2)[frequency/blade_pass_frequency > 2]

        case 'kresja':
            spectral_distribution = 3.4929944 * (np.log(frequency / blade_pass_frequency / 4)) ** 2

        case _:
            spectral_distribution = 3.4929944 * (np.log(frequency / blade_pass_frequency / 2.5)) ** 2

    return spectral_distribution

def compute_fan_discharge_spectral_distribution(blade_pass_frequency, frequency, method_broadband):
    
    # Spectral distribution
    match method_broadband:
        case 'alliedsignal':
            spectral_distribution[frequency/blade_pass_frequency > 2] = (13.97197769 * (np.log(frequency / blade_pass_frequency / 2)) ** 2)[frequency/blade_pass_frequency > 2]
            
        case 'kresja':
            spectral_distribution = 3.4929944 * (np.log(frequency / blade_pass_frequency / 2.5)) ** 2

        case _:
            spectral_distribution = 3.4929944 * (np.log(frequency / blade_pass_frequency / 2.5)) ** 2

    return spectral_distribution

def compute_fan_source_noise(dTt_fan_star, mdot_fan_star, N_fan_star, A_fan_star, d_fan_star, 
                             blade_number, vane_number, M_tip_rel_design, rotor_stator_spacing, 
                             theta, M_0, c_0, T_0, rho_0, 
                             n_harmonics, n_engines, 
                             frequency,
                             method_broadband, method_rotor_stator_interactions, noise_direction, flight_segment,
                             flag_broadband, flag_tones, flag_combination_tones, flag_inlet_distortions, flag_inlet_guide_vanes, flag_liner_suppression):
    """
    Calculates fan noise mean-square acoustic pressure (msap) using Berton's implementation of the fan noise method.

    Parameters
    ----------
    dTt_fan_star : float

    mdot_fan_star : float

    N_fan_star : float

    A_fan_star : float

    d_fan_star : float

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

    frequency : np.ndarray

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
    temperature_rise = dTt_fan_star * T_0  # [K]
    rpm = N_fan_star * 60 * c_0 / (d_fan_star * np.sqrt(_A_REF))  # [rpm]
    M_tip_tangential = (d_fan_star * np.sqrt(_A_REF) / 2) * rpm * 2 * np.pi / 60 / c_0
    mdot_fan = mdot_fan_star * rho_0 * c_0 * _A_REF  # [kg/s]
    blade_pass_frequency = rpm * blade_number / 60. / (1 - M_0 * np.cos(theta * np.pi / 180))  # [Hz]
    M_flow = mdot_fan / (rho_0 * A_fan_star * _A_REF * c_0)
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
                spectral_distribution = compute_fan_inlet_spectral_distribution(blade_pass_frequency, frequency, method_broadband)
                msap += 10 ** (0.1 * (broadband_level - spectral_distribution))

            if flag_tones:
                tone_level = compute_inlet_tone_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_rotor_stator_interactions, flag_inlet_distortions)
                harmonics = calculate_inlet_harmonics(tone_level, M_tip, M_tip_tangential, blade_number, vane_number, n_harmonics, theta, blade_pass_frequency, frequency, method_rotor_stator_interactions, flight_segment, flag_inlet_distortions, flag_inlet_guide_vanes)
                msap += harmonics

        case "discharge":
            if flag_broadband:
                broadband_level = compute_discharge_broadband_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_broadband, flag_inlet_distortions, flag_inlet_guide_vanes)
                spectral_distribution = compute_fan_discharge_spectral_distribution(blade_pass_frequency, frequency, method_broadband)
                msap += 10 ** (0.1 * (broadband_level - spectral_distribution))
            
            if flag_tones:
                tone_level = compute_discharge_tone_level(temperature_term, M_tip, M_tip_rel_design, rotor_stator_spacing, theta, method_rotor_stator_interactions, flag_inlet_distortions, flag_inlet_guide_vanes)
                harmonics = calculate_discharge_harmonics(tone_level, M_tip, M_tip_tangential, blade_number, vane_number, n_harmonics, blade_pass_frequency, frequency, method_rotor_stator_interactions, flag_inlet_guide_vanes)
                msap += harmonics
            
    # Compute combination tones
    if flag_combination_tones and M_tip > 0:
        combination_tone_level = compute_combination_tone_level(temperature_term, M_tip, theta, blade_pass_frequency, frequency, method_rotor_stator_interactions, flag_inlet_guide_vanes)
        msap += combination_tone_level
            
    # Fan liner suppression
    if flag_liner_suppression:
        msap *= tables.get_liner_suppression(frequency, theta, noise_direction)

    # Multiply for number of engines
    return msap * n_engines

    # TODO: implement shielding
    # Fan inlet shielding
    # if settings['shielding'] and component == 'fan_inlet':
    #     msap_j = msap_j / (10 ** (shield[i, :] / 10.))
    # msap_fan[i, :] = msap_j
    # return msap_fan


# Core noise modules
def compute_core_source_noise(mdot_combustor_inlet_star, Tt_combustor_inlet_star, Tt_combustor_outlet_star, Pt_combustor_inlet_star, dTt_combustor_design_star, theta, M_0, frequency, n_engines):
    """
	Compute core noise mean-square acoustic pressure (msap).

	Parameters
	----------
	mdot_combustor_inlet_star : float
    
    Tt_combustor_inlet_star : float
    
    Tt_combustor_outlet_star : float
    
    Pt_combustor_inlet_star : float
    
    dTt_combustor_design_star : float
    
    theta : float
    
    M_0 : float
    
    c_0 : float
    
    frequency : np.ndarray
    
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
    g_TT = dTt_combustor_design_star ** (-4)
    # Source: Hultgren, 2012: A comparison of combustor models Equation 6
    # elif settings['core_turbine_attenuation_method'] == 'pw':
    #     zeta = (rho_te_c_star[i] * c_te_c_star[i]) / (rho_ti_c_star[i] * c_ti_c_star[i])
    #     g_TT = 0.8 * zeta / (1 + zeta) ** 2
    # else:
    #     raise ValueError('Invalid method to account for turbine attenuation effects of combustor noise. Specify GE/PW.')

    # Calculate acoustic power (Pi_star)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    Pi_star = 8.85e-7 * (mdot_combustor_inlet_star / A_c_star) * ((Tt_combustor_outlet_star - Tt_combustor_inlet_star) / Tt_combustor_inlet_star) ** 2 * Pt_combustor_inlet_star ** 2 * g_TT

    # Calculate directivity function (D)
    # Take the D function as SAE ARP876E Table 18 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table II
    directivity = 10 ** tables.get_directivity(theta)

    # Calculate the spectral function (S)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
    f_p = 400. / (1 - M_0 * np.cos(theta * np.pi / 180.))

    # Take the S function as SAE ARP876E Table 17 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table III
    spectral_distribution = 10 ** tables.get_spectral_distribution(np.log10(frequency/f_p))

    # Calculate mean-square acoustic pressure (msap)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
    msap = Pi_star * A_c_star / (4 * np.pi * r_s_star ** 2) * directivity * spectral_distribution / (1. - M_0 * np.cos(np.pi / 180. * theta)) ** 4

    # Multiply with number of engines
    msap = msap * n_engines

    # Normalize msap by reference pressure
    return msap/_P_REF**2


# Jet noise modules
def compute_jet_mixing_source_noise(V_jet_star, rho_jet_star, A_jet_star, Tt_jet_star, theta, delta_jet, M_0, c_0, frequency, n_engines):
    """
    Compute jet mixing noise mean-square acoustic pressure (msap).

    Parameters
    ----------
    V_jet_star : float

    rho_jet_star : float

    A_jet_star : float

    Tt_jet_star : float

    theta : float

    delta_jet : float

    M_0 : float

    c_0 : float

    frequency : np.ndarray

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
    omega = tables.get_density_exponent(np.log10(V_jet_star))

    # Calculate power deviation factor (P)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table III
    p = 10**tables.get_power_deviation_factor(np.log10(V_jet_star))
    
    # Calculate acoustic power (Pi_star)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 3
    Pi_star = _K_JET * rho_jet_star ** omega * V_jet_star ** 8 * p

    # Calculate directivity function (D)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
    directivity = 10**tables.get_directivity(theta, np.log10(V_jet_star))

    # Calculate Strouhal frequency adjustment factor (xi)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
    # TODO: check xi = min(1, xi)
    xi = tables.get_strouhal_correction(V_jet_star, theta)

    # Calculate Strouhal number (St)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Eq. 9
    D_j_star = np.sqrt(4 * A_jet_star / np.pi)  # Jet diamater [-] (rel. to sqrt(A_e))
    f_star = frequency * np.sqrt(_A_REF) / c_0
    St = (f_star * D_j_star) / (xi * (V_jet_star - M_0))

    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VI
    spectral_distribution = 10**(-tables.get_spectral_distribution(theta, Tt_jet_star, np.log10(V_jet_star), np.log10(St), frequency.size)/10)

    # Calculate forward velocity index (m_theta)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VII
    m_theta = tables.get_forward_velocity_index(theta)

    # Calculate mean-square acoustic pressure (msap)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 8
    msap = Pi_star * A_jet_star / (4 * np.pi * r_s_star ** 2) * directivity * spectral_distribution / (1 - M_0 * np.cos(np.pi / 180. * (theta - delta_jet))) * ((V_jet_star - M_0) / V_jet_star) ** m_theta

    # Multiply with number of engines
    # Normalize msap by reference pressure
    return msap * n_engines / _P_REF**2

def compute_jet_shock_source_noise(V_jet_star, M_jet, A_jet_star, Tt_jet_star, theta, delta_jet, M_0, c_0, frequency, n_shock, n_engines):
                     
    """
    Compute jet mixing noise mean-square acoustic pressure (msap).

    Parameters
    ----------
    V_jet_star : 
    
    M_jet : 
    
    A_jet_star : 
    
    Tt_jet_star : 
    
    M_0 : 
    
    c_0 : 

    frequency : np.ndarray

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
    if M_jet > 1:
        # Calculate beta function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 4
        beta = (M_jet ** 2 - 1) ** 0.5

        # Calculate eta (exponent of the pressure ratio parameter)
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 5
        if beta > 1:
            if Tt_jet_star < 1.1:
                eta = 1.
            else:
                eta = 2.
        else:
            eta = 4.

        # Calculate f_star
        # Source: Zorumski report 1982 part 2. Chapter 8.5 page 8-5-1 (symbols)
        f_star = frequency * np.sqrt(_A_REF) / c_0

        # Calculate sigma parameter
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 3
        sigma = 7.80 * beta * (1 - M_0 * np.cos(np.pi / 180 * theta)) * np.sqrt(A_jet_star) * f_star

        # Calculate W function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 6-7
        b = 0.23077
        W = 0
        for k in np.arange(1, n_shock):
            sum_inner = 0
            for m in np.arange(n_shock - k):
                # Calculate q_km
                q_km = 1.70 * k / V_jet_star * (1 - 0.06 * (m + (k + 1) / 2)) * (1 + 0.7 * V_jet_star * np.cos(np.pi / 180 * theta))

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
        if Tt_jet_star < 1.1:
            log10H = log10H - 0.2
        H = (10 ** log10H)

        # Calculate mean-square acoustic pressure (msap)
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 1
        msap = 1.92e-3 * A_jet_star / (4 * np.pi * r_s_star ** 2) * (1 + W) / (1 - M_0 * np.cos(np.pi / 180. * (theta - delta_jet))) ** 4 * beta ** eta * H
    
    else:
        msap = np.zeros(frequency.size) * M_jet ** 0

    # Normalize msap by reference pressure
    return msap * n_engines / _P_REF**2


# Airframe noise modules
def compute_wing_noise(wing_span, wing_area, M_0, c_0, rho_0, mu_0, theta, phi, frequency, flag_delta_wing, flag_aerodynamically_clean_wing):
    """Compute wing trailing edge mean-square acoustic pressure (msap).

    Parameters
    ----------
    wing_span : float

    wing_area : float

    M_0 : float
        ambient Mach number [-]
    c_0 : float
        ambient speed of sound [m/s]
    rho_0 : float
        ambient density [kg/m3]
    mu_0 : float
        ambient dynamic viscosity [kg/m/s]
    theta : float
        polar directivity angle [deg]
    phi : float
        azimuthal directivity angle [deg]
    frequency : np.ndarray
        1/3rd octave frequency [Hz]
    flag_delta_wing : bool
        
    flag_aerodynamically_clean_wing : bool

    Returns
    -------
    np.ndarray

    """

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    boundary_layer_thickness_star = 0.37 * (wing_area / wing_span ** 2) * (rho_0 * M_0 * c_0 * wing_area / (mu_0 * wing_span)) ** (-0.2)

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if flag_aerodynamically_clean_wing:
        acoustic_power = _K_WING_AERODYNAMICALLY_CLEAN * M_0 ** 5 * boundary_layer_thickness_star
    else:
        acoustic_power = _K_WING_CONVENTIONAL * M_0 ** 5 * boundary_layer_thickness_star

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    directivity = 4. * np.cos(phi * np.pi / 180.) ** 2 * np.cos(theta / 2 * np.pi / 180.) ** 2

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    strouhal = frequency * boundary_layer_thickness_star * wing_span / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
    if flag_delta_wing:
        spectral_distribution = 0.613 * (10 * strouhal) ** 4 * ((10 * strouhal) ** 1.35 + 0.5) ** (-4)
    else:
        spectral_distribution = 0.485 * (10 * strouhal) ** 4 * ((10 * strouhal) ** 1.5 + 0.5) ** (-4)

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_source_star = _R_SOURCE/wing_span
    return 1 / (4 * np.pi * r_source_star ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (acoustic_power * directivity * spectral_distribution)

def compute_horizontal_tail_noise(horizontal_tail_span, horizontal_tail_area, wing_span, M_0, c_0, rho_0, mu_0, theta, phi, frequency, flag_aerodynamically_clean_wing):
    """
    Compute horizontal tail trailing edge mean-square acoustic pressure (msap).

    Parameters
    ----------
    horizontal_tail_span : float
     
    horizontal_tail_area : float
    
    wing_span : float

    M_0 : float
        ambient Mach number [-]
    c_0 : float
        ambient speed of sound [m/s]
    rho_0 : float
        ambient density [kg/m3]
    mu_0 : float
        ambient dynamic viscosity [kg/m/s]
    theta : float
        polar directivity angle [deg]
    phi : float
        azimuthal directivity angle [deg]
    frequency : np.ndarray
        1/3rd octave frequency [Hz]
        
    flag_aerodynamically_clean_wing : bool

    Returns
    -------
    np.ndarray

    """

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    boundary_layer_thickness_star = 0.37 * (horizontal_tail_area / horizontal_tail_span ** 2) * (rho_0 * M_0 * c_0 * horizontal_tail_area / (mu_0 * horizontal_tail_span)) ** (-0.2)

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if flag_aerodynamically_clean_wing:
        acoustic_power = _K_WING_AERODYNAMICALLY_CLEAN * M_0 ** 5 * boundary_layer_thickness_star * (horizontal_tail_span / wing_span) ** 2
    else:
        acoustic_power = _K_WING_CONVENTIONAL * M_0 ** 5 * boundary_layer_thickness_star * (horizontal_tail_span / wing_span) ** 2
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    directivity = 4 * np.cos(phi * np.pi / 180.) ** 2 * np.cos(theta / 2 * np.pi / 180.) ** 2

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    strouhal = frequency * boundary_layer_thickness_star * horizontal_tail_span / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
    spectral_distribution = 0.485 * (10 * strouhal) ** 4 * ((10 * strouhal) ** 1.5 + 0.5) ** (-4)

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_source_star = _R_SOURCE / wing_span
    return 1. / (4. * np.pi * r_source_star ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (acoustic_power * directivity * spectral_distribution)

def compute_vertical_tail_noise(vertical_tail_span, vertical_tail_area, wing_span, M_0, c_0, rho_0, mu_0, theta, phi, frequency, flag_delta_wing, flag_aerodynamically_clean_wing):
    """
    Compute vertical tail trailing edge mean-square acoustic pressure (msap).

    Parameters
    ----------
    vertical_tail_span : float
    
    vertical_tail_area, wing_span : float

    M_0 : float
        ambient Mach number [-]
    c_0 : float
        ambient speed of sound [m/s]
    rho_0 : float
        ambient density [kg/m3]
    mu_0 : float
        ambient dynamic viscosity [kg/m/s]
    theta : float
        polar directivity angle [deg]
    phi : float
        azimuthal directivity angle [deg]
    frequency : np.ndarray
        1/3rd octave frequency [Hz]
    flag_delta_wing : bool
        
    flag_aerodynamically_clean_wing : bool

    Returns
    -------
    np.ndarray

    """

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    boundary_layer_thickness_star = 0.37 * (vertical_tail_area / vertical_tail_span ** 2) * (rho_0 * M_0 * c_0 * vertical_tail_area / (mu_0 * vertical_tail_span)) ** (-0.2)

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if flag_aerodynamically_clean_wing:
        acoustic_power = _K_WING_AERODYNAMICALLY_CLEAN * M_0 ** 5 * boundary_layer_thickness_star * (vertical_tail_span / wing_span) ** 2
    else:
        acoustic_power = _K_WING_CONVENTIONAL * M_0 ** 5 * boundary_layer_thickness_star * (vertical_tail_span / wing_span) ** 2
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    directivity = 4 * np.sin(phi * np.pi / 180.) ** 2 * np.cos(theta / 2 * np.pi / 180.) ** 2

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    strouhal = frequency * boundary_layer_thickness_star * vertical_tail_span / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
    if flag_delta_wing:
        spectral_distribution = 0.613 * (10 * strouhal) ** 4 * ((10 * strouhal) ** 1.35 + 0.5) ** (-4)
    else:
        spectral_distribution = 0.485 * (10 * strouhal) ** 4 * ((10 * strouhal) ** 1.35 + 0.5) ** (-4)

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_source_star = _R_SOURCE / wing_span
    return 1. / (4 * np.pi * r_source_star ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (acoustic_power * directivity * spectral_distribution)

def compute_leading_edge_slat_noise(wing_span, wing_area, M_0, c_0, rho_0, mu_0, theta, phi, frequency):
    """
    Compute leading-edge slat mean-square acoustic pressure (msap).

    Parameters
    ----------
    wing_span : float
     
    wing_area : float

    M_0 : float
        ambient Mach number [-]
    c_0 : float
        ambient speed of sound [m/s]
    rho_0 : float
        ambient density [kg/m3]
    mu_0 : float
        ambient dynamic viscosity [kg/m/s]
    theta : float
        polar directivity angle [deg]
    phi : float
        azimuthal directivity angle [deg]
    frequency : np.ndarray
        1/3rd octave frequency [Hz]
    flag_delta_wing : bool
        
    Returns
    -------
    np.ndarray

    """
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    boundary_layer_thickness_star = 0.37 * (wing_area / wing_span ** 2) * (rho_0 * M_0 * c_0 * wing_area / (mu_0 * wing_span)) ** (-0.2)

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 4
    acoustic_power_1 = 4.464e-5 * M_0 ** 5 * boundary_layer_thickness_star  # Slat noise
    acoustic_power_2 = 4.464e-5 * M_0 ** 5 * boundary_layer_thickness_star  # Added trailing edge noise
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    directivity = 4 * np.cos(phi * np.pi / 180.) ** 2 * np.cos(theta / 2 * np.pi / 180.) ** 2
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-12-13
    strouhal = frequency * boundary_layer_thickness_star * wing_span / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
    spectral_distribution_1 = 0.613 * (10 * strouhal) ** 4 * ((10. * strouhal) ** 1.5 + 0.5) ** (-4)
    spectral_distribution_2 = 0.613 * (2.19 * strouhal) ** 4 * ((2.19 * strouhal) ** 1.5 + 0.5) ** (-4)
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_source_star = _R_SOURCE / wing_span
    return 1 / (4 * np.pi * r_source_star ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (
            acoustic_power_1 * directivity * spectral_distribution_1 + acoustic_power_2 * directivity * spectral_distribution_2
        )

def compute_trailing_edge_flap_noise(theta_flaps, flaps_span, flaps_area, wing_span, flaps_slot_number, M_0, c_0, theta, phi, frequency):
    """
    Compute trailing-edge flap mean-square acoustic pressure (msap).

    Parameters
    ----------
    flaps_span : float
    
    flaps_area : float
    
    wing_span : float
    
    flaps_slot_number : int

    M_0 : float
        ambient Mach number [-]
    c_0 : float
        ambient speed of sound [m/s]
    rho_0 : float
        ambient density [kg/m3]
    mu_0 : float
        ambient dynamic viscosity [kg/m/s]
    theta : float
        polar directivity angle [deg]
    phi : float
        azimuthal directivity angle [deg]
    frequency : np.ndarray
        1/3rd octave frequency [Hz]
    flag_delta_wing : bool
        
    Returns
    -------
    np.ndarray

    """
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 14-15
    match flaps_slot_number:
        case 1 | 2:
            acoustic_power = 2.787e-4 * M_0 ** 6 * flaps_area / wing_span ** 2 * np.sin(theta_flaps * np.pi / 180.) ** 2
        case 3:
            acoustic_power = 3.509e-4 * M_0 ** 6 * flaps_area / wing_span ** 2 * np.sin(theta_flaps * np.pi / 180.) ** 2
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 16
    directivity = 3 * (
            np.sin(theta_flaps * np.pi / 180.) * np.cos(theta * np.pi / 180.) + 
            np.cos(theta_flaps * np.pi / 180.) * np.sin(theta * np.pi / 180.) * np.cos(phi * np.pi / 180.)
        ) ** 2

    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 17-18-19
    strouhal = frequency * flaps_area / (M_0 * flaps_span * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))

    match flaps_slot_number:
        case 1| 2:
            spectral_distribution = 216.49 * strouhal ** (-3)
            spectral_distribution[strouhal < 2] = (0.0480 * strouhal)[strouhal < 2]
            spectral_distribution[(2 <= strouhal)*(strouhal <= 20)] = (0.1406 * strouhal ** (-0.55))[(2 <= strouhal)*(strouhal <= 20)]
        case 3:
            spectral_distribution = 17078 * strouhal ** (-3)
            spectral_distribution[strouhal < 2] = (0.0257 * strouhal)[strouhal < 2]
            spectral_distribution[(2 <= strouhal)*(strouhal <= 75)] = (0.0536 * strouhal ** (-0.0625))[(2 <= strouhal)*(strouhal <= 75)]
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_source_star = _R_SOURCE / wing_span
    return 1. / (4 * np.pi * r_source_star ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (acoustic_power * directivity * spectral_distribution)

def compute_landing_gear_noise(i_landing_gear, 
                         main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, 
                         nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length, wing_span, 
                         M_0, c_0, theta, phi, frequency):
    """
    Compute landing gear mean-square acoustic pressure (msap)

    Parameters
    ----------
    main_gear_tire_diameter : float
    
    nose_gear_tire_diameter : float
    
    main_gear_wheel_number : int
    
    nose_gear_wheel_number : int
    
    main_gear_length : float
    
    nose_gear_length : float
    
    wing_span : float
    
    i_landing_gear : bool
    
    M_0 : float
        ambient Mach number [-]
    c_0 : float
        ambient speed of sound [m/s]
    rho_0 : float
        ambient density [kg/m3]
    mu_0 : float
        ambient dynamic viscosity [kg/m/s]
    theta : float
        polar directivity angle [deg]
    phi : float
        azimuthal directivity angle [deg]
    frequency : np.ndarray
        1/3rd octave frequency [Hz]
        
    Returns
    -------
    np.ndarray

    """

    if i_landing_gear == 1:

        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
        strouhal_nose_gear = frequency * nose_gear_tire_diameter / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
        
        # Calculate noise power and spectral distribution function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
        match nose_gear_wheel_number:
            case 1 | 2:
                acoustic_power_nose_gear_wheel = 4.349e-4 * M_0 ** 6 * nose_gear_wheel_number * (nose_gear_tire_diameter / wing_span) ** 2
                acoustic_power_nose_gear_strut = 2.753e-4 * M_0 ** 6 * (nose_gear_tire_diameter / wing_span) ** 2 * (nose_gear_length / nose_gear_tire_diameter)
                spectral_distribution_nose_gear_wheel = 13.59 * strouhal_nose_gear ** 2 * (12.5 + strouhal_nose_gear ** 2) ** (-2.25)
                spectral_distribution_nose_gear_strut = 5.32 * strouhal_nose_gear ** 2 * (30 + strouhal_nose_gear ** 8) ** (-1)
            case 4:
                acoustic_power_nose_gear_wheel = 3.414 - 4 * M_0 ** 6 * nose_gear_wheel_number * (nose_gear_tire_diameter / wing_span) ** 2
                acoustic_power_nose_gear_strut = 2.753e-4 * M_0 ** 6 * (nose_gear_tire_diameter / wing_span) ** 2 * (nose_gear_length / nose_gear_tire_diameter)
                spectral_distribution_nose_gear_wheel = 0.0577 * strouhal_nose_gear ** 2 * (1 + 0.25 * strouhal_nose_gear ** 2) ** (-1.5)
                spectral_distribution_nose_gear_strut = 1.28 * strouhal_nose_gear ** 3 * (1.06 + strouhal_nose_gear ** 2) ** (-3)
        
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
        strouhal_main_gear = frequency * main_gear_tire_diameter / (M_0 * c_0) * (1 - M_0 * np.cos(theta * np.pi / 180.))
        
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
        match main_gear_wheel_number:
            case 1 | 2:
                acoustic_power_main_gear_wheel = 4.349e-4 * M_0 ** 6 * main_gear_wheel_number * (main_gear_tire_diameter / wing_span) ** 2
                acoustic_power_main_gear_strut = 2.753e-4 * M_0 ** 6 * (main_gear_tire_diameter / wing_span) ** 2 * (main_gear_length / main_gear_tire_diameter)
                spectral_distribution_main_gear_wheel = 13.59 * strouhal_main_gear ** 2 * (12.5 + strouhal_main_gear ** 2) ** (-2.25)
                spectral_distribution_main_gear_strut = 5.32 * strouhal_main_gear ** 2 * (30 + strouhal_main_gear ** 8) ** (-1)
            case 4:
                acoustic_power_main_gear_wheel = 3.414e-4 * M_0 ** 6 * main_gear_wheel_number * (main_gear_tire_diameter / wing_span) ** 2
                acoustic_power_main_gear_strut = 2.753e-4 * M_0 ** 6 * (main_gear_tire_diameter / wing_span) ** 2 * (main_gear_length / main_gear_tire_diameter)
                spectral_distribution_main_gear_wheel = 0.0577 * strouhal_main_gear ** 2 * (1 + 0.25 * strouhal_main_gear ** 2) ** (-1.5)
                spectral_distribution_main_gear_strut = 1.28 * strouhal_main_gear ** 3 * (1.06 + strouhal_main_gear ** 2) ** (-3)
        
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 23-24
        directivity_wheel = 1.5 * np.sin(theta * np.pi / 180.) ** 2
        directivity_strut = 3 * np.sin(theta * np.pi / 180.) ** 2 * np.sin(phi * np.pi / 180.) ** 2
        
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
        # If landing gear is down
        r_source_star = _R_SOURCE / wing_span
        return 1 / (4 * np.pi * r_source_star ** 2) / (1 - M_0 * np.cos(theta * np.pi / 180.)) ** 4 * (
                    nose_gear_number * (acoustic_power_nose_gear_wheel * spectral_distribution_nose_gear_wheel * directivity_wheel + 
                                        acoustic_power_nose_gear_strut * spectral_distribution_nose_gear_strut * directivity_strut) +
                    main_gear_number * (acoustic_power_main_gear_wheel * spectral_distribution_main_gear_wheel * directivity_wheel + 
                                        acoustic_power_main_gear_strut * spectral_distribution_main_gear_strut * directivity_strut)
                    )

    else:
        return np.zeros(frequency.size)

def compute_airframe_source_noise(noise_component_lst,
                                  wing_span, wing_area, 
                                  horizontal_tail_span, horizontal_tail_area,
                                  vertical_tail_span, vertical_tail_area,
                                  theta_flaps, flaps_span, flaps_area, flaps_slot_number,
                                  i_landing_gear, main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length,
                                  M_0, c_0, rho_0, mu_0, theta, phi, frequency, 
                                  flag_delta_wing, flag_aerodynamically_clean_wing, flag_high_speed_research_suppression):
    """
    Compute airframe noise mean-square acoustic pressure (msap).

    Parameters
    ----------
    noise_components : []

    wing_span : float
    
    wing_area  : float
    
    horizontal_tail_span : float
    
    horizontal_tail_area : float
    
    vertical_tail_span : float
    
    vertical_tail_area : float
    
    theta_flaps : float
    
    flaps_span : float
    
    flaps_area : float
    
    flaps_slot_number : int

    i_landing_gear : bool

    main_gear_tire_diameter : float
    
    main_gear_wheel_number : int

    main_gear_length : float
    
    nose_gear_tire_diameter : float
    
    nose_gear_wheel_number : int

    nose_gear_length : float
    
    M_0 : float
    
    c_0 : float
    
    rho_0 : float
    
    mu_0 : float
    
    theta : float

    phi : float

    frequency : np.ndarray
    
    flag_delta_wing : bool
    
    flag_aerodynamically_clean_wing : bool
    
    flag_high_speed_research_suppression : bool

    Returns
    -------
    np.ndarray

    """
    
    tables = AirframeNoiseTables()

    # Allocate
    msap = np.zeros(frequency.size,)

    if M_0 != 0:
        
        if 'wing' in noise_component_lst:
            msap += compute_wing_noise(wing_span, wing_area, M_0, c_0, rho_0, mu_0, theta, phi, frequency, flag_delta_wing, flag_aerodynamically_clean_wing)

        if 'vertical_tail' in noise_component_lst:
            msap += compute_vertical_tail_noise(vertical_tail_span, vertical_tail_area, wing_span, M_0, c_0, rho_0, mu_0, theta, phi, frequency, flag_delta_wing, flag_aerodynamically_clean_wing)
                                                              
        if 'horizontal_tail' in noise_component_lst:
            msap += compute_horizontal_tail_noise(horizontal_tail_span, horizontal_tail_area, wing_span, M_0, c_0, rho_0, mu_0, theta, phi, frequency, flag_aerodynamically_clean_wing)

        if 'leading_edge_slats' in noise_component_lst:
            msap += compute_leading_edge_slat_noise(wing_span, wing_area, M_0, c_0, rho_0, mu_0, theta, phi, frequency)
            
        if 'trailing_edge_flaps' in noise_component_lst:
            msap += compute_trailing_edge_flap_noise(theta_flaps, flaps_span, flaps_area, wing_span, flaps_slot_number, M_0, c_0, theta, phi, frequency)
            
        if 'landing_gear' in noise_component_lst:
            msap += compute_landing_gear_noise(i_landing_gear, 
                                         main_gear_number, main_gear_tire_diameter, main_gear_wheel_number, main_gear_length, 
                                         nose_gear_number, nose_gear_tire_diameter, nose_gear_wheel_number, nose_gear_length, 
                                         wing_span, 
                                         M_0, c_0, theta, phi, frequency)
                                         
        if flag_high_speed_research_suppression:
            # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
            suppression = tables.get_high_speed_research_suppression(frequency, theta)
            msap *= suppression

    return msap / _P_REF ** 2

