from pyna.constants import (
    _R_SOURCE,
    _ACOUSTIC_IMPEDANCE_SEALEVEL
)

from pyna.frequency_bands import get_msap_subbands


# def compute_propagation(msap_source, flag_atmospheric_absorption, flag_ground_effects):

#     # Number of observers
#     n_obs = np.shape(settings['x_observer_array'])[0]

#     # Extract inputs
#     r = inputs['r']
#     x = inputs['x']
#     z = inputs['z']
#     c_bar = inputs['c_bar']
#     rho_0 = inputs['rho_0']
#     I_0 = inputs['I_0']
#     beta = inputs['beta']
#     msap_source = inputs['msap_source']
    

#     # Apply spherical spreading and characteristic impedance effects to the MSAP
#     # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 1
#     if settings['direct_propagation']:
#         msap_r = msap_source * (_R_SOURCE ** 2 / r[k, i] ** 2) * (_ACOUSTIC_IMPEDANCE_SEALEVEL = 409.74 / I_0[i])
#     else:
#         msap_r = msap_source

#     # Generate sub-banding
#     msap_prop_i = np.zeros(settings['n_frequency_bands'])
#     if flag_atmospheric_absorption or flag_ground_effects:

#         msap_sb = split_subbands(settings, msap_r)

#         # Initialize solution vectors
#         if flag_atmospheric_absorption:
#             # ---------- Apply atmospheric absorption on sub-bands ----------
#             # Compute average absorption factor between observer and source
#             alpha_f = data.abs_f(data.f_sb, z[i])

#             # Compute absorption (convert dB to Np: 1dB is 0.115Np)
#             # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 14
#             msap_sb = msap_sb * np.exp(-2 * 0.115 * alpha_f * (r[k, i] - _R_SOURCE))

#         # ---------- Apply ground effects on sub-bands ----------
#         if flag_ground_effects:
#             # Ground reflection factor
#             G = ground_effects(settings, data, r[k, i], beta[k, i], settings['x_observer_array'][k, :], c_bar[k, i], rho_0[i])
                
#             # Apply ground effects
#             msap_sb = msap_sb * G

#         # Compute absorbed msap by adding up the msap at all the sub-band frequencies
#         # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 22

#         for j in np.arange(settings['n_frequency_bands']):
#             msap_prop_i[j] = np.sum(msap_sb[j*settings['n_frequency_subbands']:(j+1)*settings['n_frequency_subbands']])

#     else:
#         msap_prop_i = msap_r

#     outputs['msap_prop'][k, i, :] = msap_prop_i.clip(min=1e-99)