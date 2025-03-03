import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class FanNoiseTables:
    
    def __init__(self):
        
        # Fan noise tables
        with open('tables/source_fan.json', 'r') as file:
            data = json.load(file)
        self.data = {key : np.array(data[key]) for key in data.keys()}

    def get_inlet_broadband_directivity(self, theta, method):

        if method not in ["original", "alliedsignal", "geae", "kresja"]:
            raise ValueError(f"Method {method} not available for get_inlet_broadband_directivity.")

        return self._get_inlet_broadband_directivity(theta, method if method == "kresja" else "others")
    
    def get_discharge_broadband_directivity(self, theta, method):

        if method not in ["original", "alliedsignal", "geae", "kresja"]:
            raise ValueError(f"Method {method} not available for get_discharge_broadband_directivity.")

        return self._get_discharge_broadband_directivity(theta, method if method in ["alliedsignal", "kresja"] else "others")
    
    def get_inlet_tones_directivity(self, theta, method):

        if method not in ["original", "alliedsignal", "geae", "kresja"]:
            raise ValueError(f"Method {method} not available for get_inlet_tones_directivity.")

        return self._get_inlet_tones_directivity(theta, method if method in ["alliedsignal", "kresja"] else "others")
            
    def get_discharge_tones_directivity(self, theta, method):

        if method not in ["original", "alliedsignal", "geae", "kresja"]:
            raise ValueError(f"Method {method} not available for get_discharge_tones_directivity.")

        return self._get_discharge_tones_directivity(theta, method if method in ["alliedsignal", "kresja"] else "others")

    def get_combination_tones_directivity(self, theta, method):

        if method not in ["original", "alliedsignal", "geae", "kresja"]:
            raise ValueError(f"Method {method} not available for get_combination_tones_directivity.")

        return self._get_combination_tones_directivity(theta, method)

    def get_combination_tones_tipmach(self, M_tip, subharmonic, method):
        
        if method not in ["original", "alliedsignal", "geae", "kresja"]:
            raise ValueError(f"Method {method} not available for get_combination_tones_tipmach.")

        return self._get_combination_tones_tipmach(M_tip, subharmonic, method)

    def get_combination_tones_spectral_distribution(self, f_bpf, subharmonic, method):
        
        if method not in ["original", "alliedsignal", "geae", "kresja"]:
            raise ValueError(f"Method {method} not available for get_combination_tones_spectral_distribution.")

        return self._get_combination_tones_spectral_distribution(f_bpf, subharmonic, method)

    def get_cleanup_turbulent_control_structures(self, method, flight_segment, i_harmonic, theta):

        if method not in ["original", "alliedsignal", "geae", "kresja"]:
            raise ValueError(f"Method {method} not available for get_cleanup_turbulent_control_structures.")
        
        return self._get_cleanup_turbulent_control_structures(self, method, flight_segment, i_harmonic)

    @staticmethod
    def get_filter_constants(filter_bandwidth):

        """
        Parameters
        ----------
        filter_bandwidth : float
        
        Returns
        -------
        float :
        float :
        float :
        float :
        """

        f_1 = 0.78250188 + 0.10874906 * filter_bandwidth
        f_2 = 1 - 0.10874906 * filter_bandwidth
        f_3 = 1 + 0.12201845 * filter_bandwidth
        f_4 = 1.2440369 - 0.12201845 * filter_bandwidth

        return f_1, f_2, f_3, f_4

    def get_liner_suppression(self, frequency,  theta, noise_direction):

        if noise_direction not in ["inlet", "discharge"]:
            raise ValueError(f"Noise direction {noise_direction} not available for get_liner_suppression.")

        return self._get_liner_suppression(frequency,  theta, noise_direction)

    def _get_inlet_broadband_directivity(self, theta, method):
        return np.interp(
            theta,
            self.data[f"inlet_broadband_directivity_{method}_x_0"],
            self.data[f"inlet_broadband_directivity_{method}_y"]
            )
    
    def _get_discharge_broadband_directivity(self, theta, method):
        return np.interp(
            theta,
            self.data[f"discharge_broadband_directivity_{method}_x_0"],
            self.data[f"discharge_broadband_directivity_{method}_y"]
        )
    
    def _get_inlet_tones_directivity(self, theta, method):
        return np.interp(
            theta,
            self.data[f"inlet_tones_directivity_{method}_x_0"],
            self.data[f"inlet_tones_directivity_{method}_y"]
        )
    
    def _get_discharge_tones_directivity(self, theta, method):
        return np.interp(
            theta,
            self.data[f"discharge_tones_directivity_{method}_x_0"],
            self.data[f"discharge_tones_directivity_{method}_y"]
        )

    def _get_combination_tones_directivity(self, theta, method):
        return np.interp(
            theta,
            self.data[f"combination_tones_directivity_{method}_x_0"],
            self.data[f"combination_tones_directivity_{method}_y"]
        )

    def _get_combination_tones_tipmach(self, M_tip, subharmonic, method):
        
        if M_tip < self.data[f"combination_tones_tipmach_{method}"][subharmonic-1]:
            return self.data[f"combination_tones_tipmach_slope1_{method}"][subharmonic-1] * np.log10(M_tip) + self.data[f"combination_tones_tipmach_offset1_{method}"][subharmonic-1]
        else:
            return self.data[f"combination_tones_tipmach_slope2_{method}"][subharmonic-1] * np.log10(M_tip) + self.data[f"combination_tones_tipmach_offset2_{method}"][subharmonic-1]

    def _get_combination_tones_spectral_distribution(self, f_bpf, subharmonic, method):
        
        # For frequencies less than the subharmonic:
        if f_bpf <= 1 / 2**subharmonic:
            return self.data[f"combination_tones_spectral_offset4_{method}"][subharmonic-1] * np.log10(f_bpf) + self.data[f"combination_tones_spectral_offset4_{method}"][subharmonic-1]
        
        # For frequencies greater than the subharmonic:
        else:
            return self.data[f"combination_tones_spectral_offset3_{method}"][subharmonic-1] * np.log10(f_bpf) + self.data[f"combination_tones_spectral_offset3_{method}"][subharmonic-1]

    def _get_cleanup_turbulent_control_structures(self, method, flight_segment, i_harmonic, theta):

        if method == "geae":
            # Compute suppression factors for GE#s "Flight cleanup Turbulent Control Structure."
            # Approach or takeoff values to be applied to inlet discrete interaction tones
            # at bpf and 2bpf.  Accounts for observed in-flight tendencies.

            match flight_segment:
                case "takeoff":
                    match i_harmonic:
                        case 1:
                            turbulent_control_structures_term = np.interp(theta, self.data["geae_cleanup_turbulent_control_structures_x_0"], self.data["geae_cleanup_turbulent_control_structures_takeoff_harmonic_1_y"])
                        case 2:
                            turbulent_control_structures_term = np.interp(theta, self.data["geae_cleanup_turbulent_control_structures_x_0"], self.data["geae_cleanup_turbulent_control_structures_takeoff_harmonic_2_y"])
                        case _:
                            turbulent_control_structures_term = 0
                case 'approach':
                    match i_harmonic:    
                        case 1:
                            turbulent_control_structures_term = np.interp(theta, self.data["geae_cleanup_turbulent_control_structures_x_0"], self.data["geae_cleanup_turbulent_control_structures_approach_harmonic_1_y"])
                        case 2:
                            turbulent_control_structures_term = np.interp(theta, self.data["geae_cleanup_turbulent_control_structures_x_0"], self.data["geae_cleanup_turbulent_control_structures_approach_harmonic_2_y"])
                        case _:
                            turbulent_control_structures_term = 0
                case _:
                    turbulent_control_structures_term = 0
        
        else:
            turbulent_control_structures_term = 0

        return turbulent_control_structures_term


    def _get_liner_suppression(self, frequency,  theta, noise_direction):

        f_interp = RegularGridInterpolator(
            (
                self.data[f'{noise_direction}_liner_suppression_x_0'], 
                self.data[f'{noise_direction}_liner_suppression_x_1']), 
                self.data[f'{noise_direction}_liner_suppression_y']
            )

        return  f_interp((frequency,  theta), method="linear")


class CoreNoiseTables:

    def __init__(self):
        """ Load core noise tables. """

        with open('tables/source_core.json', 'r') as file:
            data = json.load(file)
        self.data = {key : np.array(data[key]) for key in data.keys()}

    def get_directivity(self, theta):
        """
        Parameters
        ----------
        theta : float
        
        Returns 
        -------
        theta : directivity level (log10_D)
        """

        if theta < 0. or theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")
        
        return self._get_directivity(theta)
    
    def get_spectral_distribution(self, log10_f_fp):
        """
        Parameters
        ----------
        log10_f_fp : np.ndarray

        Returns
        -------
        np.ndarray : 
            combustion spectral level (log10_S) 
        """
        if np.any(log10_f_fp < -1.1) or np.any(log10_f_fp > 1.6):
            raise ValueError(f"log10_f_fp = {log10_f_fp} is outside the domain of the noise tables [-1.1, 1.6].")

        return self._get_spectral_distribution(log10_f_fp)

    def _get_directivity(self, theta):
        return np.interp(theta, self.data['directivity_x_0'], self.data['directivity_y'])

    def _get_spectral_distribution(self, log10_f_fp):
        return np.interp(log10_f_fp, self.data['spectral_distribution_x_0'], self.data['spectral_distribution_y'])


class JetMixingNoiseTables:

    def __init__(self):
        """Load jet mixing noise tables."""
        
        with open('tables/source_jet_mixing.json', 'r') as file:
            data = json.load(file)
        self.data = {key : np.array(data[key]) for key in data.keys()}

    def get_density_exponent(self, log10_V_j_star):
        """
        Parameters
        ----------
        log10_V_j_star : float

        Returns
        -------
        np.float : 
            density exponent (omega)
        
        """
        
        if log10_V_j_star < -0.45 or log10_V_j_star > 0.25:
            raise ValueError(f"log10_V_j_star = {log10_V_j_star} is outside the domain of the noise tables [-0.45, 0.25].")

        return self._get_density_exponent(log10_V_j_star)

    def get_power_deviation_factor(self, log10_V_j_star):
        """
        Parameters
        ----------
        log10_V_j_star : float

        Returns
        -------
        np.float : 
            power deviation factor (log10_P)
        """
        if log10_V_j_star < -0.4 or log10_V_j_star > 0.4:
            raise ValueError(f"log10_V_j_star = {log10_V_j_star} is outside the domain of the noise tables [-0.4, 0.4].")
        
        return self._get_power_deviation_factor(log10_V_j_star)

    def get_directivity(self, theta, log10_V_j_star):
        """
        Parameters
        ----------
        theta : float

        log10_V_j_star : float

        Returns
        -------
        np.float : 
            polar directivity level (log10_D) 
        """

        if log10_V_j_star < -0.4 or log10_V_j_star > 0.4:
            raise ValueError(f"log10_V_j_star = {log10_V_j_star} is outside the domain of the noise tables [-0.4, 0.4].")

        if theta < 0. or theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")

        return self._get_directivity(theta, log10_V_j_star)

    def get_strouhal_correction(self, V_j_star, theta):
        """
        Parameters
        ----------
        V_j_star : float

        theta : float

        Returns
        -------
        np.float : 
            strouhal number correction factor (xi)
        """    

        if V_j_star < 0 or V_j_star > 2.5:
            raise ValueError(f"V_j_star = {V_j_star} is outside the domain of the noise tables [0, 2.5].")

        if theta < 0. or theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")

        return self._get_strouhal_correction(V_j_star, theta)

    def get_forward_velocity_index(self, theta):
        """
        Parameters
        ----------
        theta : float

        Returns
        -------
        np.float : 
            forward velocity index (m)
        """

        if theta < 0. and theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")
        
        return self._get_forward_velocity_index(theta)
    
    def get_spectral_distribution(self, theta, Tt_j_star, log10_V_j_star, log10_St, n_frequency_bands):
        """
        Parameters
        ----------
        theta : float
        
        Tt_j_star : float
        
        log10_V_j_star : float 
        
        log10_St : np.ndarray
        
        n_frequency_bands : int

        Returns
        -------
        np.float : 
            nomralized spectral distribution level (-10*log10_F)
        """
        if theta < 0. or theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")
        
        if log10_V_j_star < -0.4 or log10_V_j_star > 0.4:
            raise ValueError(f"log10_V_j_star = {log10_V_j_star} is outside the domain of the noise tables [-0.4, 0.4].")

        if Tt_j_star < 0. or Tt_j_star > 7.:
            raise ValueError(f"Tt_j_star = {Tt_j_star} is outside the domain of the noise tables [0., 7.].")

        if np.any(log10_St < -2) or np.any(log10_St > 2.5):
            raise ValueError(f"log10_St = {log10_St} is outside of the noise tables [-2., 2.5].")

    	# Source: Zorumski report 1982 part 2. Chapter 8.4 Table VI
        if 1 <= Tt_j_star <= 3.5:
            return self._get_spectral_distribution(theta, Tt_j_star, log10_V_j_star, log10_St, n_frequency_bands)

    	# Add linear extrapolation for jet temperature
    	# Computational fix for data unavailability
        elif Tt_j_star > 3.5:
            minus_log10F_a = self._get_spectral_distribution(theta, 3.5, log10_V_j_star, log10_St, n_frequency_bands)
            minus_log10F_b = self._get_spectral_distribution(theta, 3.4, log10_V_j_star, log10_St, n_frequency_bands)

            return (minus_log10F_a - minus_log10F_b) / 0.1 * (Tt_j_star - 3.5) + minus_log10F_a
        
        else:
            minus_log10F_a = self._get_spectral_distribution(theta, 1.1, log10_V_j_star, log10_St)
            minus_log10F_b = self._get_spectral_distribution(theta, 1.0, log10_V_j_star, log10_St)
            
            return (minus_log10F_a - minus_log10F_b) / 0.1 * (Tt_j_star - 1.0) + minus_log10F_b
            

    def _get_density_exponent(self, log10_V_j_star):
        
        return np.interp(log10_V_j_star, self.data['density_exponent_x_0'], self.data['density_exponent_y'])

    def _get_power_deviation_factor(self, log10_V_j_star):

        return np.interp(log10_V_j_star, 
                         self.data['power_deviation_factor_x_0'], 
                         self.data['power_deviation_factor_y'])

    def _get_directivity(self, theta, log10_V_j_star):

        f_interp = RegularGridInterpolator(
            (self.data['directivity_x_0'], self.data['directivity_x_1']), 
            self.data['directivity_y']
            )

        return  f_interp((theta, log10_V_j_star), method="linear")

    def _get_strouhal_correction(self, V_j_star, theta):
        
        f_interp = RegularGridInterpolator(
            (self.data['strouhal_correction_x_0'], self.data['strouhal_correction_x_1']), 
            self.data['strouhal_correction_y']
            )

        return f_interp((V_j_star, theta), method="linear")
	
    def _get_forward_velocity_index(self, theta):

        return np.interp(theta, 
                         self.data['forward_velocity_index_x_0'],
                         self.data['forward_velocity_index_y'])
        
    def _get_spectral_distribution(self, theta, Tt_j_star, log10_V_j_star, log10_St, n_frequency_bands):
        
        f_interp = RegularGridInterpolator(
            (
                self.data['spectral_distribution_x_0'], 
                self.data['spectral_distribution_x_1'], 
                self.data['spectral_distribution_x_2'], 
                self.data['spectral_distribution_x_3'],
            ), 
            self.data['spectral_distribution_y'], 
            )
        
        return f_interp(
            (
                theta*np.ones(n_frequency_bands,),
                Tt_j_star*np.ones(n_frequency_bands), 
                log10_V_j_star*np.ones(n_frequency_bands), 
                log10_St
            ),
            method="linear"
            )
	

class JetShockNoiseTables:
    
    def __init__(self):
        """Load jet shock noise tables """

        with open('tables/source_jet_shock.json', 'r') as file:
            data = json.load(file)
        self.data = {key : np.array(data[key]) for key in data.keys()}

    def get_correlation_coefficient_spectrum(self, log10_sigma):
        """ Source: Zorumski report 1982 part 2. Chapter 8.5 Table III (+ cut-off at 0 for log10sigma > 2) 
        
        Parameters
        ----------
        log10_sigma : np.ndarray

        Returns
        -------
        np.ndarray : 
            correlation coefficient spectrum (C)
        """
        if np.any(log10_sigma < -0.7) or np.any(log10_sigma > 2.5):
            raise ValueError(f"log10_sigma = {log10_sigma} is outside the domain of the noise tables [-0.7, 2.5].")
        
        return self._get_correlation_coefficient_spectrum(log10_sigma)
    
    def get_group_source_strength_spectrum(self, log10_sigma):    	
        """Source: Zorumski report 1982 part 2. Chapter 8.5 Table III (+ linear extrapolation in logspace for log10sigma < 0; as given in SAEARP876)
        
        Parameters
        ----------
        log10_sigma : np.ndarray

        Returns
        -------
        np.ndarray : 
            group source strength (log10_H)

        """
        if np.any(log10_sigma < -0.8) or np.any(log10_sigma > 2.5):
            raise ValueError(f"log10_sigma = {log10_sigma} is outside the domain of the noise tables [0.0, 2.5].")

        return self._get_group_source_strength_spectrum(log10_sigma)
    
    def _get_correlation_coefficient_spectrum(self, log10_sigma):
        return np.interp(
            log10_sigma, 
            self.data['correlation_coeficient_spectrum_x_0'], 
            self.data['correlation_coeficient_spectrum_y']
        )
    
    def _get_group_source_strength_spectrum(self, log10_sigma):
        return np.interp(
            log10_sigma, 
            self.data['group_source_strength_spectrum_x_0'], 
            self.data['group_source_strength_spectrum_y']
        )
    

class AirframeNoiseTables:
    def __init__(self):
        """Load jet shock noise tables """

        with open('tables/source_airframe.json', 'r') as file:
            data = json.load(file)
        self.data = {key : np.array(data[key]) for key in data.keys()}

    def get_high_speed_research_suppression(self, frequency,  theta):
        return self._get_high_speed_research_suppression(frequency,  theta)
    
    def _get_high_speed_research_suppression(self, frequency,  theta):

        f_interp = RegularGridInterpolator(
            (
                self.data[f'high_speed_research_suppression_x_0'], 
                self.data[f'high_speed_research_suppression_x_1']), 
                self.data[f'high_speed_research_suppression_y']
            )

        return  f_interp((frequency,  theta), method="linear")
