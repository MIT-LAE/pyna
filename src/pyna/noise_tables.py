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

        if method not in ["kresja", "others"]:
            raise ValueError(f"Method {method} not available for get_inlet_broadband_directivity.")

        return self._get_inlet_broadband_directivity(theta, method)
    
    def get_discharge_broadband_directivity(self, theta, method):

        if method not in ["alliedsignal", "kresja", "others"]:
            raise ValueError(f"Method {method} not available for get_discharge_broadband_directivity.")

        return self._get_discharge_broadband_directivity(theta, method)
    
    def get_inlet_tones_directivity(self, theta, method):

        if method not in ["alliedsignal", "kresja", "others"]:
            raise ValueError(f"Method {method} not available for get_inlet_tones_directivity.")

        return self._get_inlet_tones_directivity(theta, method)
            
    def get_discharge_tones_directivity(self, theta, method):

        if method not in ["alliedsignal", "kresja", "others"]:
            raise ValueError(f"Method {method} not available for get_discharge_tones_directivity.")

        return self._get_discharge_tones_directivity(theta, method)

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


class CoreNoiseTables:

    def __init__(self):
        
        # Core noise tables
        with open('tables/source_core.json', 'r') as file:
            data = json.load(file)
        self.data = {key : np.array(data[key]) for key in data.keys()}

    def get_directivity(self, theta):
        
        if theta < 0. or theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")
        
        return self._get_directivity(theta)
    
    def get_spectral_distribution(self, log10_f_fp):

        if log10_f_fp < -1.1 or log10_f_fp > 1.6:
            raise ValueError(f"log10_f_fp = {log10_f_fp} is outside the domain of the noise tables [-1.1, 1.6].")

        return self._get_spectral_distribution(log10_f_fp)

    def _get_directivity(self, theta):
        return np.interp(theta, self.data['directivity_x_0'], self.data['directivity_y'])

    def _get_spectral_distribution(self, log10_f_fp):
        return np.interp(log10_f_fp, self.data['spectral_distribution_x_0'], self.data['spectral_distribution_y'])


class JetMixingNoiseTables:

    def __init__(self):

        # Jet noise tables
        with open('tables/source_jet_mixing.json', 'r') as file:
            data = json.load(file)
        self.data = {key : np.array(data[key]) for key in data.keys()}


    def get_density_exponent(self, log10_V_j_star):
        
        if log10_V_j_star < -0.45 or log10_V_j_star > 0.25:
            raise ValueError(f"log10_V_j_star = {log10_V_j_star} is outside the domain of the noise tables [-0.45, 0.25].")

        return self._get_density_exponent(log10_V_j_star)

    def get_power_deviation_factor(self, log10_V_j_star):

        if log10_V_j_star < -0.4 or log10_V_j_star > 0.4:
            raise ValueError(f"log10_V_j_star = {log10_V_j_star} is outside the domain of the noise tables [-0.4, 0.4].")
        
        return self._get_power_deviation_factor(log10_V_j_star)

    def get_directivity(self, theta, log10_V_j_star):

        if log10_V_j_star < -0.4 or log10_V_j_star > 0.4:
            raise ValueError(f"log10_V_j_star = {log10_V_j_star} is outside the domain of the noise tables [-0.4, 0.4].")

        if theta < 0. or theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")

        return self._get_directivity(theta, log10_V_j_star)

    def get_strouhal_correction(self, V_j_star, theta):

        if V_j_star < 0 or V_j_star > 2.5:
            raise ValueError(f"V_j_star = {V_j_star} is outside the domain of the noise tables [0, 2.5].")

        if theta < 0. or theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")

        return self._get_strouhal_correction(V_j_star, theta)

    def get_forward_velocity_index(self, theta):

        if theta < 0. and theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")
        
        return self._get_forward_velocity_index(theta)
    
    def get_spectral_distribution(self, theta, Tt_j_star, log10_V_j_star, log10_St, n_frequency_bands):

        if theta < 0. and theta > 180.:
            raise ValueError(f"theta = {theta} is outside the domain of the noise tables [0., 180.].")
        
        if log10_V_j_star < -0.4 and log10_V_j_star > 0.4:
            raise ValueError(f"log10_V_j_star = {log10_V_j_star} is outside the domain of the noise tables [-0.4, 0.4].")

        if Tt_j_star < 0. and Tt_j_star > 7.:
            raise ValueError(f"Tt_j_star = {Tt_j_star} is outside the domain of the noise tables [0., 7.].")

        if log10_St < -2 and log10_St > 2.5:
            raise ValueError(f"log10_St = {log10_St} is outside of the noise tables [-2., 2.5].")

    	# Source: Zorumski report 1982 part 2. Chapter 8.4 Table VI
        if 1 <= Tt_j_star <= 3.5:
            return self._get_spectral_distribution(theta, Tt_j_star, log10_V_j_star, log10_St, n_frequency_bands)

    	# Add linear extrapolation for jet temperature
    	# Computational fix for data unavailability
        elif Tt_j_star > 3.5:
            mlog10F_a = self._get_spectral_distribution(theta, 3.5, log10_V_j_star, log10_St, n_frequency_bands)
            mlog10F_b = self._get_spectral_distribution(theta, 3.4, log10_V_j_star, log10_St, n_frequency_bands)

            return (mlog10F_a - mlog10F_b) / 0.1 * (Tt_j_star - 3.5) + mlog10F_a
        
        else:
            mlog10F_a = self._get_spectral_distribution(theta, 1.1, log10_V_j_star, log10_St)
            mlog10F_b = self._get_spectral_distribution(theta, 1.0, log10_V_j_star, log10_St)
            
            return (mlog10F_a - mlog10F_b) / 0.1 * (Tt_j_star[i] - 1.0) + mlog10F_b
            

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
	
	