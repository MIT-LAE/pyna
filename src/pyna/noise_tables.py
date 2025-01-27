import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class NoiseTables:
    """Class containing noise model data tables
    """
    def __init__(self) -> None:
        
        # Fan noise tables
        with open('tables/source_fan.json', 'r') as file:
            data = json.load(file)
        self.fan = {key : np.array(data[key]) for key in data.keys()}

        # Core noise tables
        with open('tables/source_core.json', 'r') as file:
            data = json.load(file)
        self.core = {key : np.array(data[key]) for key in data.keys()}

        # Jet noise tables
        with open('tables/source_jet_mixing.json', 'r') as file:
            data = json.load(file)
        self.jet_mixing = {key : np.array(data[key]) for key in data.keys()}


class JetMixingNoiseTables(NoiseTables):

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
        
        return np.interp(log10_V_j_star, self.jet_mixing['density_exponent_x_0'], self.jet_mixing['density_exponent_y'])

    def _get_power_deviation_factor(self, log10_V_j_star):

        return np.interp(log10_V_j_star, 
                         self.jet_mixing['power_deviation_factor_x_0'], 
                         self.jet_mixing['power_deviation_factor_y'])

    def _get_directivity(self, theta, log10_V_j_star):

        f_interp = RegularGridInterpolator(
            (self.jet_mixing['directivity_x_0'], self.jet_mixing['directivity_x_1']), 
            self.jet_mixing['directivity_y']
            )

        return  f_interp((theta, log10_V_j_star), method="linear")

    def _get_strouhal_correction(self, V_j_star, theta):
        
        f_interp = RegularGridInterpolator(
            (self.jet_mixing['strouhal_correction_x_0'], self.jet_mixing['strouhal_correction_x_1']), 
            self.jet_mixing['strouhal_correction_y']
            )

        return f_interp((V_j_star, theta), method="linear")
	
    def _get_forward_velocity_index(self, theta):

        return np.interp(theta, 
                         self.jet_mixing['forward_velocity_index_x_0'],
                         self.jet_mixing['forward_velocity_index_y'])
        
    def _get_spectral_distribution(self, theta, Tt_j_star, log10_V_j_star, log10_St, n_frequency_bands):
        
        f_interp = RegularGridInterpolator(
            (
                self.jet_mixing['spectral_distribution_x_0'], 
                self.jet_mixing['spectral_distribution_x_1'], 
                self.jet_mixing['spectral_distribution_x_2'], 
                self.jet_mixing['spectral_distribution_x_3'],
            ), 
            self.jet_mixing['spectral_distribution_y'], 
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
	
	