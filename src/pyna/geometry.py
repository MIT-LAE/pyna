import numpy as np
from pyna.atmosphere import (
    speed_of_sound,
    temperature_at_altitude
)


def _compute_source_observer_vector(x, y, z, observer_position, source_z_offset):
    return np.array([
         observer_position[0] - x,
         observer_position[1] - y,
        -observer_position[2] + (z + source_z_offset)
    ])

def _compute_source_observer_distance(x, y, z, observer_position, source_z_offset=0.):
    """Compute distance between source and observer.

    Parameters
    ----------
    x : Q_
        x coordinate of source.
    y : Q_
        y coordinate of source.
    z : Q_
        z coordinate of source.
    observer_position : Q_[np.ndarray]
        [x, y, z] coordinates of observer.
    source_z_offset : float, optional
    
    Return
    ------
    Q_
        Distance between source and observer.
    """

    r = _compute_source_observer_vector(
        x, 
        y, 
        z, 
        observer_position, 
        source_z_offset
    )
    
    return np.linalg.norm(r, axis=0)

def _compute_elevation_angle(z, r):
    """Compute elevation angle.
    
    Parameters
    ----------
    z : Q_
    
    r : Q_

    Returns
    -------
    float : 
        Elevation angle
    """
        
    return np.arcsin(z / r)

def _compute_directivy_angles(x, y, z, observer_position, alpha, gamma, source_z_offset=0):
    """Compute polar directivity angle

    Parameters
    ----------
    x : float

    y : float

    z : float

    alpha : float

    gamma : float

    observer_position : np.ndarray

    source_z_offset : float, optional
    
    Return
    ------
    float
        Polar directivy angle [deg]
    float
        Azimuthal directivity angle [deg]
    """

    r = _compute_source_observer_vector(
        x,
        y,
        z,
        observer_position,
        source_z_offset
    )
    r_normalized = r / np.linalg.norm(r, axis=0)

    # Compute body angles (psi_body, theta_body, phi_body): angle of body w.r.t. horizontal
    theta_body = alpha + gamma
    phi_body = 0.
    psi_body = 0.

    # Transformation direction cosines (Euler angles) to the source coordinate system (i.e. take position of the aircraft into account)
    cth  = np.cos(theta_body)
    sth  = np.sin(theta_body)
    cphi = np.cos(phi_body)
    sphi = np.sin(phi_body)
    cpsi = np.cos(psi_body)
    spsi = np.sin(psi_body)

    n_vcr_s_0 = (
        cth * cpsi * r_normalized[0] + 
        cth * spsi * r_normalized[1] -
        sth * r_normalized[2]
    )
    n_vcr_s_1 = (
        (-spsi * cphi + sphi * sth * cpsi) * r_normalized[0] + 
        ( cphi * cpsi + sphi * sth * spsi) * r_normalized[1] + 
        sphi * cth * r_normalized[2]
    )
    n_vcr_s_2 = (
        ( spsi * sphi + cphi * sth * cpsi) * r_normalized[0] + 
        (-sphi * cpsi + cphi * sth * spsi) * r_normalized[1] + 
        cphi * cth * r_normalized[2]
    )

    return (
        np.arccos(n_vcr_s_0),
        np.arctan2(n_vcr_s_1, n_vcr_s_2)
    )

def _compute_average_speed_of_sound_source_observer(z, T_0, z_observer, n_layers_atmosphere=11):
    
    z_layers = np.linspace(z_observer, z, n_layers_atmosphere)
    T_layers = temperature_at_altitude(z_layers, z, T_0)
    c_layers = speed_of_sound(T_layers)
    return np.mean(c_layers)

def _compute_observer_time(t_source, distance_source_observer, average_speed_of_sound):
    """Compute observed time

    Parameters
    ----------
    t_source:
        Source time
    distance_source_observer : float
        Distance between source and observer
    average_speed_of_sound : float
        Average speed of sound between source and observer 
    
    Returns
    -------
    float : 
        Observer time.
    """
    # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 20
    return t_source + distance_source_observer / average_speed_of_sound

def compute_source_observer_distance(x, y, z, observer_position, source_z_offset=0.):
    vf = np.vectorize(_compute_source_observer_distance, excluded={"observer_position", "source_z_offset"})
    return vf(x=x, y=y, z=z, observer_position=observer_position, source_z_offset=source_z_offset)

def compute_elevation_angle(z, r):
    vf = np.vectorize(_compute_elevation_angle)
    return vf(z, r)

def compute_directivy_angles(x, y, z, observer_position, alpha, gamma, source_z_offset=0):
    vf = np.vectorize(_compute_directivy_angles, excluded={"observer_position", "source_z_offset"})
    return vf(x=x, y=y, z=z, observer_position=observer_position, alpha=alpha, gamma=gamma, source_z_offset=source_z_offset)

def compute_average_speed_of_sound_source_observer(z, T_0, z_observer, n_layers_atmosphere=11):
    vf = np.vectorize(_compute_average_speed_of_sound_source_observer, excluded={"z_observer", "n_layers_atmosphere"})
    return vf(z=z, T_0=T_0, z_observer=z_observer, n_layers_atmosphere=n_layers_atmosphere)

def compute_observer_time(t_source, distance_source_observer, average_speed_of_sound):
    vf = np.vectorize(_compute_observer_time) 
    return vf(t_source, distance_source_observer, average_speed_of_sound)
