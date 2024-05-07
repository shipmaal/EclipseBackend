from typing import Union
import warnings

from astropy.coordinates import SkyCoord, Latitude, Longitude
from astropy import units as u

import numpy as np
import pandas as pd
import test


def sky_coord_unit_converter(object: SkyCoord, unit: u.Unit):
    return SkyCoord(x=object.cartesian.x, 
                    y=object.cartesian.y, 
                    z=object.cartesian.z, 
                    unit=unit, 
                    representation_type='cartesian')


def lstsq_matrix(n):
    l = np.array(range(-2, 3)).T
    return np.column_stack([l**i for i in range(n)])


def calc_poly_bess_coefs(n, b):
    """
    Calculate the polynomial coefficients for the Besselian elements
    using the least squares method
    n: int, the degree of the polynomial
    b: list, the values of the Besselian elements
    """
    A = lstsq_matrix(n)
    return np.linalg.lstsq(A, b, rcond=None)[0].tolist()


def bowring(z, ρ, μ, i=3):
    """
    Bowring's method for solving the inverse problem of geodesy
    z: float, the geocentric latitude
    ρ: float, the geocentric distance
    μ: float, the longitude
    i: int, the number of iterations
    Taken from Matlab Documentation for the function geocentric2geodetic 
    """
    a = (6378.137 * u.km).to(u.earthRad).value
    b = (6356.752 * u.km).to(u.earthRad).value
    f = (a - b) / a
    e_2 = 1 - b**2 / a**2
    e_2_ = e_2 / (1 - e_2)

    ret = μ
    for _ in range(i):
        β = np.arctan((1-f)*np.sin(ret)/(np.cos(ret)))
        ret = np.arctan((z + b * e_2_ * (np.sin(β))**3) / (ρ - a * e_2 * (np.cos(β))**3))
    return ret


def poly(t, attribute):
        ans = 0
        for i in range(len(attribute)):
            ans += attribute[i] * t ** i
        return ans

def dec_to_hms(t, t0=0):
    decimal_hour = t0 + t
    hour = int(decimal_hour)
    minute = (decimal_hour - hour) * 60
    second = (minute - int(minute)) * 60
    if round(second) == 60:
        minute += 1
        second = 0

    if round(minute) == 60:
        hour += 1
        minute = 0

    return hour, int(minute), second


frank_path = test.frank_path()
frank_lats = Latitude(np.array(frank_path[0]) * u.deg)
frank_lons = Longitude(np.array(frank_path[1]) * u.deg, wrap_angle=180 * u.deg)
def output_data(λ, φ, i, t):
    frank_lat = frank_lats[i]
    frank_lon = frank_lons[i]

    hour, minute, second = dec_to_hms(t, 18)

    λ_direction = ' W' if λ.degree < 0 else ' E'
    φ_direction = ' S' if φ.degree < 0 else ' N'

    lon_direction = ' W' if frank_lon < 0 else ' E'
    lat_direction = ' S' if frank_lat < 0 else ' N'

    def format_dms(angle, direction):
        return np.abs(angle).to_string(unit='deg', sep=('°', "'", '"'), precision=2, pad=True) + direction

    λ_dms = format_dms(λ, λ_direction)
    φ_dms = format_dms(φ, φ_direction)
    lon_dms = format_dms(frank_lon, lon_direction)
    lat_dms = format_dms(frank_lat, lat_direction)

    print(f'Time(TDT): {hour}:{minute:02}:{second:04.1f}\tLongitude: {λ_dms}\tLatitude: {φ_dms}')
    # print(f'Time(TDT): {hour}:{int(minute):02}:{second:04.1f}\tFrank Longitude: {lon_dms}\tFrank Latitude: {lat_dms}')
    # print(f'Lon diff: {np.abs(λ.value - frank_lon.value)}\tLat diff: {np.abs(φ.value - frank_lat.value)}')


def output_data_for_df(t, lon1, lat1, lon2, lat2):
    hour, minute, second = dec_to_hms(t, 18)


    sep_tuple = ('°', "'", '"')
    def get_direction(angle: Union[Longitude, Latitude]):
        if isinstance(angle, Longitude):
            return 'W' if angle < 0 else 'E'
        return 'S' if angle < 0 else 'N'

    def format_dms(angle):
        return np.abs(angle).to_string(unit='deg', sep=sep_tuple, precision=2, pad=True) + f' {get_direction(angle)}'

    λ_dms = format_dms(lon1)
    φ_dms = format_dms(lat1)
    lon_dms = format_dms(lon2)
    lat_dms = format_dms(lat2)

    time_str = f'{hour}:{minute:02}:{second:04.1f}'
    lon_diff = np.abs(lon1 - lon2).value
    lat_diff = np.abs(lat1 - lat2).value

    return [time_str, λ_dms, φ_dms, lon_dms, lat_dms, lon_diff, lat_diff]   

def fund_to_geo(ξ, η, d, μ, i):
    """
    Convert fundamental plane coordinates to geocentric coordinates
    1. Calculate ξ, η1, ς1 from x and y given by equations 8.333-9 and 8.333-10
    2. Eq 8.333-13 gives φ1 and θ. 
    3. Eq 8.333-6 gives φ
    4. Eq 8.333-14 or 8.333-7 gives ς
    """
    ξ, η, d, μ = ξ[i], η[i], d[i], μ[i]
    a = (6378.137 * u.km).to(u.earthRad) # equatorial radius
    b = (6356.752 * u.km).to(u.earthRad) # polar radius
    e_2 = 1 - b**2 / a**2 # eccentricity squared

    ρ1 = np.sqrt(1 - e_2 * (np.cos(d)) ** 2) 
    ρ2 = np.sqrt(1 - e_2 * (np.sin(d)) ** 2) 

    sind1 = np.sin(d) / ρ1
    cosd1 = np.sqrt(1 - e_2) * np.cos(d) / ρ1

    sind1d2 = e_2 * np.sin(d) * np.cos(d) / (ρ1 * ρ2)
    cosd1d2 = np.sqrt(1 - e_2) / (ρ1 * ρ2)

    η1 = η / ρ1
    ς1 = np.sqrt((1 * u.earthRad) ** 2 - ξ ** 2 - η1 ** 2)
    ς = ρ2 * (ς1 * cosd1d2 - η1 * sind1d2)
    
    R1_d1 = np.array([
        [1, 0, 0],
        [0, cosd1, -sind1],
        [0, sind1, cosd1]
    ])

    fund = np.array([ξ.value, η1.value, ς1.value])
    cosφ1sinθ, sinφ1, cosφ1cosθ = R1_d1 @ fund
    
    θ = np.arctan(cosφ1sinθ / cosφ1cosθ)

    z = np.sqrt(1 - e_2) * sinφ1 / ς.value
    ρ  = cosφ1sinθ / (ς.value * np.sin(θ))

    φ = np.arcsin(sinφ1)
    φ = bowring(z, ρ, φ)

    λ = (θ * u.rad).to(u.deg) - μ
    φ = φ.to(u.deg)

    λ = Longitude(λ, wrap_angle=180 *u.deg)
    φ = Latitude(φ)

    return λ, φ