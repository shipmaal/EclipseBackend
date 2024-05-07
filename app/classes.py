from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Union, Optional
from collections import namedtuple

from astropy.coordinates import SkyCoord, solar_system_ephemeris, get_sun, get_body
from astropy import units as u
from astropy.units import Quantity
from astropy.time import Time

from utils import *


@dataclass
class ElementStorage:
    x: list
    y: list
    d: list
    l1: list
    l2: list
    mu: list
    tan_f1: list
    tan_f2: list

Quantities = namedtuple('Quantities', ['x', 'y', 'd', 'l1', 'l2', 'μ', 'tan_f1', 'tan_f2'])


class ModifiedTime(Time):
    """
    A subclass of astropy.time.Time that adds a decimal hour attribute
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        time = datetime.strptime(self.isot, "%Y-%m-%dT%H:%M:%S.%f")
        self.decimal_hour = time.hour + time.minute / 60 + time.second / 3600


class BesselianElements:
    def __init__(self, T0, orb1, orb2):
        self.times = Time([T0 + i * u.hour for i in np.arange(-2, 3, 1)])
        with solar_system_ephemeris.set("jpl"):
            if orb1 == 'sun':
                self.orb1 = get_sun(self.times)
            else:
                self.orb1 = get_body(orb1, self.times)
            self.orb2 = get_body(orb2, self.times)
            self.elements = self.calc_elements(self)
            self.poly_table = self.calc_poly_table(self)
    
    @staticmethod
    def calc_elements(self):
        """
        Calculate the Besselian elements for the given times
        All angles are in radians for calculations but are converted to degrees for storage
        All distances are in earth radii
        
        """
        alpha, delta, r = self.orb2.ra.rad * u.rad, self.orb2.dec.rad * u.rad, self.orb2.distance.to(u.earthRad)
        S = sky_coord_unit_converter(self.orb1, u.earthRad).cartesian
        M = sky_coord_unit_converter(self.orb2, u.earthRad).cartesian
        G = S - M
        
        G_distance = G.norm()

        a = SkyCoord(G.x, G.y, G.z, representation_type='cartesian').spherical.lon.to(u.rad)
        d = SkyCoord(G.x, G.y, G.z, representation_type='cartesian').spherical.lat.to(u.rad)

        GST = self.times.sidereal_time("apparent", "greenwich")
        mu = (GST - a).to(u.deg)

        # 8.322-6
        x = r * np.cos(delta) * np.sin(alpha - a)
        y = r * (np.sin(delta) * np.cos(d) - np.cos(delta) * np.sin(d) * np.cos(alpha - a))
        z = r * (np.sin(delta) * np.sin(d) + np.cos(delta) * np.cos(d) * np.cos(alpha - a))

        d_s = 6.957e8 / 6.3781e6
        k = 0.2725076

        # 8.323-1
        sin_f1 = (d_s + k) / G_distance
        sin_f2 = (d_s - k) / G_distance

        # 8.323-6
        c1 = z + k/sin_f1
        c2 = z - k/sin_f2

        tan_f1 = [np.tan(np.arcsin(sin_f1i)) for sin_f1i in sin_f1.value]
        tan_f2 = [np.tan(np.arcsin(sin_f2i)) for sin_f2i in sin_f2.value]

        # 8.323-7
        l1 = c1 * tan_f1
        l2 = c2 * tan_f2

        return ElementStorage(x.value.tolist(), 
                              y.value.tolist(),
                              d.to(u.deg).value.tolist(),
                              l1.value.tolist(),
                              l2.value.tolist(),
                              mu.value.tolist(),
                              tan_f1, tan_f2)    
    
    @staticmethod
    def calc_poly_table(self):
        elements = self.elements
        return ElementStorage(
            calc_poly_bess_coefs(4, elements.x),
            calc_poly_bess_coefs(4, elements.y),
            calc_poly_bess_coefs(3, elements.d),
            calc_poly_bess_coefs(3, elements.l1),
            calc_poly_bess_coefs(3, elements.l2),
            calc_poly_bess_coefs(2, elements.mu),
            [np.average(elements.tan_f1)],
            [np.average(elements.tan_f2)]
        )
    
    ElementType = Optional[Union[ElementStorage, dict[str, any]]]
    

    def compute_elements(self, 
                         t_array: np.ndarray, 
                         attributes: ElementType = None) -> Quantities:
        if attributes is None:
            attributes = self.poly_table

        if isinstance(attributes, ElementStorage):
            attributes = asdict(attributes)

        polys = {name: (lambda t, attr=attr: poly(t, attr)) for name, attr in attributes.items()}
        values = {name: [func(i) for i in t_array] for name, func in polys.items()}
        x, y, d, l1, l2, μ, tan_f1, tan_f2 = values.values()

        x, y = x * u.earthRad, y * u.earthRad
        l1, l2 = l1 * u.earthRad, l2 * u.earthRad
        d, μ = d * u.deg, μ * u.deg
        tan_f1, tan_f2 = tan_f1 * u.dimensionless_unscaled, tan_f2 * u.dimensionless_unscaled

        return Quantities(x, y, d, l1, l2, μ, tan_f1, tan_f2)
    