import elp82b
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord, get_body
from astropy import units as u
from astropy.time import Time

cp, _ = elp82b.elp82b(2460409.251, 1e-12, 10)

print(f'cp norm: {np.linalg.norm(cp)}')
x, y, z = cp  # kilometers
print(f'x: {x}, y: {y}, z: {z}')

# Define the coordinates with proper units
cartesian_position = CartesianRepresentation(x * u.km, y * u.km, z * u.km)

# Use SkyCoord to handle the coordinate transformation
# Assume these coordinates are relative to the Earth-centered inertial frame (GCRS)
sky_coord = SkyCoord(cartesian_position, frame='icrs', obstime=Time(2460409.25, format="jd", scale="tdb"))
print(sky_coord.cartesian)
moon = get_body('moon', Time(2460409.25, format="jd", scale="tdb"), ephemeris='jpl')
print(f'Cartesian Moon: {moon.cartesian}')
print(f'moon norm: {np.linalg.norm(moon.cartesian.xyz.to(u.km))}')









