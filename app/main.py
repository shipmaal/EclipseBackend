from fastapi import FastAPI
import elp2000

from dataclasses import asdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from astropy.coordinates import Latitude, Longitude
from astropy.coordinates import SkyCoord, solar_system_ephemeris, get_sun, get_body
from astropy import units as u

from utils import fund_to_geo, output_data_for_df
from classes import BesselianElements, ModifiedTime
import test


app = FastAPI()
# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain('./cert.pem', keyfile='./key.pem')


@app.get("/")
async def root():
    # 18:00 UT April 8, 2024
    # 2460409.25
    # 2444239.5
    T0 = ModifiedTime(2300000.5, format='jd')
    moon = get_body('moon', T0, ephemeris='jpl')
    print(moon.cartesian.xyz.to(u.km))

    
    # franks attributes
    attributes = {
        'x': [-0.318157, 0.5117105, 0.0000326, -0.0000085],
        'y': [0.219747, 0.2709586, -0.0000594, -0.0000047],
        'd': [7.58620, 0.014844, -0.000002],
        'l1': [0.535813, 0.0000618, -0.0000128],
        'l2': [-0.010274, 0.0000615, -0.0000127],
        'μ': [89.59122, 15.004084],
        'tan_f1': [0.0046683],
        'tan_f2': [0.0046450]
    }
    tn = 1 + 52/60
    t_array = np.linspace(0, tn, int(30*tn+1))
    print(len(t_array))
    elements = BesselianElements(T0, 'sun', 'moon')
    calced_elements = elements.compute_elements(t_array)
    frank_elements = elements.compute_elements(t_array, attributes)

    
    lons = []
    lats = []
    frank_path = test.frank_path()
    frank_lats = Latitude(np.array(frank_path[0]) * u.deg)
    frank_lons = Longitude(np.array(frank_path[1]) * u.deg, wrap_angle=180 * u.deg)
    df = pd.DataFrame(columns=['Time(TDT)', 'Longitude', 'Latitude', 'Frank Longitude', 'Frank Latitude', 'Lon diff', 'Lat diff'])
    for i, t in enumerate(t_array):
        calced_lon_i, calced_lat_i = fund_to_geo(calced_elements.x, calced_elements.y, calced_elements.d, calced_elements.μ, i)
        frank_lon_i, frank_lat_i = fund_to_geo(frank_elements.x, frank_elements.y, frank_elements.d, frank_elements.μ, i)
        df.loc[i] = output_data_for_df(t, frank_lon_i, frank_lat_i, frank_lons[i], frank_lats[i])
        # df.loc[i] = output_data_for_df(t, calced_lon_i, calced_lat_i, frank_lon_i, frank_lat_i)
        # lons.append(frank_lon_i.value)
        # lats.append(frank_lat_i.value)
    
    # print(df)
    # plt.plot(df['Lon diff'])
    # plt.xlabel('Time')
    # plt.ylabel('Longitude Difference')
    # plt.savefig('lon_diff_plot.png')
    mapQ = False

    if mapQ:
        m = Basemap(width=12000000,height=9000000,projection='lcc',
                    resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
        m.drawcoastlines()
        m.drawmapboundary(fill_color='aqua') 
        m.fillcontinents(color='coral',lake_color='aqua')

        parallels = np.arange(0.,81,10.)
        # labels = [left,right,top,bottom]
        m.drawparallels(parallels,labels=[False,True,True,False])

        meridians = np.arange(10.,351.,20.)
        m.drawmeridians(meridians,labels=[True,False,False,True])
        m.scatter(lons, lats, latlon=True, s=0.5, c='green')

        plt.savefig('map.png')

    return {
        "message": "Besselian Elements",
        "elements": asdict(elements.elements),
        "poly_table": asdict(elements.poly_table)
    }

