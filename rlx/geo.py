import numpy as np
from bokeh import layouts as bl
from bokeh import models as bm
from bokeh import plotting as bp
import pandas as pd
import bokeh
import shapefile

try:
    from shapely import geometry as sg
except OSError as err:
    print "WARN! could not load shapely library"
    print err

def latlon_to_meters(lats, lons):
    origin_shift = 2 * np.pi * 6378137 / 2.0
    mlon = lons * origin_shift / 180.0
    mlat = np.log(np.tan((90 + lats) * np.pi / 360.0)) / (np.pi / 180.0)
    mlat = mlat * origin_shift / 180.0
    return mlat, mlon

def meters_to_latlon(mlat, mlon):
    origin_shift = (np.pi * 6378137 )
    lons = mlon * 180./origin_shift
    mlat = mlat * 180./origin_shift
    lats = np.arctan(np.exp(mlat*np.pi/180. ) ) * 360. / np.pi - 90
    return lats, lons

def latlon_to_wgs(lat, lon):
    lat_degs, lat_mins, lat_secs = degs_to_wgs(np.abs(lat))
    lon_degs, lon_mins, lon_secs = degs_to_wgs(np.abs(lon))

    rlat = "%02d%02d%02d"%(np.round(lat_degs), np.round(lat_mins), np.round(lat_secs))
    rlon = "%03d%02d%02d"%(np.round(lon_degs), np.round(lon_mins), np.round(lon_secs))

    rlat += "N" if lat>0 else "S"
    rlon += "E" if lon>0 else "W"

    return rlat, rlon


def degs_to_wgs(k):
    degs = int(k)
    r = k-degs
    mins = int(r*60)
    r = r*60 - mins
    secs = r*60
    return degs, mins, secs


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def base_map(tools=None, axis_visible=True, **kwargs):
    if tools is None:
        tools = [bm.WheelZoomTool(), bm.BoxZoomTool(),
                 bm.ResetTool(), bm.PanTool()]

    p = bp.figure(tools=tools, outline_line_color=None, min_border=0, **kwargs)
    p.axis.visible = axis_visible
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    from bokeh.tile_providers import CARTODBPOSITRON
    p.add_tile(CARTODBPOSITRON)
    return p


def read_shapefile(shp_path):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' column holding
    the geometry information. This uses the pyshp package
    """

    # read file, parse out the records and shapes
    sf = shapefile.Reader(shp_path)
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]

    # write into a dataframe
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)

    return df
