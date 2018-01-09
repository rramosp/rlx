import numpy as np
from bokeh import layouts as bl
from bokeh import models as bm
from bokeh import plotting as bp
import pandas as pd
import bokeh
import shapefile
from shapely import affinity

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
    shapes  = sf.shapes()
    coords = [np.r_[s.points] for s in shapes]
    types  = [s.shapeType for s in shapes]
    bboxes = [np.r_[s.bbox] for s in shapes]

    # write into a dataframe
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=coords)
    df = df.assign(bbox=bboxes)
    df = df.assign(shape_type=types)

    return df

def linepol_intersect(line, pol, scale=1):
    """
    returns a list of lines corresponding to the intersection of a shapely line and polygon.
    in general, the intersection is a set of lines.
    if there is no intersection it returns an empty list.

    line and pol can be a list or ndarray of pairs

    scale allows to scale the polygon before intersecting. scale=2 doubles the size
    typically one would scale by 1.05 (or something small) to make sure
    bounday lines are included in the interesection
    
    example:
    
        pol = sg.Polygon([(3., -2.0), (4.5, -1.), (5.0, -2.0), (4.0, -3.0), (3.0, -3.0), (4.0, -2.0)])
        line = sg.LineString([[5,-3], [4,-2.5],[3.5,-2], [3,-1]])
        lint = linpol_intersect(line, pol)
        plt.plot(*line.xy)
        plt.plot(*pol.boundary.xy)
        for g in lint:
            plt.plot(*g.xy, color="red", lw=10, alpha=.5)    
    """
    if type(line)==list or type(line)==np.ndarray:
       line = sg.LineString(line)

    if type(pol)==list or type(pol)==np.ndarray:
       pol = sg.Polygon(pol)

    if scale!=1:
       pol = affinity.scale(pol, scale, scale)

    lint = pol.intersection(line)
    if type(lint)==sg.LineString:
        return [lint]
    elif type(lint)==sg.MultiLineString:
        return list(lint.geoms)
    elif type(lint)==sg.GeometryCollection:
        return list(lint.geoms)
    else:
        return []


def linespol_intersects(lines, pol, scale=1):
    """
    intersects a set of lines with a polygon
    returns:
        rlines: the lines with in the input list that intersect the polygon
        rintersects: a True/False list signalling which lines in the input
                     list intersected the polygon
    """
    
    from rlx.utils import pbar
    rlines, rintersects = [], []
    for line in pbar()(lines):
        intersection = linepol_intersect(line, pol, scale)
        if len(intersection)>0:
            rlines.append(intersection)
            rintersects.append(True)
        else:
            rintersects.append(False)
    return rlines, np.r_[rintersects]


def resample_path(path, sampling_distance):
    """
    resamples path at constant distances
    
    path: a 2D nd array, each row is a 2D point
    sampling_distance: the distance between samples in the resulting path
    """
    
    def sample_at_distance(path, sampling_distance):
        cumdists = np.cumsum(np.sqrt(np.sum((path[1:]-path[:-1])**2, axis=1)))
        mid = np.argwhere(cumdists>sampling_distance)
        if len(mid)>0:
            k = mid[0][0]
            p0=path[k]
            p1=path[k+1]
            r = p1-(cumdists[k]-sampling_distance)*(p1-p0)/np.linalg.norm(p1-p0)
            return r, np.r_[[r]+list(path[k+1:])]

        return path[-1], None

    remaining_path = path
    sampled_path   = [remaining_path[0]]
    while remaining_path is not None:
        p, remaining_path = sample_at_distance(remaining_path, sampling_distance=sampling_distance)
        sampled_path.append(p)
        
    sampled_path = np.r_[sampled_path]
    return sampled_path


def get_javascript_google_map(map_id, api_key, lat, lon, zoom,
                              map_type="roadmap", heatmap_data="[[]]"):
    JS = """
       <div id="map_##MAP_ID##" style="height: 600px; width: 100%"></div>

       <script>

       var map_##MAP_ID##
       function initMap_##MAP_ID##() {
          ##MAP_DATA##
          window.##MAP_ID##_counter = 0;
          map_##MAP_ID## = new google.maps.Map(document.getElementById('map_##MAP_ID##'), {
            zoom: ##ZOOM##,
            center: {lat: ##LAT##, lng: ##LON##},
            mapTypeId: '##MAP_TYPE##'
          });
          window.heatmap_##MAP_ID## = new google.maps.visualization.HeatmapLayer({
            data: window.##MAP_ID##_LOCS[0],
            map: map_##MAP_ID##,
            max_intensity: 15,
            radius: 30
          });

       }


       </script>
       <script async defer
            src="https://maps.googleapis.com/maps/api/js?key=##API_KEY##&libraries=visualization&callback=initMap_##MAP_ID##">
       </script>

    """

    JS = JS.replace("##MAP_ID##", map_id) \
        .replace("##LAT##", str(lat)) \
        .replace("##LON##", str(lon)) \
        .replace("##MAP_TYPE##", map_type) \
        .replace("##API_KEY##", api_key) \
        .replace("##MAP_DATA##", heatmap_data) \
        .replace("##ZOOM##", str(zoom))

    return JS
