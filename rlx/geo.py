import numpy as np
from bokeh import layouts as bl
from bokeh import models as bm
from bokeh import plotting as bp
import pandas as pd
import bokeh
import os
import shapefile
from shapely import affinity
import shapely as sh
from PIL import Image
from urllib import urlopen
import cStringIO
import gmaps
import matplotlib.pyplot as plt
import itertools
from rlx.utils import flatten, pbar
import utm
import re
from skimage.io import imsave, imread
import descartes

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


def read_shapefile(shp_path, utm_zone_number=None, utm_zone_letter=None):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' column holding
    the geometry information. This uses the pyshp package.

    If utm_zone_number and utm_zone_letter are specified it assumes coords
    are in utm and additional column latlon_coords is created with coordinates
    int lat lon.
    """

    # shapefiles come in multipolygons, with one list of
    # coordinates (points) and a list of indices (parts).
    # the following function creates a list of polygons from this spec.
    def get_partitions(shpoints, shparts):
        ps = np.r_[shpoints]
        pr = np.r_[shparts]

        partition = []
        parts = list(pr)+[len(ps)]
        for i in range(len(parts)-1):
            start, end = parts[i], parts[i+1]
            partition.append(ps[start:end])

        return partition

    # read file, parse out the records and shapes
    print "reading shapefile"
    sf = shapefile.Reader(shp_path)
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shapes  = sf.shapes()
    coords = [get_partitions(s.points, s.parts) for s in shapes]
    types  = [s.shapeType for s in shapes]
    bboxes = [np.r_[s.bbox] for s in shapes]

    # write into a dataframe
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=coords)
    df = df.assign(bbox=bboxes)
    df = df.assign(shape_type=types)

    if utm_zone_letter is not None and utm_zone_number is not None:
        print "converting to latlon"
        df["latlon_coords"]=[[np.r_[[utm.to_latlon(i[0],i[1], utm_zone_number, utm_zone_letter) for i in k]] for k in sc] for sc in df.coords]


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

def get_javascript_google_maps_location_list(locations):
    return "[" + " ".join(["new google.maps.LatLng(%f,%f)," % (i[0], i[1]) for i in locations])[:-1] + "]"


def get_javascript_google_maps_data(map_id, locations_lists,
                                    makelist_func=get_javascript_google_maps_location_list,
                                    vname="LOCS"):

    s = ["window." + map_id + "_"+vname+"=[];\n"] + [
        "window." + map_id + "_"+vname+"[" + str(c) + "]=" + makelist_func(i) for c, i in
        enumerate(locations_lists)]
    return ";\n".join(s)

def get_javascript_google_map(map_id, api_key, lat, lon, zoom,
                              map_type="roadmap", heatmap_data="", gesture_handling="none",
                              markers_data="", markers_info="", marker_img="https://s3.amazonaws.com/rlx/streetview_detection/dot.png"):
    JS = """
       <div id="map_##MAP_ID##" style="height: 600px; width: 100%"></div>

       <script>

       var map_##MAP_ID##

       function set_markers_##MAP_ID##(idx) {
            if (typeof window.##MAP_ID##_MARKERS === 'undefined' || window.##MAP_ID##_MARKERS === null) {
               return;
            }
            delete_markers_##MAP_ID##();
            var image = {
              url : "##MARKER_IMG##",
              size: new google.maps.Size(13, 9),
              origin: new google.maps.Point(0, 0),
              anchor: new google.maps.Point(0, 0)
           };
            window.##MAP_ID##_CURRENT_MARKERS = []
            window.##MAP_ID##_FUNCS = []

            infowindow = new google.maps.InfoWindow();
            for (var i=0; i < window.##MAP_ID##_MARKERS[idx].length; i++) {
                var marker = new google.maps.Marker({
                    position: window.##MAP_ID##_MARKERS[idx][i],
                    map: map_##MAP_ID##,
                    icon: image
                });
                if (typeof window.##MAP_ID##_MARKERSINFO !== 'undefined' && window.##MAP_ID##_MARKERSINFO !== null) {

                    funcstr = 'f = function() {console.log("action"); infowindow.setContent("CCC");'+ \
                                                                     'infowindow.setPosition({lat:LAT, lng:LNG} );'+ \
                                                                     'infowindow.open(map_##MAP_ID##);};'
                    funcstr = funcstr.replace('CCC',window.##MAP_ID##_MARKERSINFO[idx][i]);
                    funcstr = funcstr.replace('LAT', window.##MAP_ID##_MARKERS[idx][i].lat());
                    funcstr = funcstr.replace('LNG', window.##MAP_ID##_MARKERS[idx][i].lng());
                    eval(funcstr);
                    window.##MAP_ID##_FUNCS.push(f);
                    marker.addListener('click', window.##MAP_ID##_FUNCS[i]);
                }
                window.##MAP_ID##_CURRENT_MARKERS.push(marker);
            }

       }

       function delete_markers_##MAP_ID##() {
           for (var i=0; i < window.##MAP_ID##_CURRENT_MARKERS.length; i++) {
               window.##MAP_ID##_CURRENT_MARKERS[i].setMap(null);
           }
           window.##MAP_ID##_CURRENT_MARKERS = [];
       }


       function initMap_##MAP_ID##() {
          ##HEATMAP_DATA##
          ##MARKERS_DATA##
          ##MARKERS_INFO##
          window.##MAP_ID##_counter = 0;
          window.##MAP_ID##_CURRENT_MARKERS = [];
          map_##MAP_ID## = new google.maps.Map(document.getElementById('map_##MAP_ID##'), {
            zoom: ##ZOOM##,
            center: {lat: ##LAT##, lng: ##LON##},
            mapTypeId: '##MAP_TYPE##',
            gestureHandling: '##GESTURE_HANDLING##'
          });

          if (typeof window.##MAP_ID##_LOCS !== 'undefined' && window.##MAP_ID##_LOCS !== null) {
              window.heatmap_##MAP_ID## = new google.maps.visualization.HeatmapLayer({
                data: window.##MAP_ID##_LOCS[0],
                map: map_##MAP_ID##,
                max_intensity: 15,
                radius: 30
              });
          }

          set_markers_##MAP_ID##(0);
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
        .replace("##HEATMAP_DATA##", heatmap_data) \
        .replace("##MARKERS_DATA##", markers_data) \
        .replace("##MARKERS_INFO##", markers_info) \
        .replace("##ZOOM##", str(zoom))\
        .replace("##MARKER_IMG##", marker_img)\
        .replace("##GESTURE_HANDLING##", gesture_handling)

    return JS

def get_shapely_multipolygon(rawcoords):
    """
    rawcoords: a list of 2D arrays, each one representing a polygon
               the first one is the outer polygon, the rest are the "holes"
    """
    tupleofcoords = tuple(rawcoords[0])
    #the remaining linear rings, if any, are the coordinates of inner holes, and shapely needs these to be nested in a list
    if len(rawcoords) > 1:
        listofholes = list(rawcoords[1:])
    else:
        listofholes = []
    #shapely defines each polygon in a multipolygon with the polygoon coordinates and the list of holes nested inside a tuple
    eachpreppedpolygon = (tupleofcoords, listofholes)
    #finally, the prepped coordinates need to be nested inside a list in order to be used as a star-argument for the MultiPolygon constructor.
    preppedcoords = [[eachpreppedpolygon]]


    shapelymultipolygon = sh.geometry.MultiPolygon(*preppedcoords)
    return shapelymultipolygon


class GoogleMaps_Static_Image:
    @classmethod
    def from_filename(cls, fname):
        p = re.compile('(\S*)gmaps_(\S+)_(\S+)_zoom_(\d+)_(\d+)x(\d+)_([^_]+)_([^_]+).jpg').match(fname)
        if p is None:
            return None
        dir, lat, lon, zoom, px, py, mtype, logo = p.groups()
        g = GoogleMaps_Static_Image(np.float64(lat), np.float64(lon), int(zoom),
                                        (int(px), int(py)),
                                        mtype, savedir=dir, crop_google_logo=False)
        g.crop_google_logo = logo=="nologo"
        g.set_google_logo_height()
        g.h += g.google_logo_height
        return g

    def __init__(self, lat, lon, zoom, size, maptype="roadmap", apikey=None,
                 verbose=0, savedir=None, crop_google_logo=True):


        self.lat = np.float64(lat)
        self.lon = np.float64(lon)
        self.zoom = zoom
        self.w = size[0]
        self.h = size[1]
        self.maptype = maptype
        self.apikey  = apikey
        self.img     = None
        self.verbose = verbose
        self.savedir = savedir
        self.crop_google_logo = crop_google_logo
        self.set_google_logo_height()

        parallelMultiplier = np.cos(lat * np.pi / 180)
        self.degreesPerPixelX = 360. / np.power(2, self.zoom + 8)
        self.degreesPerPixelY = 360. / np.power(2, self.zoom + 8) * parallelMultiplier

    def set_google_logo_height(self):
        self.google_logo_height = 20 if self.crop_google_logo else 0

    def set_apikey(self, apikey):
        self.apikey = apikey

    def set_savedir(self, savedir):
        self.savedir = savedir

    def get_url(self, apikey=None):
        apikey = self.apikey if apikey is None else apikey
        assert apikey is not None, "must set apikey"
        s = "https://maps.googleapis.com/maps/api/staticmap?center=%s,%s&zoom=%s&size=%sx%s&maptype=%s&key=%s"% \
            (str(self.lat),str(self.lon),str(self.zoom), str(self.w), str(self.h), self.maptype, apikey)
        return s

    def get_img_size(self):
        return self.w, self.h-self.google_logo_height

    def get_img(self, apikey=None):
        if self.img is not None:
            return self.img

        if os.path.isfile(self.get_fname()):
            f = self.get_fname()
            if self.verbose>0:
                print "loading from file",f
            return Image.open(f)

        url = self.get_url(apikey)
        if self.verbose>0:
            print "retrieving", url
        file = cStringIO.StringIO(urlopen(url).read())
        self.img = Image.open(file).crop((0,0,self.w, self.h-self.google_logo_height))
        return self.img

    def get_polygon(self):
        # polygon coordinate order is lon,lat (to facilitate x,y plotting)
        bbox = self.get_bbox()
        return np.r_[[bbox["NE"][::-1], bbox["NW"][::-1], bbox["SW"][::-1], bbox["SE"][::-1]]]

    def get_point_latlon(self, x, y):
        pointLat = self.lat - self.degreesPerPixelY * ( y - self.h / 2)
        pointLng = self.lon + self.degreesPerPixelX * ( x  - self.w / 2)

        return np.r_[(pointLat, pointLng)]

    def get_bbox(self):
        return {
                "SW": self.get_point_latlon(0, self.h-self.google_logo_height),
                'NE': self.get_point_latlon(self.w, 0),
                'SE': self.get_point_latlon(self.w, self.h-self.google_logo_height),
                'NW': self.get_point_latlon(0, 0)
               }

    def get_fname(self):
        savedir = self.savedir
        assert savedir is not None, "must set savedir"
        return savedir+"/gmaps_%s_%s_zoom_%s_%sx%s_%s%s.jpg"%\
                            (str(self.lat),str(self.lon),str(self.zoom), str(self.w), str(self.h-self.google_logo_height),
                            self.maptype, "_nologo" if self.crop_google_logo else "")


    def save(self, apikey=None, overwrite=False):
        fname = self.get_fname()
        if not overwrite and os.path.isfile(fname):
            if self.verbose>0:
                print "skipping existing", fname
            return
        if self.verbose>0:
            print "saving to", fname
        self.get_img(apikey).convert('RGB').save(fname)

    def copy_to(self, lat, lon):
        return GoogleMaps_Static_Image(lat=lat, lon=lon, zoom=self.zoom, size=(self.w, self.h),
                                       maptype=self.maptype, apikey=self.apikey, savedir=self.savedir,
                                       verbose=self.verbose, crop_google_logo=self.crop_google_logo)

    def get_next_east(self):
        lat,lon = self.get_point_latlon(self.w*3/2, self.h/2)
        return self.copy_to(lat, lon)

    def get_next_west(self):
        lat,lon = self.get_point_latlon(-self.w/2, self.h/2)
        return self.copy_to(lat, lon)

    def get_next_south(self):
        lat,lon = self.get_point_latlon(self.w/2, self.h*3/2-self.google_logo_height)
        return self.copy_to(lat, lon)

    def get_next_north(self):
        lat,lon = self.get_point_latlon(self.w/2, -self.h/2+self.google_logo_height)
        return self.copy_to(lat, lon)

    def get_gmap_polygon(self):
        b = self.get_bbox()
        return gmaps.Polygon([tuple(b["NE"]), tuple(b["NW"]), tuple(b["SW"]), tuple(b["SE"])])

    def show_in_gmap(self, apikey=None):
        apikey = self.apikey if apikey is None else apikey
        assert apikey is not None, "must set apikey"

        gmaps.configure(api_key=self.apikey)
        gmap_b = self.get_gmap_polygon()
        fig = gmaps.figure(center=(self.lat, self.lon), zoom_level=self.zoom)
        fig.add_layer(gmaps.drawing_layer(features=[gmap_b]))
        return fig

    def get_html_imgtag(self, width=None, height=None, class_tag=None):
        from rlx.dashboards import get_img_tag
        k = np.array(self.get_img().convert("RGB"))
        fig=plt.figure()
        plt.imshow(k)
        plt.axis("off")
        tag = get_img_tag(fig, width, height, class_tag)
        plt.close()
        return tag

    def get_area(self,  units="m2"):
        allowed_units = ["m2", "km2"]
        assert units in allowed_units, "units must be one of "+str(allowed_units)
        bb = self.get_bbox()
        mne = latlon_to_meters(*bb["NE"])
        msw = latlon_to_meters(*bb["SW"])
        r = (mne[0]-msw[0])*(mne[1]-msw[1])
        if units=="km2":
            r = r/1e6
        return r

    def get_area_str(self):
        a = self.get_area()
        return "%d m2"%a if a<=1e4 else "%.3f km2"%(a/1e6) if a<1e5 else "%.2f km2"%(a/1e6)

    def __repr__(self):
        b = self.get_bbox()
        s =    "center:  lat %s, lon %s"%(str(self.lat), str(self.lon))
        s += "\nzoom:    %d"%self.zoom
        s += "\nsize:    %dx%d px"%(self.w,self.h-self.google_logo_height)
        s += "\nmaptype: %s"%self.maptype
        s += "\narea:    " + self.get_area_str()
        s += "\nbbox:\n"+"\n".join(["    "+k+": "+str(v) for k,v in b.iteritems()])
        return s


class GoogleMaps_Static_Mosaic:

    def __init__(self, init_gImage, nw, nh):
        """
        init_gImage: GoogleMaps_Static_Image from the top_left of the mosaic
        nw, nh: number of images wide and tall
        """
        self.apikey = init_gImage.apikey
        self.zoom   = init_gImage.zoom
        self.nw = nw
        self.nh = nh
        self.mosaic = [["" for _ in range(nw)] for _ in range(nh)]
        gw = gh = init_gImage
        for _nw in range(nw):
            for _nh in range(nh):
                self.mosaic[_nh][_nw] = gh
                gh = gh.get_next_south()
            gw = gh = gw.get_next_east()

        # the mosaic lat,lon is the mean of the composing images centers
        from rlx.utils import flatten
        flat =  flatten(self.mosaic)
        self.lat = np.mean([i.lat for i in flat])
        self.lon = np.mean([i.lon for i in flat])
        self.bbox = self.get_bbox()

    def get_single_img(self):
        init_g = self.mosaic[0][0]
        img = np.array(init_g.get_img().convert("RGB"))
        w_px,h_px = img.shape[1], img.shape[0]

        k = np.zeros((h_px*self.nh, w_px*self.nw,3)).astype(img.dtype)

        for _nw,_nh in pbar(max_value=self.nw*self.nh)(itertools.product(range(self.nw), range(self.nh))):
            k[_nh*h_px:(_nh+1)*h_px, _nw*w_px:(_nw+1)*w_px, : ] = np.array(self.mosaic[_nh][_nw].get_img().convert("RGB"))
        return Image.fromarray(k)

    def get_imgs(self):
        r = [["" for _ in range(self.nw)] for _ in range(self.nh)]
        for _nw,_nh in itertools.product(range(self.nw), range(self.nh)):
            r[_nh][_nw] = self.mosaic[_nh][_nw].get_img()
        return r

    def get_gmap_polygon(self):
        b = self.get_bbox()
        return gmaps.Polygon([tuple(b["NE"]), tuple(b["NW"]), tuple(b["SW"]), tuple(b["SE"])])

    def show_in_gmap(self, apikey=None, zoom=None, show_grid=False):
        apikey = self.apikey if apikey is None else apikey
        assert apikey is not None, "must set apikey"

        zoom = self.zoom if zoom is None else zoom
        gmaps.configure(api_key=self.apikey)
        fig = gmaps.figure(center=(self.lat, self.lon), zoom_level=zoom)
        if show_grid:
            gmap_b = [i.get_gmap_polygon() for i in flatten(self.mosaic)]
            fig.add_layer(gmaps.drawing_layer(features=gmap_b))

        fig.add_layer(gmaps.drawing_layer(features=[self.get_gmap_polygon()]))

        return fig

    def get_bbox(self):
        if hasattr(self, 'bbox') and self.bbox is not None:
            return self.bbox
        def ops_corner(corner, oplat, oplon):
            k = np.r_[[i.get_bbox()[corner] for i in flatten(self.mosaic)]]
            return oplat(k[:,0]), oplon(k[:,1])
        r = {}
        r["NE"] = ops_corner("NE", np.max, np.max)
        r["NW"] = ops_corner("NW", np.max, np.min)
        r["SW"] = ops_corner("SW", np.min, np.min)
        r["SE"] = ops_corner("SE", np.min, np.max)
        return r

    def save(self, overwrite=False):
        for i in pbar()(flatten(self.mosaic)):
            i.save(overwrite=overwrite)

    def get_area_str(self):
        t = self.mosaic[0][0]
        a = t.get_area() * self.nw*self.nh
        return "%d m2"%a if a<=1e4 else "%.3f km2"%(a/1e6) if a<1e5 else "%.2f km2"%(a/1e6)

    def __repr__(self):
        b = self.get_bbox()
        t = self.mosaic[0][0]
        s =    "center:  lat %s, lon %s"%(str(self.lat), str(self.lon))
        s += "\nzoom:    %d"%self.zoom
        s += "\ntiles:   %d"%(self.nw*self.nh)
        s += "\nsize:    %dx%d px"%(self.nw*t.w,self.nh*(t.h-t.google_logo_height))
        s += "\n         %dx%d tiles (%dx%d px each)"%(self.nw,self.nh, t.w, t.h-t.google_logo_height)
        s += "\nmaptype: %s"%t.maptype
        s += "\narea:    " + self.get_area_str()
        s += "\nbbox:\n"+"\n".join(["    "+k+": "+str(v) for k,v in b.iteritems()])
        return s


class GoogleMaps_Shapefile_Layer:

    def __init__(self, layer_name, shapefile_name, utm_zone_number, utm_zone_letter):
        self.fname = shapefile_name
        self.layer_name = layer_name
        self.utm_zone_number = utm_zone_number
        self.utm_zone_letter = utm_zone_letter
        if shapefile_name is not None:
            self.shapefile = read_shapefile(shapefile_name, utm_zone_number=utm_zone_number, utm_zone_letter=utm_zone_letter)
            self.generate_polygons()
        else:
            self.shapefile = None
        self.color_function = None

    def generate_polygons(self):
        print "generating polygons"
        self.mpols = [get_shapely_multipolygon([i[:,::-1] for i in p]) for p in pbar()(self.shapefile.latlon_coords.values)]

    def set_color_function(self, func):
        """
        func takes one argument which is a pd.Series representing a row from self.shapefile
        and returns a color spec
        """
        self.color_function = func

    def save_layer_patches_for_gmaps_img(self, gmaps_img, kwargs_list):
        for kwargs in kwargs_list:
            if not self.save_layer_patch_for_gmaps_img(gmaps_img, **kwargs):
                break

    def save_layer_patch_for_gmaps_img(self, gmaps_img, target_dir, color_func,
                                       suffix="", overlay_original=False, verbose=False,
                                       default_color="white", default_alpha=1.,
                                       single_channel_map=None, format="jpg"):
        self.set_color_function(color_func)

        lname = target_dir+"/"+(".".join(gmaps_img.get_fname().split(".")[:-1])+"_%s%s.%s"%(self.layer_name, suffix, format)).split("/")[-1]
        if os.path.isfile(lname):
            if verbose:
                print "skipping existing", lname
            return True

        bbox = sh.geometry.Polygon(gmaps_img.get_polygon())
        si = self.shapefile.iloc[[bbox.intersects(p) for p in self.mpols]]

        pols = [get_shapely_multipolygon([i[:,::-1] for i in p]) for p in si.latlon_coords.values]
        fcols = [self.color_function(i) for _, i in si.iterrows()]
        cols = [i[0] if type(i)==tuple else i for i in fcols]
        alphas = [i[1] if type(i)==tuple else 1. for i in fcols]

        if len(pols)==0:
            if verbose:
                print "no shapefile info for %s"%gmaps_img.get_fname()
            return False
        # compute bounding box for all polygons
        union = pols[0]
        for i in pols[1:]:
            union = union.union(i)

        xmin, ymin = np.r_[[sh.geometry.mapping(bbox)["coordinates"][0]]].min(axis=1)[0]
        xmax, ymax = np.r_[[sh.geometry.mapping(bbox)["coordinates"][0]]].max(axis=1)[0]
        w,h = gmaps_img.get_img_size()

        xscale = w/(xmax-xmin)
        yscale = h/(ymax-ymin)
        fig = plt.figure(figsize=(w*1./100, h*1./100), dpi=100, frameon=False)
        ax = fig.add_subplot(111)

        # make background polygon
        bpol = sh.geometry.Polygon(([0,0], [w,0], [w,h], [0,h]))

        # intersect all polygons with bounding box and scale them to img pixels
        for i in range(len(pols)):
            pol = pols[i]
            pol = pol.intersection(bbox)
            kpol = sh.affinity.translate(pol, xoff=-xmin, yoff=-ymin)
            kpol = sh.affinity.scale(kpol, xfact=w/(xmax-xmin), yfact=h/(ymax-ymin), origin=(0,0))
            bpol = bpol.difference(kpol)
            ax.add_patch(descartes.PolygonPatch(kpol, color=cols[i], lw=0, alpha=alphas[i]))

        ## add remaining space as white
        if bpol.area>0:
            ax.add_patch(descartes.PolygonPatch(bpol, color=default_color, lw=0, alpha=default_alpha))

        ax.set_xlim((0,w))
        ax.set_ylim((0,h))
        if overlay_original:
            plt.imshow(np.flip(np.array(gmaps_img.get_img()), axis=0), origin="bottom")

        ax.set_axis_off()
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)

        if verbose:
            print "saving to", lname
        fig.savefig(lname)
        plt.close()

        if single_channel_map is not None:
            print "saving single channel"
            k = imread(lname)
            k = convert_label_to_single_channel(k, single_channel_map)
            imsave(lname, k)
            return k

        return True

def convert_label_to_single_channel(multi_channel_label_img, channel_map):
    assert len(multi_channel_label_img.shape)==3, "img must be multichannel"

    if multi_channel_label_img.shape[2]==4:
        multi_channel_label_img = multi_channel_label_img[:,:,:3]
    r = (np.r_[[np.abs(multi_channel_label_img-i).sum(axis=2) for i in channel_map]].argmin(axis=0)).astype(np.uint8)
    lmap = len(channel_map)
    slmap = {i: int((i*255./lmap)) for i in range(lmap)}
    # converts single channel image to a 0-255 pixel value range
    return np.r_[[slmap[i] for i in r.flatten()]].reshape(r.shape)

def convert_label_to_multi_channel(single_channel_label_img, channel_map):
    s = single_channel_label_img.shape
    lmap = len(channel_map)
    # converts back 0-255 single channel to 0-num_classes
    slmap = {int((i*255./lmap)):i for i in range(lmap)}
    kt = np.r_[[slmap[i] for i in single_channel_label_img.flatten()]].reshape(single_channel_label_img.shape)
    return np.r_[[channel_map[i] for i in kt.flatten()]].reshape( list(s)+[3] ).astype(np.uint8)



def show_channel_map(channel_map, matplotlib_colormap=plt.cm.jet):
    w,h = 200,100
    tile_w = w/len(channel_map)
    k = np.zeros((w,h,3))

    fig = plt.figure(figsize=(20,2))
    ax = fig.add_subplot(111)

    for i in range(len(channel_map)):
        p = sh.geometry.Polygon([(tile_w*i,0),(tile_w*i,h),(tile_w*(i+1),h),(tile_w*(i+1),0)])
        ax.add_patch(descartes.PolygonPatch(p, color=tuple(np.r_[channel_map[i]]/255.), lw=0, alpha=.9))
    plt.xlim(0,w)
    plt.ylim(0,1)
    plt.title("original multichannel")
    ax.set_axis_off()
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    fig.savefig("/tmp/cols.jpg")

    plt.figure(figsize=(20,2))
    plt.title("single channel")
    k = imread("/tmp/cols.jpg")
    ks = convert_label_to_single_channel(k, channel_map)
    plt.title("single channel, levels="+str(np.unique(ks.flatten())))
    plt.imshow(ks, cmap=matplotlib_colormap)
    plt.axis("off")

    plt.figure(figsize=(20,2))
    plt.title("recovered multichannel")
    kk = convert_label_to_multi_channel(ks, channel_map)
    plt.imshow(kk, cmap=matplotlib_colormap)
    plt.axis("off")
