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
import sys
if sys.version[0]=='3':
    from urllib.request import urlopen
else:
    from urllib import urlopen
from io import BytesIO
import gmaps
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print "warning: matplotlib not loaded", e
import itertools
from rlx.utils import flatten, pbar, most_common_neighbour, humanbytes
import utm
import re
from skimage.io import imsave, imread
import descartes
import hashlib
from PIL import Image
import hashlib
from io import BytesIO
from copy import copy, deepcopy
import geopandas as gpd
import pickle
from shutil import make_archive, rmtree
from time import time
import gzip

try:
    from shapely import geometry as sg
except OSError as err:
    print ("WARN! could not load shapely library")
    print (err)

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

def get_area_str(a):
    """
    a: area in m2
    """
    return "%.2f m2"%a if a<10 else  "%d m2"%a if a<=1e4 else "%.3f km2"%(a/1e6) if a<1e5 else "%.2f km2"%(a/1e6)

def get_distance_str(d):
    """
    d: distance in m
    """
    return "%.1f"%d+"m" if d<1e3 else "%.2f"%(d/1e3)+"km"

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
    print ("reading shapefile")
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
        print ("converting to latlon")
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

class TileSet(object):

    def __init__(self, dir=None, metadata_geodf=None, dont_generate_tiles=False):
        assert not (dir is None and metadata_geodf is None), "must specify dir or metadata_geodf"
        assert not(dir is not None and metadata_geodf is not None), "can only set one: dir or metadata_geodf"

        self.dir = dir
        if self.dir is not None:
            self.metadata = gpd.read_file(self.get_metadata_filename())
        else:
            self.metadata = metadata_geodf

        if not dont_generate_tiles:
            self.tiles = [Tile.from_geoseries(i, savedir=self.dir, use_file_cache=self.dir is not None) for _,i in self.metadata.iterrows()]
            self.compute_properties()

    def __len__(self):
        return len(self.metadata)

    def get_metadata_filename(self, dir=None):
        dir = self.dir if dir is None else dir
        return dir+"/tileset.geojson" if dir is  not None else None

    def compute_properties(self):

        assert len(self.metadata)==0 or np.std(self.metadata.w)==0, "all tiles must have the same width"
        assert len(self.metadata)==0 or np.std(self.metadata.h)==0, "all tiles must have the same height"
        assert len(self.metadata)==0 or np.std(self.metadata.zoom)==0, "all tiles must have the same zoom level"

        if len(self)>0:
            areas = [i.get_area() for i in self.tiles]

            sample_tile = self.tiles[0]

            self.total_area = np.sum(areas)
            self.tile_resolution = sample_tile.get_resolution
            self.tile_w = sample_tile.w
            self.tile_h = sample_tile.h
            self.tile_zoom = sample_tile.zoom
            self.tile_area_mean = np.mean(areas)
            self.tile_area_std = np.std(areas)
            self.center_lat = np.mean(self.metadata.center_lat)
            self.center_lon = np.mean(self.metadata.center_lon)
            w,s,e,n = self.metadata.total_bounds
            self.bbox = {"SW": [w,s], "NE": [e,n], "SE": [e,s], "NW": [w,n]}

        else:
            self.total_area = 0
            self.tile_resolution = lambda: (0,0)
            self.tile_w = 0
            self.tile_h = 0
            self.tile_zoom = 0
            self.tile_area_mean = 0
            self.tile_area_std = 0
            self.center_lat = 0
            self.center_lon = 0
            self.bbox = {"SW": [0,0], "NE": [0,0], "SE": [0,0], "NW": [0,0]}
            
    def get_size_meters(self):
        mne = latlon_to_meters(*self.bbox["NE"])
        msw = latlon_to_meters(*self.bbox["SW"])
        my = mne[0]-msw[0]
        mx = mne[1]-msw[1]
        return mx, my

    def __getitem__(self, slice):

        if type(slice)==int:
            r = self.metadata.iloc[slice:slice+1]
            rtiles = self.tiles[slice:slice+1]
        else:
            r = self.metadata.iloc[slice]
            rtiles = list(np.r_[self.tiles][slice])

        r = self.__class__(metadata_geodf=r)
        r.dir = self.dir
        r.tiles = rtiles

        return r

    def __repr__(self):
        rx, ry = self.tile_resolution()
        mx, my = self.get_size_meters()

        s =  "number of tiles:     %d"%len(self)
        if self.dir is not None:
            files = [i.get_local_filename() for i in self.tiles]
            imgs_size    = humanbytes(np.sum([os.path.getsize(i) if os.path.exists(i) else 0 for i in files]))
            imgs_nbfiles = np.sum([os.path.exists(i) for i in files])
            total_files = len(os.listdir(self.dir))
            metadata_size = humanbytes(os.path.getsize(self.get_metadata_filename()))        

            s += "\nnb img files:        %d"%imgs_nbfiles
            s += "\ntotal files in disk: %d"%total_files
            s += "\nimgs size in disk:   "+imgs_size
            s += "\nmetadata size:       "+metadata_size
        else:
            s += "\n\nno disk storage"

        s += "\n"
        s += "\ntileset center:  lat %s, lon %s"%(str(self.center_lat), str(self.center_lon))
        s += "\naggregated area: "+get_area_str(np.sum(self.total_area))
        s += "\n"
        s += "\nbounding box size: "+get_distance_str(mx)+" x "+get_distance_str(my)
        s += "\nbounding box area: "+get_area_str(mx*my)
        s += "\nbounding box:\n"+"\n".join(["    "+k+": "+str(v) for k,v in self.bbox.iteritems()])
        s += "\n"
        s += "\ntile zoom:        %d"%self.tile_zoom
        s += "\ntile size:        %dx%d px"%(self.tile_w,self.tile_h) #+ "   "+get_distance_str(mx)+" x "+get_distance_str(my)
        s += "\ntike resolution:  %.2f m/pixel X %.2f m/pixel"%(rx, ry)
        s += "\ntile area:        " + get_area_str(self.tile_area_mean)+" +/- "+get_area_str(self.tile_area_std)
        return s


    def save(self, dir=None, show_progress=True, overwrite_metadata=False, **kwargs):
        if dir is None:
            dir = self.dir            

        assert dir is not None, "must set tileset dir"

        if not os.path.exists(dir):
            os.makedirs(dir)

        assert overwrite_metadata or not os.path.exists(self.get_metadata_filename(dir)), "metadata file exists, use overwrite_metadata option"
        self.metadata.crs='+init=epsg:4326'

        if overwrite_metadata and os.path.exists(self.get_metadata_filename(dir)):
            os.remove(self.get_metadata_filename(dir))

        t = pbar(max_value=len(self))(self.tiles) if show_progress else self.tiles
        for i in t:
            if i.savedir is not None and i.savedir!=dir:
                i.get_img() # forces img retrieval if tile is from a different source than dir
            i.savedir = dir
            i.use_file_cache = True
            i.save(**kwargs)

        self.dir = dir
        self.metadata.to_file(driver="GeoJSON", filename=(self.get_metadata_filename()))

    def get_ids(self):
        return [i.get_id() for i in self.tiles]

    def union(self, other_tileset):
        """
        collates tilesets removing duplicates
        """
        o = other_tileset
        assert self.dir is not None and o.dir is not None, "tilesets must have storage ('dir' must be set)"

        g = pd.concat((self.metadata, o.metadata))
        t = self.tiles + o.tiles

        # keep only tiles with no duplicates
        ids = [i.get_id() for i in t]
        idxs = [ids[i] not in ids[:i] for i in range(len(ids))]

        g = g.iloc[idxs]
        t = list(np.r_[t][idxs])

        r = self.__class__(metadata_geodf=g, dont_generate_tiles=True)
        r.tiles = t
        r.compute_properties()
        return r

    def get_gmap_polygon(self):
        b = self.bbox
        return gmaps.Polygon([tuple(b["NE"])[::-1], tuple(b["NW"])[::-1], 
                            tuple(b["SW"])[::-1], tuple(b["SE"])[::-1]])
                            
    def show_in_gmap(self, apikey, mode="show_bbox", zoom=None):
        """
        mode: "show_bbox" or "show_tiles"
        """
        assert mode=="show_bbox" or mode=="show_tiles", "mode has to be 'show_bbox' or 'show_tiles'"
        gmaps.configure(api_key=apikey)
        if mode=="show_bbox":
            pols = [self.get_gmap_polygon()]
        else:
            pols = [i.get_gmap_polygon() for i in self.tiles]

        zoom = self.tile_zoom-2 if zoom is None else zoom

        fig = gmaps.figure(center=(self.center_lat, self.center_lon), zoom_level=zoom)
        fig.add_layer(gmaps.drawing_layer(features=pols, show_controls=True))
        return fig


    @classmethod
    def generate_rect_area(cls, init_tile, ntiles_width, ntiles_height, verbose=0):
 
        gw = gh = init_tile
        tiles = []
        for _nw in range(ntiles_width):
            for _nh in range(ntiles_height):
                coords = pd.Series([_nw, _nh], index=["tilecoord_w", "tilecoord_h"])
                tiles.append(pd.concat((coords, gh.to_geoseries())))
                gh = gh.get_next_south()
            gw = gh = gw.get_next_east()
 
        return cls(metadata_geodf=gpd.GeoDataFrame(tiles))

    @classmethod
    def from_area_coverage(cls, tile_template, area):
        """
        tile_template: a tile of desired type and configuration (its coords don't matter, 
                    they will be set according to the area coords)
        area: must be a shapely polygon
        """
        w,s,e,n = area.bounds
        g = tile_template
        g.center_lat = n
        g.center_lon = w
        g.compute_properties()
        tiles = []

        gh = g
        gv = g

        if area.intersects(g.get_polygon()):
            tiles.append(g)

        print "building tiles"
        while ( area.envelope.contains(sh.geometry.Point(gv.bbox["SE"])) or \
                area.envelope.contains(sh.geometry.Point(gv.bbox["NW"])) ):
            while ( area.envelope.contains(sh.geometry.Point(gh.bbox["SE"])) or \
                    area.envelope.contains(sh.geometry.Point(gh.bbox["NW"])) ):
                gh = gh.get_next_east()
                if area.intersects(gh.get_polygon()):
                    tiles.append(gh)
            gv = gv.get_next_south()
            if area.intersects(gv.get_polygon()):
                tiles.append(gv)
            gh = gv
            
        print "building tileset"
        return cls.from_tilelist(tiles)

    @classmethod
    def from_tilelist(cls, tile_list):
        metadata = gpd.GeoDataFrame([i.to_geoseries() for i in tile_list])
        r = cls(metadata_geodf=metadata, dont_generate_tiles=True)
        r.tiles = tile_list
        r.compute_properties()
        return r

    def to_kmz(self, dest_file, color):
        head = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
<Folder>    
        """
        foot = """
</Folder>
</kml>
        """
        r = "\n".join([i.to_kmz_element() for i in self.tiles])
        xml = head+r+foot
        
        tmpdir = '/tmp/'+hashlib.sha224(str(time())).hexdigest()
        files_dir = tmpdir+"/files"
        doc_file = tmpdir+"/doc.kml"
        os.makedirs(files_dir)
        
        for tile in pbar()(self.tiles):
            tile.export_to_transparency(rgb_color=color, dest_dir=files_dir)
        
        with open(doc_file, "w") as f:
            f.write(xml)
        
        print tmpdir
        
        make_archive(dest_file, "zip", root_dir=tmpdir)
        rmtree(tmpdir)
        
        os.rename(dest_file+".zip", dest_file+".kmz")

        print "kmz written to", dest_file+".kmz"

class Tile(object):
    """
    abstract class representing an arbitrary time at a certain zoom level
    """
    def __init__(self, center_lat, center_lon, zoom, pixel_size, format="jpg",
                 savedir=None, verbose=0, use_file_cache=False):

        assert not use_file_cache or (savedir is not None and use_file_cache), "must set savedir to use file cache"

        self.zoom = zoom
        self.savedir = savedir
        self.format  = format

        self.w, self.h = pixel_size

        self.verbose = verbose

        self.center_lat = np.float64(center_lat)
        self.center_lon = np.float64(center_lon)

        self.img     = None

        self.use_file_cache=use_file_cache

        self.compute_properties()

    def compute_properties(self):
        parallelMultiplier = np.cos(self.center_lat * np.pi / 180)
        self.degreesPerPixelX = 360. / np.power(2, self.zoom + 8)
        self.degreesPerPixelY = 360. / np.power(2, self.zoom + 8) * parallelMultiplier

        self.bbox = {  'SW': self.get_pixel_latlon(0, self.h),
                       'NE': self.get_pixel_latlon(self.w, 0),
                       'SE': self.get_pixel_latlon(self.w, self.h),
                       'NW': self.get_pixel_latlon(0, 0)
                    }

    def get_gmap_polygon(self):
        b = self.bbox
        return gmaps.Polygon([tuple(b["NE"])[::-1], tuple(b["NW"])[::-1], 
                            tuple(b["SW"])[::-1], tuple(b["SE"])[::-1]])

    def show_in_gmap(self, apikey):
        gmaps.configure(api_key=apikey)
        gmap_b = self.get_gmap_polygon()
        fig = gmaps.figure(center=(self.center_lat, self.center_lon), zoom_level=self.zoom-1)
        fig.add_layer(gmaps.drawing_layer(features=[gmap_b], show_controls=True))
        return fig

    def clone_with_properties(self, props):
        """
        props must be a dictionary with new properties to be set on the clone
        """
        r = copy(self)
        for k,v in props.iteritems():
            r.__setattr__(k,v)
        r.compute_properties()
        r.img = None
        return r

    def get_id(self):
        raise NotImplementedError

    def get_url(self):
        raise NotImplementedError

    def get_next_east(self):
        raise NotImplementedError

    def get_next_west(self):
        raise NotImplementedError

    def get_next_south(self):
        raise NotImplementedError

    def get_next_north(self):
        raise NotImplementedError

    def to_geoseries(self):
        r = gpd.GeoSeries([self.__class__.__name__,
                           self.get_id(), self.center_lat, self.center_lon, self.zoom, self.w, self.h,
                           self.format, self.get_polygon()],
                           index=["class_name", "id", "center_lat", "center_lon", "zoom", "w", "h", "format", "geometry"])
        return r

    @classmethod
    def from_local_file(cls, fname):
        raise NotImplementedError

    @classmethod
    def from_geoseries(cls, gs, savedir=None, use_file_cache=False, verbose=0, **kwargs):
        """
        returns an object from the class specified in the geoseries
        """
        class_object = eval(gs["class_name"])

        # generated tiles have their own from_geoseries
        if issubclass(class_object, GeneratedTile):
            r = class_object.from_geoseries(gs, savedir, verbose)
            return r

        # otherwise append what __from_geoseries__ returns if the subclass has it
        kwargs = class_object.__from_geoseries__(gs, **kwargs)
        return class_object(center_lat=gs.center_lat, center_lon=gs.center_lon, zoom=gs.zoom,
                   pixel_size=(gs.w, gs.h), format=gs.format,
                   savedir=savedir, use_file_cache=use_file_cache, verbose=verbose, **kwargs)

    @classmethod
    def __from_geoseries__(cls, gs, **kwargs):
        return kwargs

    def get_local_filename(self):
        assert self.savedir is not None, "must set storage directory"
        this_id = self.get_id()
        assert "___" not in this_id, "id cannot contain '___'"
        return self.savedir+"/"+this_id+"."+self.format

    def before_saving_pickle(self, data):
        return data

    def after_loading_pickle(self, data):
        return data


    def get_img(self, **kwargs):
        # cache img in memory
        if self.img is not None:
            return self.img

        # cache img in file
        f = self.get_local_filename()
        if self.use_file_cache and os.path.isfile(f):
            if self.verbose>0:
                print ("loading from file",f)
            if self.format=="pklz":
                with gzip.open(f,'rb') as gz:
                    img = self.after_loading_pickle(pickle.load(gz))
            else:
                img = imread(f)
            self.img = img
            return img

        url = self.get_url(**kwargs)
        if self.verbose>0:
            print ("retrieving", url)
        file = BytesIO(urlopen(url).read())
        self.img = np.array(Image.open(file).convert("RGB"))
        if self.use_file_cache:
            self.save()
        return self.img

    def save(self, savedir=None, overwrite=False, **kwargs):
        if savedir is not None:
            self.savedir=savedir
            
        assert self.savedir is not None, "must set savedir"
        
        fname = self.get_local_filename()
        if not overwrite and os.path.isfile(fname):
            if self.verbose>0:
                print ("skipping existing", fname)
            return
        if self.verbose>0:
            print ("saving to", fname)

        img = self.get_img(**kwargs)
        if self.format=="pklz":
            with gzip.open(fname,'wb') as gz:
                pickle.dump(self.before_saving_pickle(img), gz)
        else:
            imsave(fname, img)

    def get_pixel_latlon(self, x, y):
        w,h = self.w, self.h
        pointLat = self.center_lat - self.degreesPerPixelY * ( y - h / 2)
        pointLng = self.center_lon + self.degreesPerPixelX * ( x - w / 2)

        return np.r_[(pointLng, pointLat)]

    def get_polygon(self):
        w,h = self.w, self.h
        return sh.geometry.Polygon( [self.bbox["SW"], self.bbox["SE"],
                                     self.bbox["NE"], self.bbox["NW"]]  )

    def intersection(self, geo_dataframe, show_progress=False):
        """
        intersects this tile with the geometries of a GeoDataFrame
        returns a subset of geo_dataframe with geometries trimmed
        """
        pol = self.get_polygon()
        geometries = pbar()(geo_dataframe.geometry) if show_progress else geo_dataframe.geometry
        di = geo_dataframe[[i.intersects(pol) for i in geometries]].copy()
        di["geometry"] = [i.intersection(pol) for i in di.geometry]
        return di

    def get_area(self,  units="m2"):
        allowed_units = ["m2", "km2"]
        assert units in allowed_units, "units must be one of "+str(allowed_units)
        mne = latlon_to_meters(*self.bbox["NE"])
        msw = latlon_to_meters(*self.bbox["SW"])
        r = (mne[0]-msw[0])*(mne[1]-msw[1])
        if units=="km2":
            r = r/1e6
        return r

    def get_size_meters(self):
        mne = latlon_to_meters(*self.bbox["NE"])
        msw = latlon_to_meters(*self.bbox["SW"])
        my = mne[0]-msw[0]
        mx = mne[1]-msw[1]
        return mx,my


    def get_resolution(self):
        mx, my = self.get_size_meters()
        return mx/self.w, my/self.h

    def show(self, title=None, **kwargs):
        plt.imshow(self.get_img(**kwargs))
        plt.axis("off")
        if title is not None:
            plt.title(title)

    def export_to_transparency(self, rgb_color, dest_dir):
        cm = self.get_img().copy()
        assert len(cm.shape)==2, "can only export tiles with single channel"
        assert type(rgb_color)==tuple or type(rgb_color)==list, "color spec must be a 3-tuple rgb"
        assert len(rgb_color)==3, "color spec must be a 3-tuple rgb"
        
        if np.max(cm)>1:
            cm = cm*1./np.max(cm)
            
        d = np.r_[list(rgb_color)*cm.shape[0]*cm.shape[1]].reshape(cm.shape[0], cm.shape[1],-1)
        d = np.insert(d, 3, (cm*255).astype(int), axis=2)
        fname = dest_dir+"/"+self.get_id()+".png"
        imsave(fname, d)            

    def binarize(self, vmin, vmax=None):
        """
        sets to one img values in [vmin,vmax] (both inclusive) and to zero all others.
        """
        vmax = vmin if vmax is None else vmax
        img = self.get_img()
        r = np.zeros(img.shape)
        r[ (img>=vmin) & (img<=vmax)] = 1
        r = GeneratedTile.from_image(r, self)
        r.format="png"
        return r

    def to_kmz_element(self, color="80ffffff"):
        kmz_element="""
                <GroundOverlay>
                        <name>%s</name>
                        <color>%s</color>
                        <Icon>
                                <href>%s</href>
                                <viewBoundScale>0.75</viewBoundScale>
                        </Icon>
                        <LatLonBox>
                                <north>%f</north>
                                <south>%f</south>
                                <east>%f</east>
                                <west>%f</west>
                        </LatLonBox>
                </GroundOverlay>
        """
        e,n = self.bbox["NE"]
        w,s = self.bbox["SW"]
        name = "%f,%f"%(self.center_lon, self.center_lat)
        fname = "files/"+self.get_id()+".png"
        return kmz_element%(name, color, fname,n,s,e,w)    

    def __repr__(self):

        mx,my = self.get_size_meters()
        rx,ry = self.get_resolution()

        s =    "id:         "+self.get_id()
        s += "\nclass:      "+self.__class__.__name__
        s += "\ncenter:     lat %s, lon %s"%(str(self.center_lat), str(self.center_lon))
        s += "\nzoom:       %d"%self.zoom
        s += "\nsize:       %dx%d px"%(self.w,self.h)+ "   "+get_distance_str(mx)+" x "+get_distance_str(my)
        s += "\nresolution: %.2f m/pixel X %.2f m/pixel"%(rx, ry)
        s += "\narea:       " + get_area_str(self.get_area())
        s += "\nbbox:\n"+"\n".join(["    "+k+": "+str(v) for k,v in self.bbox.iteritems()])
        return s

class SampleTile(Tile):
    def __init__(self, img):

        h, w = img.shape[0], img.shape[1]
        super(SampleTile, self).__init__(center_lat=0, center_lon=0, 
                                         zoom=1, pixel_size=(w,h), format="jpg",
                                         savedir=None, verbose=0, use_file_cache=False)
        self.img = img

    def get_id(self):
        return "0"


class GeneratedTile(Tile):

    def __init__(self):
        pass

    @classmethod
    def from_image(cls, image, generated_fromtile, savedir=None):
        r = cls()
        r.properties = {k:v for k,v in  generated_fromtile.to_geoseries().iteritems()}
        r.properties["generated_from"] = r.properties["class_name"]
        r.properties["class_name"] = r.__class__.__name__
        r.img = image
        r.savedir = savedir
        r.verbose = generated_fromtile.verbose
        for k,v in r.properties.iteritems():
            r.__setattr__(k,v)
        r.id = generated_fromtile.get_id()
        r.compute_properties()
        return r

    @classmethod
    def from_geoseries(cls, gs, savedir, verbose=0):

        assert gs["class_name"] == cls.__name__, "geoseries must have class_name='%s'"%cls.__name__

        r = cls()
        r.properties = {k:v for k,v in  gs.iteritems()}
        for k,v in r.properties.iteritems():
            r.__setattr__(k,v)
        r.savedir = savedir
        r.use_file_cache = True
        r.verbose = verbose
        r.compute_properties()
        r.img = None
        return r    

    def get_id(self):
        return self.id

    def to_geoseries(self):
        r = super(GeneratedTile, self).to_geoseries()
        extra = gpd.GeoSeries([self.generated_from],index=["generated_from"])
        return pd.concat((r,extra))

    def get_url(self):
        raise "Generated Tiles have no URL, make sure this tile is reconstructed with reference to the appropriate folder containing imgs"

    def __repr__(self):
        s = super(GeneratedTile, self).__repr__()
        s += "\ngenerated_from:  %s"%self.generated_from
        return s

class SegmentationProbabilitiesTile(GeneratedTile):

    @classmethod
    def from_probabilities(cls, source_tile, prediction_probabilities):
        """
        source_tile: the tile for which the predictions are made
        prediction_probabilities: a matrix of size w,h,n where w,h are the source_tile 
                                  image dimensions and n is the number of classes
        """

        assert source_tile.h == prediction_probabilities.shape[0], "tile and probabilities must have the same w,h dimensions"
        assert source_tile.w == prediction_probabilities.shape[1], "tile and probabilities must have the same w,h dimensions"
        
        r = copy(source_tile)
        r.img = prediction_probabilities
        r.format = "pklz"
        r.num_classes = prediction_probabilities.shape[2]
        r.compute_properties()

        s = cls()#r.center_lat, r.center_lon, r.zoom, (r.w, r.h))
        instance_attrs = [[i, type(eval("r."+i))] for i in dir(r) if not i.startswith("__") and str(type(eval("r."+i)))!="<type 'instancemethod'>"]
        for k,v in instance_attrs:
            s.__setattr__(k, r.__getattribute__(k))
        s.compute_properties()
        s.img = prediction_probabilities
        return s

    def before_saving_pickle(self, data):
        return (data*(2**8)).astype("uint8")

    def after_loading_pickle(self, data):
        return data.astype("float")/(2**8)

    def get_img(self):
        img = super(SegmentationProbabilitiesTile, self).get_img()
        return img

    def get_maxclass_prediction(self):
        r = GeneratedTile.from_image(self.get_img().argmax(axis=2), self)
        r.format="png"
        return r

    def get_classmap(self, class_number, binary=False, threshold=None):
        # must retrieve the image to know the number of classes
        img = self.get_img()[:,:,class_number].copy()

        assert not(binary and threshold is None), "when using binary must specify threshold"
        assert class_number <= self.get_num_classes(), "this tile has only %d classes"%self.get_num_classes()

        
        if threshold is not None:
            img[img<threshold] = 0
            
        if binary:
            img[img!=0] = 1
            
        r = GeneratedTile.from_image(img, self)
        r.format = "png"
        return r

    def get_false_positive_map(self, label_tile, class_number, threshold):
        lab = label_tile.get_img()

        assert len(lab.shape)==2, "label must be single channel"


        lab = lab==class_number
        d   = self.get_classmap(class_number, binary=True, threshold=threshold).get_img().copy()
        d[lab==1]=0
        r = GeneratedTile.from_image(d, self)
        r.format="png"
        return r

    def plot_class_probabilities(self):
        img = self.get_img()
        ncols = 4
        nrows = int(np.ceil(self.get_num_classes()*1./ncols))
        plt.figure(figsize=(ncols*4, nrows*4))
        img = self.get_img()
        for i in range(self.get_num_classes()):
            plt.subplot(nrows, ncols, i+1)
            plt.imshow(self.get_classmap(i).get_img())
            plt.axis("off")
            plt.title("CLASS %d"%i)

    def maxprob_prediction_metrics(self, label_tile):
        l = label_tile.get_img()
        assert len(l.shape)==2, "label must be a single channel image (2 dims matrix)"
        
        p = self.get_maxclass_prediction().get_img()
        
        r = []
        for i in range(self.get_num_classes()):
            cl = np.zeros(l.shape)
            cl[l==i]=1
            
            cp = np.zeros(p.shape)
            cp[p==i]=1
        
            r.append({"class": i, 
                    "px_accuracy": np.sum((cp==1)&(cl==1))*1./np.sum(cl==1), 
                    "n_pixels":int(np.sum(cl)),  
                    "pct_area": np.sum(cl)*1./np.product(cl.shape),
                    "iou": np.sum((cp==1)&(cl==1))*1./np.sum((cp==1)|(cl==1))})
        
        return pd.DataFrame(r)

    def get_num_classes(self):
        if not hasattr(self, "num_classes"):
            self.num_classes = self.get_img().shape[2]
        return self.num_classes

    def threshold_prediction_metrics(self, label_tile, threshold):
        l = label_tile.get_img()
        assert len(l.shape)==2, "label must be a single channel image (2 dims matrix)"

        r = []
        for i in range(self.get_num_classes()):
            cl = np.zeros(l.shape)
            cl[l==i]=1

            cp = self.get_classmap(class_number=i, binary=True, threshold=threshold).get_img()

            r.append({"class": i, 
                    "px_accuracy": np.sum((cp==1)&(cl==1))*1./np.sum(cl==1), 
                    "n_pixels":int(np.sum(cl)),  
                    "pct_area": np.sum(cl)*1./np.product(cl.shape),
                    "iou": np.sum((cp==1)&(cl==1))*1./np.sum((cp==1)|(cl==1))})

        return pd.DataFrame(r)
        
class GMaps_StaticAPI_Tile(Tile):

    def __init__(self, maptype="roadmap", use_descriptive_id=False, apikey=None,**kwargs):

        super(GMaps_StaticAPI_Tile, self).__init__(**kwargs)

        self.maptype = maptype
        self.apikey  = apikey
        self.google_logo_height = 20 # to remove google logo
        self.use_descriptive_id = use_descriptive_id
        self.extra_id = ""

    def get_url(self, apikey=None):
        self.apikey = apikey if apikey is not None else self.apikey
        assert self.apikey is not None, "must set apikey"
        w,h = self.w, self.h
        s = "https://maps.googleapis.com/maps/api/staticmap?center=%s,%s&zoom=%s&size=%sx%s&maptype=%s&key=%s"% \
            (str(self.center_lat),str(self.center_lon),str(self.zoom), str(w),
             str(h+self.google_logo_height), self.maptype, self.apikey)
        return s

    def to_geoseries(self):
        r = super(GMaps_StaticAPI_Tile, self).to_geoseries()
        extra = gpd.GeoSeries([self.maptype, int(self.use_descriptive_id)],index=["maptype", "use_descriptive_id"])
        return pd.concat((r,extra))

    @classmethod
    def __from_geoseries__(cls, gs, apikey=None):
        """
        this method is to be called from the parent from_geoseries.
        returns a dictionary extracted from gs (a Pandas GeoSeries)
        to be appended to the constructor call.
        """
        return {"maptype": gs["maptype"], "use_descriptive_id": gs["use_descriptive_id"], "apikey": apikey}


    def get_id(self):
        r = "gmaps_%s_%s_zoom_%s_%sx%s_%s%s"%\
                        (str(self.center_lat),str(self.center_lon),str(self.zoom), str(self.w), str(self.h), self.maptype, self.extra_id)

        if self.use_descriptive_id:
            return r
        return hashlib.sha224(r).hexdigest()

    def get_img(self, **kwargs):
        """
            removes google logo
        """
        img = super(GMaps_StaticAPI_Tile, self).get_img(**kwargs)
        self.img = img[:self.h, :self.w]
        return self.img

    def get_next_east(self):
        lon,lat = self.get_pixel_latlon(self.w*3/2, self.h/2)
        return self.clone_with_properties({"center_lat": lat, "center_lon":lon})

    def get_next_west(self):
        lon,lat = self.get_pixel_latlon(-self.w/2, self.h/2)
        return self.clone_with_properties({"center_lat": lat, "center_lon":lon})

    def get_next_south(self):
        lon,lat = self.get_pixel_latlon(self.w/2, self.h*3/2)
        return self.clone_with_properties({"center_lat": lat, "center_lon":lon})

    def get_next_north(self):
        lon,lat = self.get_pixel_latlon(self.w/2, -self.h/2)
        return self.clone_with_properties({"center_lat": lat, "center_lon":lon})

    def __repr__(self):
        s = super(GMaps_StaticAPI_Tile, self).__repr__()
        s += "\nmaptype:            %s"%self.maptype
        s += "\nuse_descriptive_id: %s"%self.use_descriptive_id
        return s

    @classmethod
    def from_local_file(cls, fname, use_descriptive_id=True, preload_img=True, format="jpg"):
        """
        only when using descriptive ids so that metadata is encoded in the file name
        """

        p = re.compile('(\S*)gmaps_(\S+)_(\S+)_zoom_(\d+)_(\d+)x(\d+)_([^_]+)(_\S*).'+format).match(fname)
        if p is None:
            return None
        dir, lat, lon, zoom, px, py, mtype, extra_id = p.groups()
        g = cls(center_lat = np.float64(lat), center_lon = np.float64(lon), 
                zoom = int(zoom), pixel_size = (int(px), int(py)),
                maptype=mtype, format = "jpg", savedir=dir, use_file_cache=True)
        g.use_descriptive_id = use_descriptive_id
        g.extra_id = extra_id
        g.format = format
        if preload_img:
            g.get_img()
        if len(dir)!=0:
            g.savedir=dir
            g.use_file_cache=True
        return g

from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection

def katana_polygon_split(geometry, threshold, count=0):
    """Split a Polygon into two parts across it's shortest dimension"""
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if max(width, height) <= threshold or count == 250:
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geometry]
    if height >= width:
        # split left to right
        a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/2)
        b = box(bounds[0], bounds[1]+height/2, bounds[2], bounds[3])
    else:
        # split top to bottom
        a = box(bounds[0], bounds[1], bounds[0]+width/2, bounds[3])
        b = box(bounds[0]+width/2, bounds[1], bounds[2], bounds[3])
    result = []
    for d in (a, b,):
        c = geometry.intersection(d)
        if not isinstance(c, GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (Polygon, MultiPolygon)):
                result.extend(katana_polygon_split(e, threshold, count+1))
    if count > 0:
        return result
    # convert multipart into singlepart
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result


def read_googlemaps_kml(kml_file):
    from fastkml import kml
    with open(kml_file, 'rt') as f:
        area = kml.KML()
        area.from_string(f.read())
    pm = list(list(list(area.features())[0].features())[0].features())[0]
    el = pm.etree_element()
    pol = el[3]
    coords = list(pol.getiterator())[3]
    coords = np.r_[[np.r_[i.split(",")[:2]].astype(float) for i in coords.text.split(" ")]]

    area = sh.geometry.Polygon(coords)    
    return area

class XGoogleMaps_Static_Image:
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
                print ("loading from file",f)
            return Image.open(f)

        url = self.get_url(apikey)
        if self.verbose>0:
            print ("retrieving", url)
        file = BytesIO(urlopen(url).read())
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
                print ("skipping existing", fname)
            return
        if self.verbose>0:
            print ("saving to", fname)
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
        print a
        return "%.2f m2"%a if a<10 else "%d m2"%a if a<=1e4 else "%.3f km2"%(a/1e6) if a<1e5 else "%.2f km2"%(a/1e6)

    def __repr__(self):
        b = self.get_bbox()
        s =    "center:  lat %s, lon %s"%(str(self.lat), str(self.lon))
        s += "\nzoom:    %d"%self.zoom
        s += "\nsize:    %dx%d px"%(self.w,self.h-self.google_logo_height)
        s += "\nmaptype: %s"%self.maptype
        s += "\narea:    " + self.get_area_str()
        s += "\nbbox:\n"+"\n".join(["    "+k+": "+str(v) for k,v in b.iteritems()])
        return s


class XGoogleMaps_Static_Mosaic:

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


class XGoogleMaps_Shapefile_Layer:

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
        print ("generating polygons")
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
                                       single_channel_map=None, format="jpg",
                                       min_classes_per_img=None, use_255_range_in_single_channel=True):
        self.set_color_function(color_func)

        lname = target_dir+"/"+(".".join(gmaps_img.get_fname().split(".")[:-1])+"_%s%s.%s"%(self.layer_name, suffix, format)).split("/")[-1]
        if os.path.isfile(lname):
            if verbose:
                print ("skipping existing", lname)
            return True

        bbox = sh.geometry.Polygon(gmaps_img.get_polygon())
        si = self.shapefile.iloc[[bbox.intersects(p) for p in self.mpols]]

        pols = [get_shapely_multipolygon([i[:,::-1] for i in p]) for p in si.latlon_coords.values]
        fcols = [self.color_function(i) for _, i in si.iterrows()]
        cols = [i[0] if type(i)==tuple else i for i in fcols]
        alphas = [i[1] if type(i)==tuple else 1. for i in fcols]

        if len(pols)==0:
            if verbose:
                print ("no intersecting polygons in shapefile for %s"%gmaps_img.get_fname())
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
        used_colors = []
        for i in range(len(pols)):
            pol = pols[i]
            pol = pol.intersection(bbox)
            kpol = sh.affinity.translate(pol, xoff=-xmin, yoff=-ymin)
            kpol = sh.affinity.scale(kpol, xfact=w/(xmax-xmin), yfact=h/(ymax-ymin), origin=(0,0))
            bpol = bpol.difference(kpol)
            ax.add_patch(descartes.PolygonPatch(kpol, color=cols[i], lw=0, alpha=alphas[i]))
            used_colors.append(str(cols[i]))

        used_colors = np.unique(used_colors)

        ## if not enough classes skip it
        if verbose:
            print ("found %d classes (min is %d) in %s"%(len(used_colors), min_classes_per_img, gmaps_img.get_fname()))
        if min_classes_per_img is not None and len(used_colors)< min_classes_per_img:
            if verbose:
                print ("not enough classes (%d found) in %s"%(len(used_colors), gmaps_img.get_fname()))
            plt.close()
            return False

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
            print ("saving to", lname)
        fig.savefig(lname)
        plt.close()

        if single_channel_map is not None:
            k = imread(lname)
            k = convert_label_to_single_channel(k, single_channel_map, use_255_range_in_single_channel)
            k = most_common_neighbour(k, (6,6))
            imsave(lname, k)

        return True

def convert_label_to_single_channel(multi_channel_label_img, channel_map, use_255_range=True):
    assert len(multi_channel_label_img.shape)==3, "img must be multichannel"

    if multi_channel_label_img.shape[2]==4:
        multi_channel_label_img = multi_channel_label_img[:,:,:3]
    r = (np.r_[[np.abs(multi_channel_label_img-i).sum(axis=2) for i in channel_map]].argmin(axis=0)).astype(np.uint8)
    if use_255_range:
        lmap = len(channel_map)
        slmap = {i: int((i*255./lmap)) for i in range(lmap)}
        # converts single channel image to a 0-255 pixel value range
        r = np.r_[[slmap[i] for i in r.flatten()]].reshape(r.shape)
    return r

def convert_label_to_multi_channel(single_channel_label_img, channel_map, use_255_range=True):
    s = single_channel_label_img.shape
    sf = single_channel_label_img.flatten()
    lmap = len(channel_map)
    if use_255_range:
        # converts back 0-255 single channel to 0-num_classes
        slmap = {int((i*255./lmap)):i for i in range(lmap)}
    else:
        slmap = {i:i for i in range(lmap)}

    kt = np.r_[[slmap[i] for i in sf]].reshape(single_channel_label_img.shape)
    return np.r_[[channel_map[i] for i in kt.flatten()]].reshape( list(s)+[3] ).astype(np.uint8)



def show_channel_map(channel_map, matplotlib_colormap=plt.cm.jet, use_255_range=True):
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
    ks = convert_label_to_single_channel(k, channel_map, use_255_range)
    plt.title("single channel, levels="+str(np.unique(ks.flatten())))
    plt.imshow(ks, cmap=matplotlib_colormap)
    plt.axis("off")

    plt.figure(figsize=(20,2))
    plt.title("recovered multichannel")
    kk = convert_label_to_multi_channel(ks, channel_map, use_255_range)
    plt.imshow(kk, cmap=matplotlib_colormap)
    plt.axis("off")
