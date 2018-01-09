from rlx.utils import read_password_protected_zip, pbar, flatten
import rlx.geo as geo
from rlx import geo, utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry as sg
import gmaps
import utm


def get_intersecting_paths(vias, region):
    """
    vias: a dataframe with a columns 'coords', containing a list of tuples with lat/lon
    region: a 2d np array with two columns (lat/lon) defining a polygon

    returns:
        path: a dataframe which augments vias with a column named 'fragments'
              containing only the rows in vias which with fragments falling
              within the region
    """
    print "obtaining paths within region"
    b_clipped, b_intersects = geo.linespol_intersects(vias.coords.values, region, scale=1.1)
    b_full = vias.coords.iloc[b_intersects]
    print "paths intersecting", len(b_full)
    print "clipped paths", len(b_clipped)
    print "clipped path fragments", len(flatten(b_clipped))

    b_full = vias.iloc[b_intersects].copy()
    b_full["fragments"] = b_clipped
    return b_full


def set_sampling_locations(paths, sampling_distance):
    b_resampled = []
    for _, via in paths.iterrows():
        fragment_samples = []
        for k in via.fragments:
            # convert to meters to resample
            mlat, mlon = geo.latlon_to_meters(*np.r_[list(k.xy)])
            # resample (sampling_distance is in meters)
            r = geo.resample_path(np.r_[[mlat, mlon]].T,
                                  sampling_distance=sampling_distance)
            # convert back to lat/lon
            r = np.r_[[geo.meters_to_latlon(*r.T)]][0].T
            # append
            fragment_samples.append(sg.LineString(r))
        b_resampled.append(fragment_samples)
    paths["resampled_fragments"] = b_resampled

    # computes orientations
    f_orientations = []
    for f in paths.resampled_fragments:
        orientations = []
        for l in f:
            mcoords = np.r_[list(geo.latlon_to_meters(*np.r_[list(l.xy)]))].T
            diffs = mcoords[1:]-mcoords[:-1]
            o = [360 - geo.angle_between(i, [0, 1]) for i in diffs]
            # set orientation of last point to the one of the previous point
            o += [o[-1]]
            orientations.append(o)
        f_orientations.append(orientations)

    paths["resampled_orientations"] = f_orientations

    return paths


def show_fragments(paths, region, region_name):
    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    plt.plot(region[:, 0], region[:, 1], color="red", alpha=.2, lw=7)
    for _, via in paths.iterrows():
        coords = np.r_[via.coords]
        plt.plot(coords[:, 0], coords[:, 1], color="black")
    plt.axis("off")
    plt.title("intersecting roads with "+region_name)

    plt.subplot(122)
    plt.plot(region[:, 0], region[:, 1], color="red", alpha=.2, lw=7)
    for _, via in paths.iterrows():
        for fragment in via.fragments:
            plt.plot(*fragment.coords.xy, color="black")
    plt.axis("off")
    plt.title("clipped intersecting roads within "+region_name)


def show_google_map(paths, API_key, region):

    lines = []
    for f in pbar()(paths.fragments):
        flines = []
        for l in f:
            line_coords = np.r_[list(l.coords.xy)].T
            for i in range(len(line_coords)-1):
                flines.append(gmaps.Line(start=tuple(line_coords[i][::-1]),
                                         end=tuple(line_coords[i+1][::-1])))
        lines.append(flines)
    lines = flatten(lines)
    print "found", len(lines), "line segments"

    markers = []

    for o, f in pbar()(zip(flatten(paths.resampled_orientations),
                           flatten(paths.resampled_fragments))):
        coords = np.r_[list(f.xy)].T
        markers.append([gmaps.Marker((coords[i][1], coords[i][0]),
                        info_box_content=str(o[i])) for i in range(len(coords))])
    markers = flatten(markers)
    print "found", len(markers), "sampling locations"

    gmaps.configure(api_key=API_key)
    gmap_b = gmaps.Polygon([(i[1], i[0]) for i in region])
    fig = gmaps.figure(center=tuple(region.mean(axis=0)[::-1]), zoom_level=16)
    fig.add_layer(gmaps.drawing_layer(features=[gmap_b]+lines+markers))
    return fig


def streetview_http_request(API_key, lat, lon, heading, fov=60,
                            pitch=0, size="640x640"):

    params = (size, lat, lon, heading, pitch, fov, API_key)
    s = r"https://maps.googleapis.com/maps/api/streetview?size=%s&location=%f,%f&heading=%f&pitch=%f&fov=%f&key=%s"%(params)
    return s


def get_streetview_requests(b_full, API_key):
    sv_requests = []
    for o,f in pbar()(zip(flatten(b_full.resampled_orientations), flatten(b_full.resampled_fragments))):
        sv_item = []
        for i in range(len(o)):
            s_right = streetview_http_request(API_key, f.xy[1][i], f.xy[0][i], (o[i]+90)%360)
            s_front = streetview_http_request(API_key, f.xy[1][i], f.xy[0][i], o[i])
            s_left = streetview_http_request(API_key, f.xy[1][i], f.xy[0][i], (o[i]-90)%360)
            sv_item.append([o[i], f.xy[0][i], f.xy[1][i], s_front, s_right, s_left])
        sv_requests.append(pd.DataFrame(sv_item, columns=["orientation", "lon", "lat", "front", "right", "left"]))
    print "total number of street view requests", np.sum([len(i) for i in sv_requests])*3
    return sv_requests


def get_streetview_images(requests, dest_dir, API_key):
    from skimage.io import imsave
    import os
    skipped = 0
    for reqs in pbar()(requests):
        for _, req in reqs.iterrows():
            for k in ["front", "right", "left"]:
                fname = "sv_lat_%f_lon_%f_%s.jpg" % (req.lat, req.lon, k)
                if not os.path.isfile(dest_dir+fname):
                    img = utils.get_http_image(req[k])
                    if np.max(np.histogram(img.flatten())[0])<np.product(img.shape)*.9:
                        imsave(dest_dir+fname, img)
                    else:
                        skipped += 1
    print "skipped", skipped, "images with more than 90% of pixels with the same value"
