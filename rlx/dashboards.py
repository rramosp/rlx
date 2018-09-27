def create_html(out_html_file, html_template, bokeh_components={}, matplotlib_components={}, html_components={}):
    from bokeh import embed as be
    import os

    if not os.path.isfile(html_template):
        print "template", html_template, "does not exist"
        return

    html = open(html_template, 'r').read()  # .replace('\n', '')

    if len(bokeh_components) > 0:
        script, divs = be.components(bokeh_components.values(), wrap_plot_info=False)
        html = html.replace("__BOKEH_SCRIPT__", script)
        for i in range(len(bokeh_components)):
            print "generating bokeh", i
            div = '<table><tr><td><div class="bk-root"><div class="bk-plotdiv" id="' + divs[i].elementid + '"></div></div></td></tr></table>\n'
            html = html.replace("__" + bokeh_components.keys()[i] + "__", div)

    for k in matplotlib_components.keys():
        print "generating matplotlib", k
        html = html.replace("__" + k + "__", get_img_tag(matplotlib_components[k]))

    for k in html_components.keys():
        html = html.replace("__" + k + "__", html_components[k])

    fh = open(out_html_file, "w")
    fh.write(html)
    fh.close()


def get_img_tag(fig, width=None, height=None, class_tag=None):
    import os

    fig.savefig("aa.png", transparent=True, bbox_inches='tight', pad_inches=0)
    data_uri = open('aa.png', 'rb').read().encode('base64').replace('\n', '')
    wstr = "" if width is None else "width=%s"%str(width)
    hstr = "" if height is None else "height=%s"%str(height)
    cstr = "" if class_tag is None else "class='%s'"%str(class_tag)
    img_tag = ('<img %s %s %s src="data:image/png;base64,{0}">'%(cstr, wstr, hstr)).format(data_uri)
    os.remove("aa.png")
    return img_tag


javascript_funcs = """
function wait_element(){

    var retry = 10;
    var timeout = 250;
    window.FX_waiting = true;

    if(typeof window.FX_content !== "undefined"){
        window.FX_waiting = false;
        window["FX_success"](window.FX_content);
    }
    else{
        if (window.FX_count==retry) {
            window.FX_waiting = false;
            window["FX_fail"]("RETR: max retries reached")
        } else {
            setTimeout(wait_element, timeout);
            window.FX_count = window.FX_count + 1;
        }
    }
}

function meters_to_latlon(mlat, mlon) {
    var origin_shift = (Math.PI * 6378137 )
    var lons = mlon * 180/origin_shift
    var mlat = mlat * 180/origin_shift    
    var lats = Math.atan(Math.exp(mlat*Math.PI/180. ) ) * 360. / Math.PI - 90
    return [lats, lons]
}

function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function on_get_url(url, function_success, function_fail) {
    if (typeof window.FX_waiting !== "undefined"  && window.FX_waiting) {
        function_fail("BUSY: busy waiting for previous url ... discarding")
        return
    }

    delete window.FX_content;
    window.FX_success = function_success;
    window.FX_fail    = function_fail;
    window.FX_count   = 0;
    var s = document.createElement("script");
    s.src = url;
    element_id = ""+getRandomInt(1000,2000);
    s.id = element_id;
    document.body.appendChild(s);
    wait_element();
    var element = document.getElementById(element_id);
    element.parentNode.removeChild(element);
}


function center_map(x, y, map_width, fig) {
    // x,y:       lon, lat in meters
    // map_width: desired map width in meters
    // fig:       bokeh figure object containing the map

    r = fig.plot_width/fig.plot_height

    fig.x_range.start = x-map_width/2
    fig.x_range.end   = x+map_width/2

    fig.y_range.start = y-map_width/r/2
    fig.y_range.end   = y+map_width/r/2

}

"""
