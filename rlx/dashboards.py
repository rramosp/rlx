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
            div = '<table><tr><td><div class="bk-root"><div class="bk-plotdiv" id="' + divs[i][
                "elementid"] + '"></div></div></td></tr></table>\n'
            html = html.replace("__" + bokeh_components.keys()[i] + "__", div)

    for k in matplotlib_components.keys():
        print "generating matplotlib", k
        html = html.replace("__" + k + "__", get_img_tag(matplotlib_components[k]))

    for k in html_components.keys():
        html = html.replace("__" + k + "__", html_components[k])

    fh = open(out_html_file, "w")
    fh.write(html)
    fh.close()

def get_img_tag(fig):
    import os

    fig.savefig("aa.png", transparent=True, bbox_inches='tight',pad_inches=0)
    data_uri = open('aa.png', 'rb').read().encode('base64').replace('\n', '')
    img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    os.remove("aa.png")
    return img_tag
