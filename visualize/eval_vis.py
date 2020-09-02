import numpy as np

from bokeh.plotting import figure, show, output_file

#40, 60, 80, 100
#7567, 14917, 28393, 41371
#x = np.linspace(0.1, 5, 80)

perf_x = [40, 60, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180]
base_perf_y = [2223, 3427, 4651, 4837, 5765, 6651, 7213, 6564, 20386, 10025, ]
sug_perf_y = [2364, 3652, 4625, 4780, 5623, 5821, 6763, 6713, 7671, 8563, 9015, ]

p = figure(x_range=(20, 140), y_range=(0, 22000),
           background_fill_color="#fafafa", plot_width=400, plot_height=300)
p.line(perf_x, base_perf_y, line_color='orange', legend="baseline", line_width=2)
p.square(perf_x, base_perf_y, fill_color=None, legend="baseline", line_color='orange', size=6)

p.line(perf_x, sug_perf_y, line_color='green', legend="suggested", line_width=2)
p.circle(perf_x, sug_perf_y, fill_color=None, legend="suggested", line_color="green", size=6)
p.yaxis.axis_label = "Total weighted completion time"


"""
time_x = [40, 60, 80, 100, 120]
base_time_y = [10.6, 13.6, 16.4, 18.5, 21.6, 25.1]
sug_time_y = [29.7, 42.2, 53.9, 65.6, 78.0]
sug_pred_time_y = [24.4, 27.1, 48.9, 60.6, 72.7]
sug_sche_time_y = [5.2, 5.1, 5.0, 5.0, 5.2]


p = figure(x_range=(20, 140), y_range=(0, 160),
           background_fill_color="#fafafa", plot_width=400, plot_height=300)
p.line(time_x, base_time_y, line_color='orange', legend="baseline", line_width=2)
p.square(time_x, base_time_y, fill_color=None, legend="baseline", line_color='orange', size=6)

p.line(time_x, sug_time_y, line_color='green', legend="suggested (total)",line_width=2)
p.circle(time_x, sug_time_y, fill_color=None, legend="suggested (total)", line_color="green", size=6)

p.line(time_x, sug_pred_time_y, line_color='green', line_dash='dotted', legend="suggested (for prediction)", line_width=2)
p.diamond(time_x, sug_pred_time_y, fill_color=None, legend="suggested (for prediction)", line_color="green", size=6)

p.line(time_x, sug_sche_time_y, line_color='green', line_dash='dotted', legend="suggested (for resource allocation)", line_width=2)
p.triangle(time_x, sug_sche_time_y, fill_color=None, legend="suggested (for resource allocation)", line_color="green", size=6)

p.yaxis.axis_label = "Time(secs)"
"""

"""
p = figure(title="log axis example", y_axis_type="log",
           x_range=(0, 5), y_range=(0.001, 10**22),
           background_fill_color="#fafafa")


p.line(x, np.sqrt(x), legend="y=sqrt(x)",
       line_color="tomato", line_dash="dashed")

p.line(x, x, legend="y=x")
p.circle(x, x, legend="y=x")

p.line(x, x**2, legend="y=x**2")
p.circle(x, x**2, legend="y=x**2",
         fill_color=None, line_color="olivedrab")

p.line(x, 10**x, legend="y=10^x",
       line_color="gold", line_width=2)

p.line(x, x**x, legend="y=x^x",
       line_dash="dotted", line_color="indigo", line_width=2)

p.line(x, 10**(x**2), legend="y=10^(x^2)",
       line_color="coral", line_dash="dotdash", line_width=2)

"""
p.xaxis.axis_label = "Number of instances"
#p.xaxis.label_text_font = "times"
#p.xaxis.label_text_font_style = "italic"


#p.yaxis.label_text_font = "times"
#p.yaxis.label_text_font_style = "italic"

p.legend.location = "top_left"
p.legend.label_text_font = "times"
p.legend.label_text_font_style = "italic"
output_file("logplot.html", title="log plot example")

show(p)