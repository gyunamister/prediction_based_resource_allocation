import numpy as np

from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot

#x = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
suggested = {'40': [2355, 2360, 2305, 2267, 2356, 2591, 2860, 2722, 2921, 3048, 2975], '60':[3499, 3488, 3594, 3621, 3634, 3934, 4477, 4505, 4451, 4387, 4387], '80':[4502, 4621, 4698, 4677, 4637, 5265, 5740, 6158, 6051, 6165, 6226], '100':[5571, 5617, 5684, 5839, 5774, 6497, 7339, 7753, 7404, 7767, 7327], '120':[7105, 7091, 7300, 7521, 7581, 8423, 9273, 9165, 9285, 9555, 9340], '140':[8048, 8055, 8460, 8507, 8507, 9556, 10682, 10987, 10538, 11267, 10671]}

plot_list = list()
for key in suggested:
	_min, _max = min(suggested[key]), max(suggested[key])
	_min_len = 0
	while _min > 10:
		_min /= 10
		_min_len += 1
	_max_len = 0
	while _max > 10:
		_max /= 10
		_max_len += 1
	#p = figure(y_range=(2000,12000), background_fill_color="#fafafa", plot_width=400, plot_height=300)
	p = figure(title='Number of Instances: {}'.format(key), background_fill_color="#fafafa", plot_width=300, plot_height=300)
	#inverse = list()
	#inverse = [inverse.insert(0, x) for x in suggested[key]]
	reverse = list(reversed(suggested[key]))
	p.line(x, reverse, line_color='olivedrab', line_width=2)
	p.circle(x, reverse, fill_color=None, line_color="olivedrab", size=6)
	plot_list.append(p)
	p.xaxis.axis_label = "Prediction accuracy (%)"
	p.yaxis.axis_label = "Total weighted completion time"
grid = gridplot(plot_list, ncols=3)

# show the results
show(grid)

#from bokeh.io import export_png

#export_png(grid, filename="grid.png")
