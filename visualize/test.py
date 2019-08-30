from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

#output_file("bar_nested_colormapped.html")

#perf_x = ['1/Mar/2012', '2/Mar/2012', '3/Mar/2012', '4/Mar/2012', '5/Mar/2012', '6/Mar/2012', '7/Mar/2012', '8/Mar/2012', '9/Mar/2012', '10/Mar/2012', '11/Mar/2012', '12/Mar/2012', '13/Mar/2012', '14/Mar/2012']
perf_x = ["{}. Mar.".format(x) for x in range(1,15,1)]
approaches = ['baseline', 'suggested']

data = {'perf_x' : perf_x,
        'baseline'   : [11139, 8598, 3168, 200, 5038, 7081, 4488, 5255, 7691, 2597, 225, 5414, 5599, 4396],
        'suggested'   : [5319, 3423, 1916, 73, 2548, 2644, 1532, 1927, 2600, 2058, 172, 2214, 1754, 1646]
        }

palette = ["#c9d9d3", "#718dbf"]

# this creates [ ("Apples", "2015"), ("Apples", "2016"), ("Apples", "2017"), ("Pears", "2015), ... ]
x = [ (date, approach) for date in perf_x for approach in approaches ]
counts = sum(zip(data['baseline'], data['suggested']), ()) # like an hstack

data = dict(x=x, counts=counts)
print(data)
source = ColumnDataSource(data=dict(x=x, counts=counts))

p = figure(x_range=FactorRange(*x), title="Fruit Counts by Year",
           toolbar_location=None, tools="")

p.vbar(x='x', top='counts', width=0.5, source=source, line_color="white",
       fill_color=factor_cmap('x', palette=palette, factors=approaches, start=1, end=2))

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None

show(p)