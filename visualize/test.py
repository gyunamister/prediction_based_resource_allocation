from bokeh.plotting import figure, show, output_file

output_file('hbar.html')

p = figure(plot_width=400, plot_height=400)
#p.hbar(y=[1]*3, height=0.5, left=[0,5,7], right=[1.2, 5.5, 7.7], color="navy")
p.hbar(y=[1], height=0.5, left=[0], right=[1.2], color='navy')
p.hbar(y=[1], height=0.5, left=[5], right=[5.2], color='navy')

show(p)