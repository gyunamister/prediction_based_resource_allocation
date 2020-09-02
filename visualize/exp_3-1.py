import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 14
num_instances = [270, 244, 197, 11, 182, 100, 141, 155, 188, 110, 14, 95, 105, 121]
perf_x = ["{}. Mar. \n ({})".format(x, num_instances[x-1]) for x in range(1,15,1)]
baseline = [11139, 8598, 3168, 200, 5038, 7081, 4488, 5255, 7691, 2597, 225, 5414, 5599, 4396]
suggested = [5319, 3423, 1916, 73, 2548, 2644, 1532, 1927, 2600, 2058, 172, 2214, 1754, 1646]


# create plot
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [4, 1]})
ax1.set_ylim(1000,12000)
ax2.set_ylim(0,250)

lower_yticks = np.arange(1000,12000,1000)
ax1.set_yticks(lower_yticks)
ax1.set_facecolor('#fafafa')

upper_yticks = np.arange(0,251,50)
ax2.set_yticks(upper_yticks)
ax2.set_facecolor('#fafafa')


index = np.arange(n_groups)
bar_width = 0.35
opacity = 1

rects1 = ax1.bar(index, baseline, bar_width, edgecolor='white',
alpha=opacity,
color='orange',
label=r"$\mathit{baseline}$")
rects1 = ax2.bar(index, baseline, bar_width, edgecolor='white',
alpha=opacity,
color='orange',
label=r"$\mathit{baseline}$")

rects2 = ax1.bar(index + bar_width, suggested, bar_width, edgecolor='white',
alpha=opacity,
color='green',
label=r"$\mathit{suggested}$")
rects2 = ax2.bar(index + bar_width, suggested, bar_width, edgecolor='white',
alpha=opacity,
color='green',
label=r"$\mathit{suggested}$")

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

#show only top x-axis tick marks on the top plot
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False) # hides the labels from the top x-axis

#show only bottom x-axis tick marks on the lower plot
ax2.xaxis.tick_bottom()


#squeeze plots closer
plt.subplots_adjust(hspace=0.1) #set to zero, if you want to join the two plots

ax2.set_xlabel(r"$\mathit{Date}$ $\mathit{(No. \ instances)}$")
ax1.set_ylabel(r"$\mathit{Total \  weighted \  completion \  time}$")
#plt.title('Scores by person')
plt.xticks(index + bar_width/2, perf_x)
ax1.legend()

#plt.tight_layout()
plt.show()