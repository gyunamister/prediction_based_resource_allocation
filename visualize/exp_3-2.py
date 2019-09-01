import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 14
num_instances = [270, 244, 197, 11, 182, 100, 141, 155, 188, 110, 14, 95, 105, 121]
perf_x = ["{}. Mar. \n ({})".format(x, num_instances[x-1]) for x in range(1,15,1)]
baseline = [28.7, 25.2, 17.9, 6.7, 14.9, 19.5, 14.5, 16.8, 22.1, 14.9, 8.9, 14.0, 14.8, 16.3]
sug_pred = [956.6, 844.2, 684.2, 21.2, 438.2, 554.2, 262.8, 458.1, 568.0, 322.9, 22.6, 231.9, 252.3, 372.2]
sug_allo = [10.7, 7.4, 7.0, 5.8, 6.6, 7.0, 6.5, 6.8, 7.0, 6.6, 6.2, 6.3, 6.6, 6.8]



# create plot
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [4, 1]})
ax1.set_ylim(100,1000)
ax2.set_ylim(0,50)

lower_yticks = np.arange(100,1000,100)
ax1.set_yticks(lower_yticks)
ax1.set_facecolor('#fafafa')

upper_yticks = np.arange(0,51,10)
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

pred_hatch = '-'*5
allo_hatch = '/'*5

rects2 = ax1.bar(index + bar_width, sug_allo, bar_width, hatch=pred_hatch, edgecolor='white',
alpha=opacity,
color='green',
label=r"$\mathit{suggested \ (for \ resource \ allocation)}$")
rects2 = ax1.bar(index + bar_width, sug_pred, bar_width, bottom=sug_allo, hatch=allo_hatch, edgecolor='white',
alpha=opacity,
color='green',
label=r"$\mathit{suggested \ (for \ prediction)}$")

rects2 = ax2.bar(index + bar_width, sug_pred, bar_width, hatch=pred_hatch, edgecolor='white',
alpha=opacity,
color='green',
label=r"$\mathit{suggested \ (for \ resource \ allocation)}$")
rects2 = ax2.bar(index + bar_width, sug_pred, bar_width, bottom=sug_allo, hatch=allo_hatch, edgecolor='white',
alpha=opacity,
color='green',
label=r"$\mathit{suggested \ (for \ prediction)}$")

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
ax1.set_ylabel(r"$\mathit{Time \  (secs)}$")
#plt.title('Scores by person')
plt.xticks(index + bar_width/2, perf_x)
ax1.legend()

#plt.tight_layout()
plt.show()