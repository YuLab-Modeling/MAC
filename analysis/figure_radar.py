import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 
# Set data
df = pd.DataFrame({
'group': ['M3G','eSCN','UMA','MS'],
"MAE'": [0.67, 0.27, 0.27, 0.15],
"RMSE'": [0.81, 0.33, 0.31, 0.22],
r'$\rho$': [-0.04, 0.33, 0.66, 0.73],
'Success rate': [16.92, 52.08, 39.30, 85.35],
"RMSD'": [0.52, 0.26, 0.24, 0.33],
r"$\overline{|\Delta d_{min}|}$'" : [0.48, 1.28, 0.96, 0.36],                     # r"$\Delta d_{min}$'" : [-0.03, 1.28, 0.92, 0.12],
"Time'": [7951.93, 7562.52, 2600.90, 2107.64],                                      # [7125.84, 6736.43, 1774.81, 1281.55],
"Binding site" : [46.26, 69.65, 80.10, 71.64],
})

for column in df.columns:
    if not column == 'group':
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
 
df["MAE'"] = 1 - df["MAE'"]
df["RMSE'"] = 1 - df["RMSE'"]
df["RMSD'"] = 1 - df["RMSD'"]
df[r"$\overline{|\Delta d_{min}|}$'"] = 1 - df[r"$\overline{|\Delta d_{min}|}$'"]
df["Time'"] = 1 - df["Time'"]

#df.to_csv('radar_delete.csv')

# ------- PART 1: Define a function that do a plot for one line of the dataset!
 
def make_spider( row, title, color):

    # number of variable
    categories=list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(1,4,row+1, polar=True, )

    #ax = plt.subplot(1,4,row+1, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='black', size=16)
    plt.tick_params('x', pad=14)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1.0"], color="black", size=12)
    plt.ylim(0,1)

    # Ind1
    values=df.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=24, color=color, y=1.2)

    
# ------- PART 2: Apply the function to all individuals
plt.figure(figsize=(20, 5))
 
# Create a color palette:
pastelDict = {
    'M3G' : '#1f77b4',  
    'eSCN' : '#ff7f0e', 
    'UMA' : '#2ca02c',
    'MS' : '#d62728',
}
 
# Loop to plot
for row in range(0, len(df.index)):
    make_spider( row=row, title=df['group'][row], color=pastelDict[df['group'][row]])

plt.subplots_adjust(wspace=0.6, hspace=0.3)

#plt.show()
plt.savefig("Figure_5_radar.png", dpi=300)