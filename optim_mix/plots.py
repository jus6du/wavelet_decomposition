import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

colors_dict = {
    'Wind': 'steelblue',        
    'PV': 'gold',
    'Discharge': 'orangered',    
    'SOC': 'darkgreen',           
    'Charge': 'purple',
    'Consumption': 'green',          
    'Dispatchable': 'crimson',       
    'Curtailment': 'cyan'    
}
def plot_betas_heatmap(df, name, year , ndpd,
                      cmin = None, cmax= None, ccenter = None, show = False, save_fig = None):
    '''
    Here we plot the absolute value of betas dataframe
    '''

    # Plot aesthetic settings
    sns.set()
    sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.})
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_palette("colorblind")  # set colors palettte

    plt.rc('font', family='serif')
    # Plot
    # Tick axis definition
    time_scales = ['0.75 ', '1.5', '3 ', '6 ', '12', 'day', '42', '84', 'week', '273.75', '547.5',
            '1095', '2190', '4380', 'year']
    y = [0.5, 1.5 ,2.5 ,3.5 ,4.5 ,5.5 ,6.5 ,7.5 ,8.5 ,9.5 ,10.5,11.5 ,12.5 ,13.5 ,14.5 ]
    #
    time = ["January, 1"] + [str(x) for x in range(30,360,30)]+["December, 31"]
    x = [x*ndpd for x in range(0,360,30)]+[365.*ndpd]
    #
    # Figure settings
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    # Z = (np.absolute(df)).transpose()
    Z = df.transpose()
    ax = sns.heatmap( Z ,
    #                norm=LogNorm(vmin=Z.min().max(), vmax=Z.max().min()),
    #                cmap = "YlOrRd",
                   cmap='coolwarm',
                   center = ccenter,
    #                robust=True, # is supposed to improve contrast
                   vmin= cmin,
                   vmax= cmax,
                   cbar = False
    )
    cbar = ax.figure.colorbar(ax.collections[0])
    # cbar.set_ticks([0.005, 0.085])
    # cbar.set_ticklabels(["Low", "High"])
    cbar.set_label('Charge - Discharge power ')
    ax.set_yticks(y)
    ax.set_yticklabels(time_scales, minor=False, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(time, minor=False, rotation=0)
    plt.ylabel('Storage time scale (hours)', fontsize=20, fontweight='bold')
    plt.xlabel('Time (Days)', fontsize=20, fontweight='bold')
    plt.title('Wavelet transform of the signal "'+ name +'" in ' + year , fontsize=20, fontweight='bold')
    plt.ylim(15,0)

    if save_fig:
        plt.savefig(save_fig)

    if show :
        plt.show(block=False)
        plt.pause(3)
    
    plt.close()


def plot_ts_optim(list_arrays, list_names, country_name, colors_dict = colors_dict, savefig = False):
    fig = go.Figure()
    signal_length = len(list_arrays[0])
    for i in range(len(list_arrays)):
        fig.add_trace(go.Scatter(x=list(range(signal_length)), y=list_arrays[i], mode='lines', name=list_names[i],marker=dict(color=colors_dict[list_names[i]])))

    fig.update_layout(
        xaxis_title = 'Time (hour)',
        yaxis_title='Power (normalized)',
        title=f'Optimization results {country_name}'
    )
    if savefig: 
        # Save figure
        fig.write_html(f"figures/{country_name}_optim_full_ts.html")

    # Show figure
    fig.show()
    
    return

def plot_stack_production(PV, wind, dispatch, country_name, colors_dict = colors_dict, savefig=False):
    # Plot the stacked production
    fig = go.Figure()
    signal_length = len(PV)
        # Add the stacked production trace
    fig.add_trace(go.Scatter(x=list(range(signal_length)), y=PV, mode='lines', name='PV', stackgroup = 'one',marker=dict(color=colors_dict['PV'])))
    fig.add_trace(go.Scatter(x=list(range(signal_length)), y=wind, mode='lines', name='Wind', stackgroup = 'one',marker=dict(color=colors_dict['Wind'])))
    fig.add_trace(go.Scatter(x=list(range(signal_length)), y=dispatch, mode='lines', name='Dispatchable', stackgroup = 'one', marker=dict(color=colors_dict['Dispatchable'])))

    if savefig:
        fig.write_html(f"figures/{country_name}_optim_stacked_prod_ts.html")

    fig.show()
    return

def plot_pie_energy(list_energy, country_name, names = ['Wind', 'PV', 'Dispatchable'], colors_dict = colors_dict, savefig=False):
    # Map the colors to the labels maintaining the order
    colors = [colors_dict[label] for label in names]

    fig = go.Figure(data=[go.Pie(labels=names, values=list_energy, marker=dict(colors=colors), textinfo='label+percent')])

    if savefig:
        fig.write_html(f"figures/{country_name}_optim_pie_chart.html")

    fig.show()
    return

def plot_storage(charge, discharge, SOC, country_name, colors_dict = colors_dict, savefig = False):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Charge/Discharge", "State of Charge (SOC)"))

    signal_length = len(charge)
    fig.add_trace(go.Scatter(x=list(range(signal_length)), y=charge, mode='lines',marker=dict(color=colors_dict['Charge']), name='Charge'),  row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(signal_length)), y=discharge, mode='lines',marker=dict(color=colors_dict['Discharge']), name='Disharge'),  row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(signal_length)), y=SOC, mode='lines', marker=dict(color=colors_dict['SOC']),name='SOC'), row=2, col=1)
    
    fig.update_xaxes(title_text='Time (hour)', row=2, col=1)
    fig.update_yaxes(title_text='Power (normalized)', row=1, col=1)
    fig.update_yaxes(title_text='State of Charge', row=2, col=1)

    fig.update_layout(
        title=f'Charge/Discharge and State of Charge (SOC) for {country_name}',
        height=600
    )
    if savefig: 
        # Save figure
        fig.write_html(f"figures/{country_name}_optim_stock_ts.html")

    # Show figure
    fig.show()
    return

def get_energy_tot(list_arrays):
    
    return


def scatter_plot(data, indic1, indic2, title = None, label1 = None, label2 = None, savefig = False):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[indic1], data[indic2], color='blue')
    for country in data.index:
         plt.textdata(data.loc[country,indic1], data.loc[country, indic2], country, fontsize=9)

    # Ajouter des titres et des labels
    plt.title(title)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plt.show()
    return

def interactive_scatter_plot(data, indic1, indic2, title=None, label1=None, label2=None):
    # Créer le graphique de dispersion interactif
    fig = px.scatter(
        data, x=indic1, y=indic2, hover_name=data.index, 
        labels={indic1: label1, indic2: label2}, 
        title=title
    )

    # Mettre à jour la mise en page
    fig.update_layout(
        hovermode='closest',
        xaxis_title=label1,
        yaxis_title=label2,
        title=title
    )

    # Sauvegarder le graphique en HTML
    fig.write_html(f'{indic1}_{indic2}_interactive_scatter_plot.html')

    # Afficher le graphique dans le navigateur
    fig.show()


def scatter_plot(data, indic1, indic2, title = None, label1 = None, label2 = None, savefig = False):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[indic1], data[indic2], color='blue')
    for country in data.index:
        #  print(data.loc[country,indic1])
         plt.text(data.loc[country,indic1], data.loc[country, indic2], country, fontsize=9)

   
    if title:
        plt.title(title)
    if label1:
        plt.xlabel(label1)
    if label2:
        plt.ylabel(label2)
    plt.grid(True)
    if savefig:
        plt.savefig(f'scatterplot_{indic1}_{indic2}')
    plt.show()
    return