import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots

colors_dict = {
    'Wind': 'rgb(255, 128, 128)',        
    'PV': 'rgb(128, 255, 128)',
    'Discharge': '#ff7f0e',    
    'SOC': '#2ca02c',           
    'Charge': '#9467bd',
    'Consumption': '#e377c2',          
    'Dispatchable': 'rgb(128, 128, 255)',       
    'Curtailment': '#17becf'    
}

def plot_ts_optim(list_arrays, list_names, country_name, savefig = False):
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

def plot_stack_production(PV, wind, dispatch, country_name, colors = colors_dict, savefig=False):
    # Plot the stacked production
    fig = go.Figure()
    signal_length = len(PV)
    # Add the stacked production trace
    fig.add_trace(go.Scatter(x=list(range(signal_length)), y=PV, mode='lines', name='PV', stackgroup = 'one',marker=dict(color=colors['PV'])))
    fig.add_trace(go.Scatter(x=list(range(signal_length)), y=wind, mode='lines', name='Wind', stackgroup = 'one',marker=dict(color=colors['Wind'])))
    fig.add_trace(go.Scatter(x=list(range(signal_length)), y=dispatch, mode='lines', name='Dispatchable', stackgroup = 'one', marker=dict(color=colors['Dispatchable'])))

    if savefig:
        fig.write_html(f"figures/{country_name}_optim_stacked_prod_ts.html")

    fig.show()
    return

def plot_pie_energy(list_energy, country_name, names = ['Wind', 'PV', 'Dispatchable'], colors_dict = colors_dict, savefig=False):
    fig = px.pie(names=names,
                values=list_energy,
                color_discrete_map=colors_dict,  # Utiliser color_discrete_sequence pour définir les couleurs
                title='Energy Production')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    # Write HTML file
    if savefig:
        fig.write_html(f"figures/{country_name}_optim_pie_chart.html")

    # Show the plot
    fig.show()
    return

def plot_storage(charge, discharge, SOC, country_name, colors = colors_dict, savefig = False):
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