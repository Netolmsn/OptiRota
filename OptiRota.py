import pandas as pd
import numpy as np
import plotly.express as px
from math import radians, sin, cos, sqrt, atan2

import plotly.io as pio
pio.renderers.default = "vscode" 

data = {
    'ID': ['C', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
    'Lat': [-23.5505, -23.545, -23.560, -23.570, -23.535, -23.555, -23.540],
    'Lon': [-46.6333, -46.640, -46.625, -46.638, -46.650, -46.615, -46.630],
    'Demanda': [0, 5, 10, 3, 7, 2, 8], 
    'T_Inicio_h': [8, 9, 10, 9, 11, 10, 8],
    'T_Fim_h': [17, 12, 11, 10, 13, 12, 9]
}
df_clientes = pd.DataFrame(data)

depot = df_clientes[df_clientes['ID'] == 'C'].iloc[0]

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em quilômetros entre dois pontos 
    (lat, lon) na superfície da Terra usando a fórmula de Haversine.
    """
    R = 6371

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

n_clientes = len(df_clientes)
dist_matrix = np.zeros((n_clientes, n_clientes))

for i in range(n_clientes):
    for j in range(n_clientes):
        dist_matrix[i, j] = haversine(
            df_clientes.iloc[i]['Lat'], df_clientes.iloc[i]['Lon'],
            df_clientes.iloc[j]['Lat'], df_clientes.iloc[j]['Lon']
        )

fig = px.scatter_mapbox(
    df_clientes,
    lat="Lat",
    lon="Lon",
    hover_name="ID",
    color="Demanda",
    size="Demanda",
    color_continuous_scale=px.colors.cyclical.IceFire,
    zoom=11,
    title="Localização dos Clientes e Depósito",
    mapbox_style="carto-positron"
)

fig.update_traces(marker=dict(size=10, opacity=0.8),
                  selector=dict(mode='markers'))

fig.add_trace(
    px.scatter_mapbox(
        depot.to_frame().T, 
        lat="Lat", 
        lon="Lon", 
        hover_name="ID"
    ).update_traces(
        marker=dict(size=15, symbol='star', color='red'),
        name="Depósito"
    ).data[0]
)

fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    mapbox_accesstoken="SEU_MAPBOX_TOKEN_AQUI"
)

fig.show()

fig.write_html("clientes_inicial.html")
print("\nArquivo 'clientes_inicial.html' gerado com sucesso!")