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

CAPACIDADE_VEICULO = 20
VELOCIDADE_MEDIA_KM_MIN = 1.0
TEMPO_SERVICO_MIN = 15

CUSTO_KM = 0.5
CUSTO_MIN_OPERACAO = 1.0
MULTA_ATRASO_POR_MIN = 5.0

START_DEPOT_MIN = df_clientes.iloc[0]['T_Inicio_h'] * 60
END_DEPOT_MIN = df_clientes.iloc[0]['T_Fim_h'] * 60

def avaliar_rota(rota):
    demanda_rota = sum(df_clientes.iloc[no]['Demanda'] for no in rota if no != 0)
    if demanda_rota > CAPACIDADE_VEICULO:
        return (999999, False, True)

    custo_total = 0
    tempo_atual_min = START_DEPOT_MIN

    for i in range(len(rota) - 1):
        origem = rota[i]
        destino = rota[i+1]

        distancia_km = dist_matrix[origem, destino]
        tempo_viagem_min = distancia_km / VELOCIDADE_MEDIA_KM_MIN

        tempo_chegada = tempo_atual_min + tempo_viagem_min

        custo_total += (distancia_km * CUSTO_KM) + (tempo_viagem_min * CUSTO_MIN_OPERACAO)

        if destino != 0:
            t_inicio_min = df_clientes.iloc[destino]['T_Inicio_h'] * 60
            t_fim_min = df_clientes.iloc[destino]['T_Fim_h'] * 60

            multa_atraso = 0
            tempo_espera = 0

            if tempo_chegada < t_inicio_min:
                tempo_espera = t_inicio_min - tempo_chegada
                tempo_inicio_servico = t_inicio_min

            elif tempo_chegada > t_fim_min:
                atraso_min = tempo_chegada - t_fim_min
                multa_atraso = atraso_min * MULTA_ATRASO_POR_MIN
                tempo_inicio_servico = tempo_chegada

            else:
                tempo_inicio_servico = tempo_chegada

            custo_total += multa_atraso

            tempo_saida = tempo_inicio_servico + TEMPO_SERVICO_MIN
            tempo_atual_min = tempo_saida

            if multa_atraso > 0:
                return (custo_total, True, False)

        else:
            if tempo_chegada > END_DEPOT_MIN:
                return (999999, True, False)

            tempo_atual_min = tempo_chegada

    return (custo_total, True, True)

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

# A visualização inicial do mapa foi omitida por ser grande, mas está logo acima.

CAPACIDADE_VEICULO = 20
VELOCIDADE_MEDIA_KM_MIN = 1.0
TEMPO_SERVICO_MIN = 15

CUSTO_KM = 0.5
CUSTO_MIN_OPERACAO = 1.0
MULTA_ATRASO_POR_MIN = 5.0

START_DEPOT_MIN = df_clientes.iloc[0]['T_Inicio_h'] * 60
END_DEPOT_MIN = df_clientes.iloc[0]['T_Fim_h'] * 60

def avaliar_rota(rota):
    demanda_rota = sum(df_clientes.iloc[no]['Demanda'] for no in rota if no != 0)
    if demanda_rota > CAPACIDADE_VEICULO:
        return (999999, False, True)

    custo_total = 0
    tempo_atual_min = START_DEPOT_MIN

    for i in range(len(rota) - 1):
        origem = rota[i]
        destino = rota[i+1]

        distancia_km = dist_matrix[origem, destino]
        tempo_viagem_min = distancia_km / VELOCIDADE_MEDIA_KM_MIN

        tempo_chegada = tempo_atual_min + tempo_viagem_min

        custo_total += (distancia_km * CUSTO_KM) + (tempo_viagem_min * CUSTO_MIN_OPERACAO)

        if destino != 0:
            t_inicio_min = df_clientes.iloc[destino]['T_Inicio_h'] * 60
            t_fim_min = df_clientes.iloc[destino]['T_Fim_h'] * 60

            multa_atraso = 0
            tempo_espera = 0

            if tempo_chegada < t_inicio_min:
                tempo_espera = t_inicio_min - tempo_chegada
                tempo_inicio_servico = t_inicio_min

            elif tempo_chegada > t_fim_min:
                atraso_min = tempo_chegada - t_fim_min
                multa_atraso = atraso_min * MULTA_ATRASO_POR_MIN
                tempo_inicio_servico = tempo_chegada

            else:
                tempo_inicio_servico = tempo_chegada

            custo_total += multa_atraso

            tempo_saida = tempo_inicio_servico + TEMPO_SERVICO_MIN
            tempo_atual_min = tempo_saida

            if multa_atraso > 0:
                return (custo_total, True, False)

        else:
            if tempo_chegada > END_DEPOT_MIN:
                return (999999, True, False)

            tempo_atual_min = tempo_chegada

    return (custo_total, True, True)

rotas = {}
for i in df_clientes[df_clientes['ID'] != 'C'].index:
    rotas[i] = [0, i, 0]

economias = []
indices_clientes = [i for i in rotas.keys()]

for i in indices_clientes:
    for j in indices_clientes:
        if i < j:
            s_ij = dist_matrix[0, i] + dist_matrix[j, 0] - dist_matrix[i, j]
            economias.append((s_ij, i, j))

economias.sort(key=lambda x: x[0], reverse=True)

rotas_finais = rotas.copy()
economia_valida_encontrada = False

for economia, i, j in economias:
    rota_i_id = next((k for k, v in rotas_finais.items() if i in v), None)
    rota_j_id = next((k for k, v in rotas_finais.items() if j in v), None)

    if rota_i_id != rota_j_id and rota_i_id is not None and rota_j_id is not None:
        rota_i = rotas_finais[rota_i_id]
        rota_j = rotas_finais[rota_j_id]

        if rota_i[-2] == i and rota_j[1] == j:

            nova_rota = rota_i[:-1] + rota_j[1:]

            custo_novo, capacidade_ok, tw_ok = avaliar_rota(nova_rota)

            if capacidade_ok and tw_ok:
                rotas_finais[rota_i_id] = nova_rota
                del rotas_finais[rota_j_id]
                economia_valida_encontrada = True

print("\n--- Resultado Refinado da Heurística CW ---")
print(f"Total de Rotas Otimizadas: {len(rotas_finais)}")

custo_total_otimizado = 0
for idx, rota in rotas_finais.items():
    custo, cap_ok, tw_ok = avaliar_rota(rota)
    demanda = sum(df_clientes.iloc[no]['Demanda'] for no in rota if no != 0)

    status_tw = "OK" if tw_ok else "VIOLADO"
    status_cap = "OK" if cap_ok else "VIOLADO"

    print(f"Veículo {idx} | Clientes: {rota} | Demanda: {demanda}/{CAPACIDADE_VEICULO} ({status_cap}) | Custo: R$ {custo:.2f} | TW: {status_tw}")
    custo_total_otimizado += custo

print(f"\nCUSTO TOTAL FINAL (A ser otimizado): R$ {custo_total_otimizado:.2f}")

plot_data = []
rotas_df = []
idx_veiculo = 1

for rota_id, rota in rotas_finais.items():
    for i in range(len(rota) - 1):
        ponto_a = df_clientes.iloc[rota[i]]
        ponto_b = df_clientes.iloc[rota[i+1]]

        plot_data.append({
            'Veiculo': f"V{idx_veiculo}",
            'Lat': [ponto_a['Lat'], ponto_b['Lat']],
            'Lon': [ponto_a['Lon'], ponto_b['Lon']],
            'Ordem': i
        })

    for no in rota[1:-1]:
        cliente = df_clientes.iloc[no].copy()
        cliente['Veiculo'] = f"V{idx_veiculo}"
        rotas_df.append(cliente)

    idx_veiculo += 1

df_rotas = pd.DataFrame(rotas_df)

fig = px.scatter_mapbox(
    df_rotas,
    lat="Lat",
    lon="Lon",
    hover_name="ID",
    color="Veiculo",
    size="Demanda",
    zoom=11,
    title="Rotas Otimizadas (Heurística Clarke-Wright)",
    mapbox_style="carto-positron"
)

for linha in plot_data:
    fig.add_trace(
        px.line_mapbox(
            lat=linha['Lat'],
            lon=linha['Lon'],
        ).update_traces(
            line=dict(width=2, color=fig.data[linha['Ordem']].line.color),
            name=f"Rota {linha['Veiculo']}",
            mode='lines'
        ).data[0]
    )

fig.add_trace(
    px.scatter_mapbox(
        depot.to_frame().T,
        lat="Lat",
        lon="Lon",
        hover_name="ID"
    ).update_traces(
        marker=dict(size=15, symbol='star', color='black'),
        name="Depósito"
    ).data[0]
)

fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    showlegend=True
)

fig.show()
fig.write_html("rotas_otimizadas.html")
print("\nArquivo 'rotas_otimizadas.html' gerado com sucesso!")

plot_data = []
rotas_df = []
idx_veiculo = 1

for rota_id, rota in rotas_finais.items():

    for i in range(len(rota) - 1):
        ponto_a = df_clientes.iloc[rota[i]]
        ponto_b = df_clientes.iloc[rota[i+1]]

        plot_data.append({
            'Veiculo': f"V{idx_veiculo}",
            'Lat': [ponto_a['Lat'], ponto_b['Lat']],
            'Lon': [ponto_a['Lon'], ponto_b['Lon']],
            'Ordem': i
        })

    for no in rota[1:-1]:
        cliente = df_clientes.iloc[no].copy()
        cliente['Veiculo'] = f"V{idx_veiculo}"
        rotas_df.append(cliente)

    idx_veiculo += 1

df_rotas = pd.DataFrame(rotas_df)

fig = px.scatter_mapbox(
    df_rotas,
    lat="Lat",
    lon="Lon",
    hover_name="ID",
    color="Veiculo",
    size="Demanda",
    zoom=11,
    title="Rotas Otimizadas (Heurística Clarke-Wright)",
    mapbox_style="carto-positron"
)

for linha in plot_data:
    fig.add_trace(
        px.line_mapbox(
            lat=linha['Lat'],
            lon=linha['Lon'],
        ).update_traces(
            line=dict(width=2, color=fig.data[linha['Ordem']].line.color),
            name=f"Rota {linha['Veiculo']}",
            mode='lines'
        ).data[0]
    )

fig.add_trace(
    px.scatter_mapbox(
        depot.to_frame().T,
        lat="Lat",
        lon="Lon",
        hover_name="ID"
    ).update_traces(
        marker=dict(size=15, symbol='star', color='black'),
        name="Depósito"
    ).data[0]
)

fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    showlegend=True
)

fig.show()
fig.write_html("rotas_otimizadas.html")
print("\nArquivo 'rotas_otimizadas.html' gerado com sucesso!")

def two_opt_optimize(routes):
    """Aplica o algoritmo 2-opt a cada rota para otimizar o custo (busca local)."""
    optimized_routes = {}
    
    for route_id, route in routes.items():
        best_route = route
        best_cost, _, _ = avaliar_rota(best_route)
        
        melhoria_encontrada = True
        while melhoria_encontrada:
            melhoria_encontrada = False
            n = len(best_route)
            
            if n < 4:
                optimized_routes[route_id] = best_route
                break
            
            for i in range(1, n - 2):
                for k in range(i + 1, n - 1):
                    new_route = best_route[:i] + best_route[k:i-1:-1] + best_route[k+1:]
                    
                    new_cost, capacidade_ok, tw_ok = avaliar_rota(new_route)
                    
                    if new_cost < best_cost and capacidade_ok and tw_ok:
                        best_route = new_route
                        best_cost = new_cost
                        melhoria_encontrada = True
                        break
                if melhoria_encontrada:
                    break
            
        optimized_routes[route_id] = best_route
        
    return optimized_routes

rotas_otimizadas_2opt = two_opt_optimize(rotas_finais)

print("\n--- Resultado Otimizado (2-opt) ---")
print(f"Total de Rotas Otimizadas: {len(rotas_otimizadas_2opt)}")

custo_total_2opt = 0
for idx, rota in rotas_otimizadas_2opt.items():
    custo, cap_ok, tw_ok = avaliar_rota(rota)
    
    demanda = sum(df_clientes.iloc[no]['Demanda'] for no in rota if no != 0)
    
    status_tw = "OK" if tw_ok else "VIOLADO"
    status_cap = "OK" if cap_ok else "VIOLADO"
    
    print(f"Veículo {idx} | Clientes: {rota} | Demanda: {demanda}/{CAPACIDADE_VEICULO} ({status_cap}) | Custo: R$ {custo:.2f} | TW: {status_tw}")
    custo_total_2opt += custo

print(f"\nCUSTO TOTAL FINAL 2-opt: R$ {custo_total_2opt:.2f}")

gantt_data = []
tw_data = []
idx_veiculo = 1

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def calcular_cronograma(rota):
    
    custo_total, capacidade_ok, tw_valida = avaliar_rota(rota)
    
    if not capacidade_ok or custo_total == 999999:
        return (custo_total, False, [])

    eventos = []
    tempo_atual_min = START_DEPOT_MIN 
    
    eventos.append({
        'Veiculo': 'TEMP', 
        'Task': f"Depósito (Partida)", 
        'Start': tempo_atual_min, 
        'Finish': tempo_atual_min, 
        'Cor': 'saida'
    })
    
    for i in range(len(rota) - 1):
        origem = rota[i]
        destino = rota[i+1]
        
        distancia_km = dist_matrix[origem, destino]
        tempo_viagem_min = distancia_km / VELOCIDADE_MEDIA_KM_MIN
        
        tempo_chegada = tempo_atual_min + tempo_viagem_min
        
        eventos.append({
            'Veiculo': 'TEMP',
            'Task': f"Viagem {df_clientes.iloc[origem]['ID']}->{df_clientes.iloc[destino]['ID']}",
            'Start': tempo_atual_min,
            'Finish': tempo_chegada,
            'Cor': 'viagem'
        })
        
        if destino != 0:
            t_inicio_min = df_clientes.iloc[destino]['T_Inicio_h'] * 60
            t_fim_min = df_clientes.iloc[destino]['T_Fim_h'] * 60
            
            tempo_espera = 0
            
            if tempo_chegada < t_inicio_min:
                tempo_espera = t_inicio_min - tempo_chegada
                tempo_inicio_servico = t_inicio_min
                cor_servico = 'espera'
            elif tempo_chegada > t_fim_min:
                tempo_inicio_servico = tempo_chegada
                cor_servico = 'atraso'
            else:
                tempo_inicio_servico = tempo_chegada
                cor_servico = 'ok'

            if tempo_espera > 0:
                  eventos.append({
                      'Veiculo': 'TEMP', 
                      'Task': f"Espera {df_clientes.iloc[destino]['ID']}", 
                      'Start': tempo_chegada, 
                      'Finish': tempo_inicio_servico,
                      'Cor': 'espera'
                  })

            tempo_saida = tempo_inicio_servico + TEMPO_SERVICO_MIN
            
            eventos.append({
                'Veiculo': 'TEMP', 
                'Task': f"Serviço {df_clientes.iloc[destino]['ID']} (TW: {t_inicio_min}-{t_fim_min}min)",
                'Start': tempo_inicio_servico,
                'Finish': tempo_saida,
                'Cor': cor_servico
            })
            
            tempo_atual_min = tempo_saida
            
        else:
            tempo_atual_min = tempo_chegada
            eventos.append({
                'Veiculo': 'TEMP', 
                'Task': "Depósito (Chegada)", 
                'Start': tempo_chegada, 
                'Finish': tempo_chegada,
                'Cor': 'saida'
            })
    
    return (custo_total, tw_valida, eventos)

# A variável de entrada deve ser 'rotas_finais' ou a variável pós-otimização, ex: rotas_otimizadas_2opt
rotas_de_entrada = rotas_finais 

gantt_data = []
tw_data = []
idx_veiculo = 1

for rota_id, rota in rotas_de_entrada.items():
    _, _, eventos = calcular_cronograma(rota)
    
    for evento in eventos:
        evento['Veiculo'] = f"V{idx_veiculo}"
        gantt_data.append(evento)
    
    for no in rota[1:-1]:
        cliente_info = df_clientes.iloc[no]
        
        tw_data.append({
            'Veiculo': f"V{idx_veiculo}",
            'Cliente_ID': cliente_info['ID'],
            'Start_TW': cliente_info['T_Inicio_h'] * 60,
            'Finish_TW': cliente_info['T_Fim_h'] * 60,
            'Label': f"TW: {cliente_info['ID']}"
        })
        
    idx_veiculo += 1

df_gantt = pd.DataFrame(gantt_data)
df_tw = pd.DataFrame(tw_data)

color_map = {
    'viagem': 'rgba(128, 128, 128, 0.5)',
    'ok': '#1f77b4',
    'espera': '#ff7f0e',
    'atraso': '#d62728',
    'saida': '#000000'
}

fig_gantt = px.timeline(
    df_gantt[df_gantt['Task'].str.contains("Serviço|Espera|Viagem")],
    x_start="Start", 
    x_end="Finish", 
    y="Veiculo", 
    color="Cor", 
    color_discrete_map=color_map,
    title="Gráfico de Gantt Detalhado com Janelas de Tempo",
    hover_name="Task",
    opacity=0.8
)

for veiculo in df_tw['Veiculo'].unique():
    df_tw_veiculo = df_tw[df_tw['Veiculo'] == veiculo]
    
    fig_gantt.add_trace(
        go.Bar(
            y=df_tw_veiculo['Veiculo'],
            x=df_tw_veiculo['Finish_TW'] - df_tw_veiculo['Start_TW'],
            base=df_tw_veiculo['Start_TW'],
            name=f"Janela TW - {veiculo}",
            marker_color='rgba(0, 255, 0, 0.15)',
            hovertemplate='Cliente: %{customdata[0]}<br>Janela: %{base} - %{x}<extra></extra>',
            customdata=df_tw_veiculo[['Cliente_ID']],
            orientation='h',
            showlegend=False
        )
    )

fig_gantt.update_traces(orientation='h', selector=dict(type='bar'), showlegend=True)

fig_gantt.update_yaxes(categoryorder="array", categoryarray=sorted(df_gantt['Veiculo'].unique()))

fig_gantt.update_xaxes(title="Tempo (Minutos do dia - Ex: 480 min = 8:00)",
                       tickvals=list(range(480, 1021, 60)),
                       ticktext=[f"{h//60}:00" for h in range(480, 1021, 60)])

fig_gantt.show()

fig_gantt.write_html("gantt_schedule_aprimorado.html")
print("\nArquivo 'gantt_schedule_aprimorado.html' gerado com sucesso!")