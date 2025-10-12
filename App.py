import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import copy 
import plotly.io as pio
from flask import Flask, request, jsonify, render_template

# Configuração do Plotly para gerar arquivos HTML
pio.renderers.default = "browser"

# --- CLASSE VRPTWSolver (Otimização) ---

class VRPTWSolver:
    def __init__(self, df_clientes, capacidade_veiculo, velocidade_media_km_min, 
                 tempo_servico_min, custo_km, custo_min_operacao, 
                 multa_atraso_por_min):
        
        self.df_clientes = df_clientes
        self.depot = df_clientes[df_clientes['ID'] == 'C'].iloc[0]
        self.CAPACIDADE_VEICULO = capacidade_veiculo
        self.VELOCIDADE_MEDIA_KM_MIN = velocidade_media_km_min
        self.TEMPO_SERVICO_MIN = tempo_servico_min
        self.CUSTO_KM = custo_km
        self.CUSTO_MIN_OPERACAO = custo_min_operacao
        self.MULTA_ATRASO_POR_MIN = multa_atraso_por_min
        
        self.START_DEPOT_MIN = self.df_clientes.iloc[0]['T_Inicio_h'] * 60
        self.END_DEPOT_MIN = self.df_clientes.iloc[0]['T_Fim_h'] * 60

        self.dist_matrix = self._calcular_matriz_distancia()
        self.n_clientes = len(df_clientes)

    def _calcular_matriz_distancia(self):
        n_clientes = len(self.df_clientes)
        dist_matrix = np.zeros((n_clientes, n_clientes))
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        for i in range(n_clientes):
            for j in range(n_clientes):
                dist_matrix[i, j] = haversine(
                    self.df_clientes.iloc[i]['Lat'], self.df_clientes.iloc[i]['Lon'],
                    self.df_clientes.iloc[j]['Lat'], self.df_clientes.iloc[j]['Lon']
                )
        return dist_matrix

    def copy_routes(self, routes):
        return {k: v[:] for k, v in routes.items()}

    def avaliar_rota(self, rota):
        # Esta função avalia custo, capacidade e TW
        demanda_rota = sum(self.df_clientes.iloc[no]['Demanda'] for no in rota if no != 0)
        if demanda_rota > self.CAPACIDADE_VEICULO:
            return (999999, False, False)

        custo_total = 0
        tempo_atual_min = self.START_DEPOT_MIN
        tw_ok = True

        for i in range(len(rota) - 1):
            origem = rota[i]
            destino = rota[i+1]
            distancia_km = self.dist_matrix[origem, destino]
            tempo_viagem_min = distancia_km / self.VELOCIDADE_MEDIA_KM_MIN
            tempo_chegada = tempo_atual_min + tempo_viagem_min
            custo_total += (distancia_km * self.CUSTO_KM) + (tempo_viagem_min * self.CUSTO_MIN_OPERACAO)

            if destino != 0:
                t_inicio_min = self.df_clientes.iloc[destino]['T_Inicio_h'] * 60
                t_fim_min = self.df_clientes.iloc[destino]['T_Fim_h'] * 60
                multa_atraso = 0

                if tempo_chegada < t_inicio_min:
                    tempo_inicio_servico = t_inicio_min
                elif tempo_chegada > t_fim_min:
                    atraso_min = tempo_chegada - t_fim_min
                    multa_atraso = atraso_min * self.MULTA_ATRASO_POR_MIN
                    tempo_inicio_servico = tempo_chegada
                    tw_ok = False
                else:
                    tempo_inicio_servico = tempo_chegada

                custo_total += multa_atraso
                tempo_saida = tempo_inicio_servico + self.TEMPO_SERVICO_MIN
                tempo_atual_min = tempo_saida
            else:
                if tempo_chegada > self.END_DEPOT_MIN:
                    return (999999, True, False)
                tempo_atual_min = tempo_chegada

        return (custo_total, True, tw_ok)

    def calculate_total_cost(self, routes):
        total_cost = 0
        if not routes: return 999999
        for rota in routes.values():
            cost, _, tw_ok = self.avaliar_rota(rota)
            if cost == 999999 or not tw_ok:
                return 999999 
            total_cost += cost
        return total_cost

    def criar_rotas_cw(self):
        # Implementação do Algoritmo de Economia (Clarke & Wright)
        rotas = {}
        for i in self.df_clientes[self.df_clientes['ID'] != 'C'].index:
            rotas[i] = [0, i, 0]

        economias = []
        indices_clientes = [i for i in rotas.keys()]

        for i in indices_clientes:
            for j in indices_clientes:
                if i < j:
                    s_ij = self.dist_matrix[0, i] + self.dist_matrix[j, 0] - self.dist_matrix[i, j]
                    economias.append((s_ij, i, j))

        economias.sort(key=lambda x: x[0], reverse=True)
        rotas_finais_cw = rotas.copy()
        
        for _, i, j in economias:
            rota_i_id = next((k for k, v in rotas_finais_cw.items() if i in v), None)
            rota_j_id = next((k for k, v in rotas_finais_cw.items() if j in v), None)

            if rota_i_id != rota_j_id and rota_i_id is not None and rota_j_id is not None:
                rota_i = rotas_finais_cw[rota_i_id]
                rota_j = rotas_finais_cw[rota_j_id]

                if rota_i[-2] == i and rota_j[1] == j:
                    nova_rota = rota_i[:-1] + rota_j[1:]
                    _, capacidade_ok, tw_ok = self.avaliar_rota(nova_rota)

                    if capacidade_ok and tw_ok:
                        rotas_finais_cw[rota_i_id] = nova_rota
                        del rotas_finais_cw[rota_j_id]
                        
        return rotas_finais_cw

    def two_opt_optimize(self, routes):
        # Implementação do 2-opt (melhoria local)
        optimized_routes = {}
        
        for route_id, route in routes.items():
            best_route = route
            best_cost, _, _ = self.avaliar_rota(best_route)
            
            melhoria_encontrada = True
            while melhoria_encontrada:
                melhoria_encontrada = False
                n = len(best_route)
                if n < 4: break

                for i in range(1, n - 2):
                    for k in range(i + 1, n - 1):
                        new_route = best_route[:i] + best_route[k:i-1:-1] + best_route[k+1:]
                        
                        new_cost, capacidade_ok, tw_ok = self.avaliar_rota(new_route)
                        
                        if new_cost < best_cost and capacidade_ok and tw_ok:
                            best_route = new_route
                            best_cost = new_cost
                            melhoria_encontrada = True
                            break 
                    if melhoria_encontrada: break
            
            optimized_routes[route_id] = best_route
        return optimized_routes

    def get_2opt_movements_intra(self, current_solution):
        # Movimentos 2-opt intra-rota para a Busca Tabu
        moves = []
        for route_id, route in current_solution.items():
            n = len(route)
            if n < 4: continue

            for i in range(1, n - 2):
                for k in range(i + 1, n - 1):
                    new_route = route[:i] + route[k:i-1:-1] + route[k+1:]
                    
                    cost, cap_ok, tw_ok = self.avaliar_rota(new_route)
                    
                    if cap_ok and tw_ok:
                        new_solution = self.copy_routes(current_solution)
                        new_solution[route_id] = new_route
                        
                        move = ('2opt', route_id, route[i], route[k])
                        moves.append((new_solution, self.calculate_total_cost(new_solution), move))
        return moves

    def get_swap_movements(self, current_solution):
        # Movimentos Swap (Intra e Inter-rota) para a Busca Tabu
        moves = []
        route_keys = list(current_solution.keys())
        
        for route1_id in route_keys:
            r1 = current_solution[route1_id]
            n1 = len(r1)
            for i in range(1, n1 - 1):
                for j in range(i + 1, n1 - 1):
                    new_r1 = r1[:]
                    new_r1[i], new_r1[j] = new_r1[j], new_r1[i]
                    
                    _, cap_ok, tw_ok = self.avaliar_rota(new_r1)
                    if not cap_ok or not tw_ok: continue
                    
                    new_solution = self.copy_routes(current_solution)
                    new_solution[route1_id] = new_r1
                    
                    move = ('swap_intra', route1_id, r1[i], r1[j])
                    moves.append((new_solution, self.calculate_total_cost(new_solution), move))

        for r1_idx, route1_id in enumerate(route_keys):
            r1 = current_solution[route1_id]
            n1 = len(r1)
            for r2_idx in range(r1_idx + 1, len(route_keys)):
                route2_id = route_keys[r2_idx]
                r2 = current_solution[route2_id]
                n2 = len(r2)
                
                for i in range(1, n1 - 1):
                    for j in range(1, n2 - 1):
                        new_r1 = r1[:]
                        new_r2 = r2[:]
                        
                        new_r1[i], new_r2[j] = r2[j], r1[i]
                        
                        _, cap1_ok, tw1_ok = self.avaliar_rota(new_r1)
                        _, cap2_ok, tw2_ok = self.avaliar_rota(new_r2)
                        
                        if cap1_ok and tw1_ok and cap2_ok and tw2_ok:
                            new_solution = self.copy_routes(current_solution)
                            new_solution[route1_id] = new_r1
                            new_solution[route2_id] = new_r2
                            
                            move = ('swap_inter', r1[i], r2[j])
                            moves.append((new_solution, self.calculate_total_cost(new_solution), move))
        return moves

    def get_insert_movements(self, current_solution):
        # Movimentos Insert (Intra e Inter-rota) para a Busca Tabu
        moves = []
        route_keys = list(current_solution.keys())
        
        for r1_idx, route1_id in enumerate(route_keys):
            r1 = current_solution[route1_id]
            n1 = len(r1)
            if n1 < 3: continue

            for i in range(1, n1 - 1):
                customer_to_move = r1[i]
                
                temp_r1_removed = r1[:i] + r1[i+1:]
                
                # Insert Intra-Route
                for k in range(1, len(temp_r1_removed)):
                    if temp_r1_removed[k] == r1[i+1] and temp_r1_removed[k-1] == r1[i-1]: continue
                    
                    new_r1 = temp_r1_removed[:k] + [customer_to_move] + temp_r1_removed[k:]
                    
                    _, cap1_ok, tw1_ok = self.avaliar_rota(new_r1)
                    if not cap1_ok or not tw1_ok: continue
                    
                    new_solution = self.copy_routes(current_solution)
                    new_solution[route1_id] = new_r1
                    
                    move = ('insert_intra', route1_id, r1[i], route1_id)
                    moves.append((new_solution, self.calculate_total_cost(new_solution), move))
                
                # Insert Inter-Route
                for r2_idx, route2_id in enumerate(route_keys):
                    if route1_id == route2_id: continue
                    
                    r2 = current_solution[route2_id]
                    n2 = len(r2)
                    
                    for k in range(1, n2):
                        new_r2 = r2[:k] + [customer_to_move] + r2[k:]
                        
                        _, cap1_ok, tw1_ok = self.avaliar_rota(temp_r1_removed)
                        _, cap2_ok, tw2_ok = self.avaliar_rota(new_r2)
                        
                        if cap1_ok and tw1_ok and cap2_ok and tw2_ok:
                            new_solution = self.copy_routes(current_solution)
                            new_solution[route2_id] = new_r2
                            
                            if len(temp_r1_removed) == 2:
                                if route1_id in new_solution: del new_solution[route1_id]
                            else:
                                new_solution[route1_id] = temp_r1_removed
                            
                            move = ('insert_inter', r1[i], route2_id)
                            moves.append((new_solution, self.calculate_total_cost(new_solution), move))
        return moves

    def tabu_search_vrptw(self, initial_routes, max_iterations=200, tabu_list_size=10):
        # Implementação da Busca Tabu (Tabu Search)
        best_solution = self.copy_routes(initial_routes)
        best_cost = self.calculate_total_cost(best_solution)

        current_solution = self.copy_routes(initial_routes)

        tabu_list = []
        
        for iteration in range(max_iterations):
            all_moves = []
            
            all_moves.extend(self.get_2opt_movements_intra(current_solution))
            all_moves.extend(self.get_swap_movements(current_solution))
            all_moves.extend(self.get_insert_movements(current_solution))

            if not all_moves: break

            all_moves.sort(key=lambda x: x[1])

            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move_for_tabu = None

            for new_solution, new_cost, move in all_moves:
                is_tabu = move in tabu_list
                aspiration_criterion = new_cost < best_cost

                if not is_tabu or aspiration_criterion:
                    if new_cost < best_neighbor_cost:
                        best_neighbor_cost = new_cost
                        best_neighbor = new_solution
                        best_move_for_tabu = move
                        if not is_tabu:
                            break

            if best_neighbor is None: break

            current_solution = best_neighbor
            
            if best_neighbor_cost < best_cost:
                best_cost = best_neighbor_cost
                best_solution = self.copy_routes(current_solution)

            if best_move_for_tabu is not None:
                tabu_list.append(best_move_for_tabu)
                if len(tabu_list) > tabu_list_size:
                    tabu_list.pop(0)

        return best_solution

    def calcular_cronograma(self, rota):
        # Calcula os eventos (viagens, serviços, esperas) para o Gráfico de Gantt
        custo_total, capacidade_ok, tw_valida = self.avaliar_rota(rota)
        if not capacidade_ok or custo_total == 999999 or not tw_valida:
            return (custo_total, False, [])

        eventos = []
        tempo_atual_min = self.START_DEPOT_MIN 
        eventos.append({'Veiculo': 'TEMP', 'Task': f"Depósito (Partida)", 'Start': tempo_atual_min, 'Finish': tempo_atual_min, 'Cor': 'saida'})
        
        for i in range(len(rota) - 1):
            origem = rota[i]
            destino = rota[i+1]
            distancia_km = self.dist_matrix[origem, destino]
            tempo_viagem_min = distancia_km / self.VELOCIDADE_MEDIA_KM_MIN
            tempo_chegada = tempo_atual_min + tempo_viagem_min
            
            eventos.append({'Veiculo': 'TEMP', 'Task': f"Viagem {self.df_clientes.iloc[origem]['ID']}->{self.df_clientes.iloc[destino]['ID']}", 'Start': tempo_atual_min, 'Finish': tempo_chegada, 'Cor': 'viagem'})
            
            if destino != 0:
                t_inicio_min = self.df_clientes.iloc[destino]['T_Inicio_h'] * 60
                t_fim_min = self.df_clientes.iloc[destino]['T_Fim_h'] * 60
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
                    eventos.append({'Veiculo': 'TEMP', 'Task': f"Espera {self.df_clientes.iloc[destino]['ID']}", 'Start': tempo_chegada, 'Finish': tempo_inicio_servico, 'Cor': 'espera'})

                tempo_saida = tempo_inicio_servico + self.TEMPO_SERVICO_MIN
                eventos.append({'Veiculo': 'TEMP', 'Task': f"Serviço {self.df_clientes.iloc[destino]['ID']} (TW: {t_inicio_min}-{t_fim_min}min)", 'Start': tempo_inicio_servico, 'Finish': tempo_saida, 'Cor': cor_servico})
                tempo_atual_min = tempo_saida
            else:
                tempo_atual_min = tempo_chegada
                eventos.append({'Veiculo': 'TEMP', 'Task': "Depósito (Chegada)", 'Start': tempo_chegada, 'Finish': tempo_chegada, 'Cor': 'saida'})
        
        return (custo_total, tw_valida, eventos)

    def gerar_visualizacoes(self, rotas_finais):
        # Gera e salva os arquivos HTML do Mapa e do Gantt
        plot_data = []
        rotas_df = []
        gantt_data = []
        tw_data = []
        idx_veiculo = 1

        for rota_id, rota in rotas_finais.items():
            for i in range(len(rota) - 1):
                ponto_a = self.df_clientes.iloc[rota[i]]
                ponto_b = self.df_clientes.iloc[rota[i+1]]
                plot_data.append({
                    'Veiculo': f"V{idx_veiculo}",
                    'Lat': [ponto_a['Lat'], ponto_b['Lat']],
                    'Lon': [ponto_a['Lon'], ponto_b['Lon']],
                    'Ordem': i
                })
            for no in rota[1:-1]:
                cliente = self.df_clientes.iloc[no].copy()
                cliente['Veiculo'] = f"V{idx_veiculo}"
                rotas_df.append(cliente)

            _, _, eventos = self.calcular_cronograma(rota)
            for evento in eventos:
                evento['Veiculo'] = f"V{idx_veiculo}"
                gantt_data.append(evento)
            
            for no in rota[1:-1]:
                cliente_info = self.df_clientes.iloc[no]
                tw_data.append({
                    'Veiculo': f"V{idx_veiculo}", 'Cliente_ID': cliente_info['ID'],
                    'Start_TW': cliente_info['T_Inicio_h'] * 60, 'Finish_TW': cliente_info['T_Fim_h'] * 60,
                    'Label': f"TW: {cliente_info['ID']}"
                })
            idx_veiculo += 1

        df_rotas = pd.DataFrame(rotas_df)

        # 1. Mapa de Rotas
        fig_mapa = px.scatter_mapbox(
            df_rotas, lat="Lat", lon="Lon", hover_name="ID", color="Veiculo", size="Demanda", zoom=11,
            title="Rotas Otimizadas", mapbox_style="carto-positron"
        )
        for linha in plot_data:
            cor_veiculo = next((t.line.color for t in fig_mapa.data if t.name == linha['Veiculo']), 'blue')
            fig_mapa.add_trace(go.Scattermapbox(
                lat=linha['Lat'], lon=linha['Lon'],
                mode='lines', line=dict(width=2, color=cor_veiculo),
                name=f"Rota {linha['Veiculo']}", showlegend=False
            ))
        fig_mapa.add_trace(go.Scattermapbox(
            lat=[self.depot['Lat']], lon=[self.depot['Lon']],
            mode='markers', marker=dict(size=15, symbol='star', color='black'),
            name="Depósito", hovertext=["Depósito"]
        ))
        fig_mapa.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, showlegend=True)
        # Salva o arquivo HTML
        fig_mapa.write_html("rotas_otimizadas.html", full_html=False, include_plotlyjs='cdn') 

        # 2. Gráfico de Gantt
        df_gantt = pd.DataFrame(gantt_data)
        df_tw = pd.DataFrame(tw_data)
        color_map = {'viagem': 'rgba(128, 128, 128, 0.5)', 'ok': '#1f77b4', 'espera': '#ff7f0e', 'atraso': '#d62728', 'saida': '#000000'}

        fig_gantt = px.timeline(
            df_gantt[df_gantt['Task'].str.contains("Serviço|Espera|Viagem")], x_start="Start", x_end="Finish", 
            y="Veiculo", color="Cor", color_discrete_map=color_map, title="Gráfico de Gantt Detalhado com Janelas de Tempo",
            hover_name="Task", opacity=0.8
        )
        
        for veiculo in df_tw['Veiculo'].unique():
            df_tw_veiculo = df_tw[df_tw['Veiculo'] == veiculo]
            for _, row in df_tw_veiculo.iterrows():
                fig_gantt.add_trace(go.Bar(
                    y=[row['Veiculo']], x=[row['Finish_TW'] - row['Start_TW']], base=[row['Start_TW']], 
                    marker_color='rgba(0, 255, 0, 0.15)', orientation='h', showlegend=False,
                    hoverinfo='text', text=f"TW: {row['Cliente_ID']} ({row['Start_TW']//60:02d}:00 - {row['Finish_TW']//60:02d}:00)"
                ))

        fig_gantt.update_yaxes(categoryorder="array", categoryarray=sorted(df_gantt['Veiculo'].unique()))
        fig_gantt.update_xaxes(title="Tempo (Minutos do dia)", tickvals=list(range(480, 1021, 60)), 
                                ticktext=[f"{h//60:02d}:00" for h in range(480, 1021, 60)])
        # Salva o arquivo HTML
        fig_gantt.write_html("gantt_schedule_aprimorado.html", full_html=False, include_plotlyjs='cdn')
        
        return fig_mapa, fig_gantt

    def run_optimization(self, max_iterations=200, tabu_list_size=10):
        rotas_iniciais_cw = self.criar_rotas_cw()
        rotas_otimizadas_2opt = self.two_opt_optimize(rotas_iniciais_cw)
        rotas_finais = self.tabu_search_vrptw(rotas_otimizadas_2opt, 
                                              max_iterations=max_iterations, 
                                              tabu_list_size=tabu_list_size)

        self.gerar_visualizacoes(rotas_finais)

        custo_final = self.calculate_total_cost(rotas_finais)
        
        return custo_final

# --- CONFIGURAÇÃO DO FLASK E ROTEAMENTO ---

app = Flask(__name__)

# Dados de exemplo fixos
data_clientes = {
    'ID': ['C', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
    'Lat': [-23.5505, -23.545, -23.560, -23.570, -23.535, -23.555, -23.540],
    'Lon': [-46.6333, -46.640, -46.625, -46.638, -46.650, -46.615, -46.630],
    'Demanda': [0, 5, 10, 3, 7, 2, 8], 
    'T_Inicio_h': [8, 9, 10, 9, 11, 10, 8],
    'T_Fim_h': [17, 12, 11, 10, 13, 12, 9]
}
df_clientes = pd.DataFrame(data_clientes)

# Parâmetros fixos para a TS
TEMPO_SERVICO_MIN = 15
CUSTO_MIN_OPERACAO = 1.0
MULTA_ATRASO_POR_MIN = 5.0
MAX_ITER = 100 
TABU_SIZE = 7

def initialize_solver(capacity, speed, cost_km):
    return VRPTWSolver(
        df_clientes, capacity, speed, 
        TEMPO_SERVICO_MIN, cost_km, CUSTO_MIN_OPERACAO, 
        MULTA_ATRASO_POR_MIN
    )

@app.route('/')
def index():
    # Rota que executa a otimização inicial e carrega o HTML
    CAPACIDADE_VEICULO = 20
    VELOCIDADE_MEDIA_KM_MIN = 1.0
    CUSTO_KM = 0.5
    
    try:
        solver = initialize_solver(CAPACIDADE_VEICULO, VELOCIDADE_MEDIA_KM_MIN, CUSTO_KM)
        solver.run_optimization(MAX_ITER, TABU_SIZE)
    except Exception as e:
        print(f"Erro na otimização inicial: {e}")
        
    return render_template('dashboard.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    # Rota que é chamada pelo JavaScript do dashboard para recalcular
    data = request.json
    try:
        capacity = data.get('capacity')
        speed = data.get('speed')
        cost_km = data.get('cost_km')

        if not all([capacity is not None, speed is not None, cost_km is not None]):
            raise ValueError("Parâmetros de otimização incompletos.")

        solver = initialize_solver(capacity, speed, cost_km)
        final_cost = solver.run_optimization(MAX_ITER, TABU_SIZE)
        
        return jsonify({
            'status': 'success',
            'final_cost': f"R$ {final_cost:.2f}",
            'message': 'Otimização concluída com sucesso.'
        })

    except Exception as e:
        print(f"Erro no processamento da requisição: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Cria a pasta 'templates' e coloca o dashboard.html nela antes de rodar!
    print("\n--- INICIANDO SERVIDOR FLASK ---")
    print("Acesse: http://127.0.0.1:5000/")
    app.run(debug=False)