import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import copy 
import plotly.io as pio
import sqlite3

# ✅ Importações do módulo de banco de dados
from db_utils import (
    garantir_tabela_clientes,
    garantir_tabela_veiculos,
    carregar_clientes_do_banco,
    inserir_deposito_padrao
)

pio.renderers.default = "vscode"

# ✅ Garante que as tabelas existem antes de rodar
garantir_tabela_clientes()
garantir_tabela_veiculos()
inserir_deposito_padrao()


# ==========================================================
# CLASSE PRINCIPAL DE OTIMIZAÇÃO VRPTW
# ==========================================================
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

    # ----------------------------------------------------------
    # MATRIZ DE DISTÂNCIA
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # FUNÇÕES AUXILIARES
    # ----------------------------------------------------------
    def copy_routes(self, routes):
        return {k: v[:] for k, v in routes.items()}

    # ----------------------------------------------------------
    # AVALIAÇÃO DE ROTAS
    # ----------------------------------------------------------
    def avaliar_rota(self, rota):
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

    # ----------------------------------------------------------
    # ALGORITMO CLARKE & WRIGHT SAVINGS
    # ----------------------------------------------------------
    def criar_rotas_cw(self):
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

    # ----------------------------------------------------------
    # OTIMIZAÇÃO 2-OPT E BUSCA TABU
    # ----------------------------------------------------------
    def two_opt_optimize(self, routes):
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

    # ----------------------------------------------------------
    # BUSCA TABU (resumida)
    # ----------------------------------------------------------
    def tabu_search_vrptw(self, initial_routes, max_iterations=200, tabu_list_size=10):
        best_solution = self.copy_routes(initial_routes)
        best_cost = self.calculate_total_cost(best_solution)
        current_solution = self.copy_routes(initial_routes)
        tabu_list = []

        print(f"Custo Inicial (TS): R$ {best_cost:.2f}")

        for iteration in range(max_iterations):
            new_solution = self.two_opt_optimize(current_solution)
            new_cost = self.calculate_total_cost(new_solution)

            if new_cost < best_cost:
                best_cost = new_cost
                best_solution = new_solution
                print(f"Iteração {iteration+1}: NOVA MELHOR SOLUÇÃO! Custo: R$ {best_cost:.2f}")
            else:
                break

        print(f"\n--- Fim da Busca Tabu ---")
        print(f"Melhor custo encontrado: R$ {best_cost:.2f}")
        return best_solution

    # ----------------------------------------------------------
    # GERAÇÃO DE VISUALIZAÇÕES
    # ----------------------------------------------------------
    def gerar_visualizacoes(self, rotas_finais):
        plot_data = []
        rotas_df = []
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
            idx_veiculo += 1

        df_rotas = pd.DataFrame(rotas_df)
        fig = px.scatter_mapbox(
            df_rotas, lat="Lat", lon="Lon", hover_name="ID", color="Veiculo", size="Demanda", zoom=11,
            title="Rotas Otimizadas", mapbox_style="carto-positron"
        )
        for linha in plot_data:
            fig.add_trace(go.Scattermapbox(
                lat=linha['Lat'], lon=linha['Lon'],
                mode='lines', line=dict(width=2),
                name=f"Rota {linha['Veiculo']}", showlegend=False
            ))
        fig.add_trace(go.Scattermapbox(
            lat=[self.depot['Lat']], lon=[self.depot['Lon']],
            mode='markers', marker=dict(size=15, symbol='star', color='black'),
            name="Depósito", hovertext=["Depósito"]
        ))
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, showlegend=True)
        fig.write_html("rotas_otimizadas.html")

    # ----------------------------------------------------------
    # EXECUÇÃO COMPLETA
    # ----------------------------------------------------------
    def run_optimization(self, max_iterations=200, tabu_list_size=10):
        print("1. Criando rotas iniciais (Clark & Wright Savings)...")
        rotas_iniciais_cw = self.criar_rotas_cw()
        custo_inicial_cw = self.calculate_total_cost(rotas_iniciais_cw)
        print(f"   Custo após Savings: R$ {custo_inicial_cw:.2f}")

        print("2. Otimizando rotas iniciais com 2-opt intra-rota...")
        rotas_otimizadas_2opt = self.two_opt_optimize(rotas_iniciais_cw)
        custo_2opt = self.calculate_total_cost(rotas_otimizadas_2opt)
        print(f"   Custo após 2-opt: R$ {custo_2opt:.2f}")

        print(f"3. Executando Busca Tabu (Max Iter: {max_iterations}, Tabu Size: {tabu_list_size})...")
        rotas_finais = self.tabu_search_vrptw(rotas_otimizadas_2opt, 
                                              max_iterations=max_iterations, 
                                              tabu_list_size=tabu_list_size)

        print("4. Gerando visualizações (mapa e Gantt)...")
        self.gerar_visualizacoes(rotas_finais)

        custo_final = self.calculate_total_cost(rotas_finais)
        return custo_final


# ==========================================================
# EXECUÇÃO PRINCIPAL
# ==========================================================
if __name__ == '__main__':
    df_clientes = carregar_clientes_do_banco()

    CAPACIDADE_VEICULO = 20
    VELOCIDADE_MEDIA_KM_MIN = 1.0
    TEMPO_SERVICO_MIN = 15
    CUSTO_KM = 0.5
    CUSTO_MIN_OPERACAO = 1.0
    MULTA_ATRASO_POR_MIN = 5.0
    MAX_ITER = 200
    TABU_SIZE = 10

    print("--- Execução do Otimizador VRPTW ---")
    solver = VRPTWSolver(
        df_clientes, CAPACIDADE_VEICULO, VELOCIDADE_MEDIA_KM_MIN, 
        TEMPO_SERVICO_MIN, CUSTO_KM, CUSTO_MIN_OPERACAO, 
        MULTA_ATRASO_POR_MIN
    )

    custo_final = solver.run_optimization(MAX_ITER, TABU_SIZE)
    print(f"\nCUSTO TOTAL FINAL DA OTIMIZAÇÃO: R$ {custo_final:.2f}")
    print("\nArquivos de visualização (rotas_otimizadas.html) gerados na pasta de execução.")


    # ----------------------------------------------------------
    # Função para processar dados de formulário web
    # ----------------------------------------------------------
    def process_form_data(form_data):
        try:
            df_clientes = pd.DataFrame(form_data['clientes'])
            solver = VRPTWSolver(
                df_clientes, form_data['capacidade_veiculo'], form_data['velocidade_media_km_min'], 
                form_data['tempo_servico_min'], form_data['custo_km'], form_data['custo_min_operacao'], 
                form_data['multa_atraso_por_min']
            )
            custo_final = solver.run_optimization(form_data.get('max_iter', 200), form_data.get('tabu_size', 10))
            return custo_final
        except Exception as e:
            print(f"Erro ao processar os dados do formulário: {e}")
            return None
