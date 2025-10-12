from flask import Flask, request, jsonify, send_from_directory, render_template
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import copy
import plotly.io as pio
import os

# Define o renderizador do Plotly - Comentado para ambiente Flask
# pio.renderers.default = "vscode" 

# ==========================================================
# FUNÇÕES DB_UTILS
# ==========================================================
DB_NAME = 'clientes.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def garantir_tabela_clientes():
    conn = get_db_connection()
    try:
        # Não vou dropar a tabela em cada run, mas em caso de problema pode descomentar
        # conn.execute('DROP TABLE IF EXISTS clientes')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS clientes (
                ID TEXT PRIMARY KEY,
                Lat REAL NOT NULL,
                Lon REAL NOT NULL,
                Demanda INTEGER NOT NULL,
                T_Inicio_h REAL NOT NULL,
                T_Fim_h REAL NOT NULL
            )
        ''')
        conn.commit()
    finally:
        conn.close()

def inserir_clientes_iniciais():
    clientes_iniciais = [
        ('C', -23.550520, -46.633308, 0, 8.0, 18.0), # Depósito (Centro de SP)
        ('C1', -23.5412, -46.6358, 5, 8.0, 10.0),
        ('C2', -23.5450, -46.6400, 10, 9.0, 11.0),
        ('C3', -23.5550, -46.6300, 3, 10.0, 12.0),
        ('C4', -23.5600, -46.6380, 7, 12.0, 14.0),
    ]
    conn = get_db_connection()
    try:
        # Usar INSERT OR IGNORE para não recriar os clientes se já existirem
        conn.executemany('INSERT OR IGNORE INTO clientes VALUES (?, ?, ?, ?, ?, ?)', clientes_iniciais)
        conn.commit()
    finally:
        conn.close()

def carregar_clientes_do_banco():
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM clientes", conn)
        return df
    finally:
        conn.close()

def inserir_novo_cliente(cliente_data):
    conn = get_db_connection()
    try:
        # Usa INSERT OR REPLACE para permitir atualização se o ID já existir
        conn.execute('''
            INSERT OR REPLACE INTO clientes (ID, Lat, Lon, Demanda, T_Inicio_h, T_Fim_h)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            cliente_data['ID'], float(cliente_data['Lat']), float(cliente_data['Lon']),
            int(cliente_data['Demanda']), float(cliente_data['T_Inicio_h']), float(cliente_data['T_Fim_h'])
        ))
        conn.commit()
        return "Cliente inserido/atualizado com sucesso!"
    except Exception as e:
        return f"Erro ao manipular cliente no DB: {str(e)}"
    finally:
        conn.close()
        
# ==========================================================
# CLASSE PRINCIPAL DE OTIMIZAÇÃO VRPTW
# ==========================================================
class VRPTWSolver:
    def __init__(self, df_clientes, capacidade_veiculo, velocidade_media_km_min, 
                 tempo_servico_min, custo_km, custo_min_operacao, 
                 multa_atraso_por_min):
        
        self.df_clientes = df_clientes
        
        # Encontra o índice do depósito (ID='C')
        depot_row = df_clientes[df_clientes['ID'] == 'C']
        if depot_row.empty:
            raise ValueError("Depósito (ID='C') não encontrado no DataFrame de clientes.")
            
        self.DEPOT_INDEX = depot_row.iloc[0].name 
        self.depot = depot_row.iloc[0] 
        
        self.CAPACIDADE_VEICULO = capacidade_veiculo
        self.VELOCIDADE_MEDIA_KM_MIN = max(0.001, velocidade_media_km_min) 
        self.TEMPO_SERVICO_MIN = tempo_servico_min
        self.CUSTO_KM = custo_km
        self.CUSTO_MIN_OPERACAO = custo_min_operacao
        self.MULTA_ATRASO_POR_MIN = multa_atraso_por_min
        
        # Conversão de horas para minutos para os TWs do depósito
        self.START_DEPOT_MIN = self.df_clientes.loc[self.DEPOT_INDEX, 'T_Inicio_h'] * 60
        self.END_DEPOT_MIN = self.df_clientes.loc[self.DEPOT_INDEX, 'T_Fim_h'] * 60

        self.dist_matrix = self._calcular_matriz_distancia()
        self.n_clientes = len(df_clientes)

    # ----------------------------------------------------------
    # MATRIZ DE DISTÂNCIA (Haversine)
    # ----------------------------------------------------------
    def _calcular_matriz_distancia(self):
        n_clientes = len(self.df_clientes)
        dist_matrix = np.zeros((n_clientes, n_clientes))
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371 # Raio da Terra em km
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
        """Copia o dicionário de rotas."""
        return {k: v[:] for k, v in routes.items()}

    # ----------------------------------------------------------
    # AVALIAÇÃO DE ROTAS (Custo e Validade)
    # ----------------------------------------------------------
    def avaliar_rota(self, rota):
        """Avalia o custo de uma única rota, verificando capacidade e TW."""
        depot_idx = self.DEPOT_INDEX
        
        demanda_rota = sum(self.df_clientes.iloc[no]['Demanda'] for no in rota if no != depot_idx)
        if demanda_rota > self.CAPACIDADE_VEICULO:
            return (999999, False, False) # Capacidade violada

        custo_total = 0
        tempo_atual_min = self.START_DEPOT_MIN
        tw_ok = True # Flag para Time Window

        for i in range(len(rota) - 1):
            origem = rota[i]
            destino = rota[i+1]
            distancia_km = self.dist_matrix[origem, destino]
            tempo_viagem_min = distancia_km / self.VELOCIDADE_MEDIA_KM_MIN
            tempo_chegada = tempo_atual_min + tempo_viagem_min
            custo_total += (distancia_km * self.CUSTO_KM) + (tempo_viagem_min * self.CUSTO_MIN_OPERACAO)

            if destino != depot_idx: # Se for um cliente
                t_inicio_min = self.df_clientes.iloc[destino]['T_Inicio_h'] * 60
                t_fim_min = self.df_clientes.iloc[destino]['T_Fim_h'] * 60
                multa_atraso = 0

                if tempo_chegada < t_inicio_min:
                    # Espera (chegou cedo)
                    tempo_inicio_servico = t_inicio_min
                elif tempo_chegada > t_fim_min:
                    # Multa por atraso (violou TW)
                    atraso_min = tempo_chegada - t_fim_min
                    multa_atraso = atraso_min * self.MULTA_ATRASO_POR_MIN
                    tempo_inicio_servico = tempo_chegada
                    tw_ok = False 
                else:
                    tempo_inicio_servico = tempo_chegada

                custo_total += multa_atraso
                tempo_saida = tempo_inicio_servico + self.TEMPO_SERVICO_MIN
                tempo_atual_min = tempo_saida
            else: # Se for o depósito de volta
                if tempo_chegada > self.END_DEPOT_MIN:
                    # Retorno ao depósito fora do TW final
                    return (999999, True, False)
                tempo_atual_min = tempo_chegada

        return (custo_total, True, tw_ok)

    def calculate_total_cost(self, routes):
        """Calcula o custo total de todas as rotas."""
        total_cost = 0
        if not routes: return 999999
        for rota in routes.values():
            cost, capacidade_ok, tw_ok = self.avaliar_rota(rota)
            if cost == 999999 or not capacidade_ok or not tw_ok:
                return 999999 
            total_cost += cost
        return total_cost

    # ----------------------------------------------------------
    # ALGORITMO CLARKE & WRIGHT SAVINGS
    # ----------------------------------------------------------
    def criar_rotas_cw(self):
        """Implementação do algoritmo Clarke & Wright Savings."""
        depot_idx = self.DEPOT_INDEX
        rotas = {}
        
        # 1. Inicializa uma rota [Depot, Cliente, Depot] para cada cliente
        for i in self.df_clientes[self.df_clientes['ID'] != 'C'].index:
            if self.avaliar_rota([depot_idx, i, depot_idx])[0] != 999999:
                # O ID da rota é o ID do primeiro (e único) cliente
                rotas[i] = [depot_idx, i, depot_idx] 
            else:
                 print(f"ATENÇÃO: Cliente {self.df_clientes.loc[i, 'ID']} não pode ser atendido em rota individual (TW ou Capacidade). Será ignorado.")

        economias = []
        indices_clientes = [i for i in rotas.keys()]

        # 2. Calcula todas as economias
        for i in indices_clientes:
            for j in indices_clientes:
                if i < j:
                    # Economia em Distância (km)
                    s_ij = self.dist_matrix[depot_idx, i] + self.dist_matrix[j, depot_idx] - self.dist_matrix[i, j]
                    economias.append((s_ij, i, j))

        economias.sort(key=lambda x: x[0], reverse=True)
        rotas_finais_cw = rotas.copy()
        
        # 3. Une rotas, verificando validade
        for _, i, j in economias:
            # Encontra as rotas que contêm os clientes i e j
            rota_i_id = next((k for k, v in rotas_finais_cw.items() if i in v and len(v)>2), None)
            rota_j_id = next((k for k, v in rotas_finais_cw.items() if j in v and len(v)>2), None)
            
            # Se for a mesma rota ou se um dos clientes já foi absorvido em uma rota não inicial
            if rota_i_id != rota_j_id and rota_i_id is not None and rota_j_id is not None:
                rota_i = rotas_finais_cw[rota_i_id]
                rota_j = rotas_finais_cw[rota_j_id]

                # Tenta união (Rota i + Rota j)
                if rota_i[-2] == i and rota_j[1] == j:
                    nova_rota = rota_i[:-1] + rota_j[1:]
                    _, capacidade_ok, tw_ok = self.avaliar_rota(nova_rota)

                    if capacidade_ok and tw_ok:
                        rotas_finais_cw[rota_i_id] = nova_rota
                        del rotas_finais_cw[rota_j_id]
                        continue

                # Tenta união (Rota j + Rota i) - Opcional, dependendo da definição de CW
                # A implementação original do CW só permite (A,i) + (j,B) -> (A,i,j,B)
                # A união acima (i + j) já tenta i no final e j no início.
                # A junção abaixo garante a união da ponta de 'j' (j=penultimo) e o início de 'i' (i=segundo)
                if rota_j[-2] == j and rota_i[1] == i:
                    nova_rota = rota_j[:-1] + rota_i[1:]
                    _, capacidade_ok, tw_ok = self.avaliar_rota(nova_rota)

                    if capacidade_ok and tw_ok:
                        rotas_finais_cw[rota_j_id] = nova_rota
                        del rotas_finais_cw[rota_i_id]
                        continue

        return rotas_finais_cw

    # ----------------------------------------------------------
    # OTIMIZAÇÃO 2-OPT (Intra-Rota)
    # ----------------------------------------------------------
    def two_opt_optimize(self, routes):
        """Aplica a heurística 2-opt em cada rota individualmente (intra-rota) para melhoria local."""
        optimized_routes = {}
        
        for route_id, route in routes.items():
            best_route = route
            best_cost, _, _ = self.avaliar_rota(best_route)
            
            melhoria_encontrada = True
            while melhoria_encontrada:
                melhoria_encontrada = False
                n = len(best_route)
                if n < 4: break 

                # Os endpoints 0 (depôt de saída) e n-1 (depôt de retorno) não podem ser movidos
                for i in range(1, n - 2):
                    for k in range(i + 1, n - 1):
                        # 2-opt: reverte a sub-rota entre i e k
                        # new_route = [..., best_route[i-1], best_route[k], ..., best_route[i], best_route[k+1], ...]
                        new_route = best_route[:i] + best_route[i:k+1][::-1] + best_route[k+1:]
                        
                        new_cost, capacidade_ok, tw_ok = self.avaliar_rota(new_route)
                        
                        if new_cost < best_cost and new_cost != 999999: # Verifica se é melhor e válido
                            best_route = new_route
                            best_cost = new_cost
                            melhoria_encontrada = True
                            break # Reinicia o loop while
                    if melhoria_encontrada: break
            
            optimized_routes[route_id] = best_route
        return optimized_routes

    # ----------------------------------------------------------
    # BUSCA TABU (Simplificada)
    # ----------------------------------------------------------
    def tabu_search_vrptw(self, initial_routes, max_iterations=200, tabu_list_size=10):
        """Busca Tabu simplificada (usa 2-opt como movimento de vizinhança)."""
        best_solution = self.copy_routes(initial_routes)
        best_cost = self.calculate_total_cost(best_solution)
        current_solution = self.copy_routes(initial_routes)
        
        print(f"Custo Inicial (TS): R$ {best_cost:.2f}")

        # A implementação simplificada usa 2-opt como um vizinho de melhoria
        # A lista Tabu é omitida aqui porque o movimento é baseado em uma heurística local
        # intra-rota, não em um movimento inter-rota (swap/relocate) que exigiria a Tabu List.
        # O loop principal é mantido para dar a chance de melhorias incrementais.
        for iteration in range(max_iterations):
            
            # Gera vizinho aplicando 2-opt (melhoria local intra-rota)
            # Como a 2-opt já itera até a melhoria local, isso funciona como uma busca de vizinhança forte.
            new_solution_candidate = self.two_opt_optimize(current_solution) 
            current_cost = self.calculate_total_cost(new_solution_candidate)

            # Se o custo for igual, tenta parar para evitar loop, mas continua para a iteração.
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = new_solution_candidate
                current_solution = new_solution_candidate
                print(f"Iteração {iteration+1}: NOVA MELHOR SOLUÇÃO! Custo: R$ {best_cost:.2f}")
            elif current_cost == 999999:
                 print(f"Iteração {iteration+1}: Solução inválida. Parando Busca Tabu.")
                 break
            else:
                # Se não houver melhora, mantém a solução anterior (Busca Tabu não-intensiva)
                current_solution = new_solution_candidate
            
            # Critério de Parada Simples
            if iteration > 0 and iteration % 50 == 0:
                 print(f"Iteração {iteration+1}: Verificando critério de parada...")
                 # Poderíamos adicionar aqui a lógica de Aceitação/Aspiração Tabu para o caso inter-rota
        
        print(f"--- Fim da Busca Tabu ---")
        return best_solution

    # ----------------------------------------------------------
    # GERAÇÃO DE VISUALIZAÇÕES (MAPA)
    # ----------------------------------------------------------
    def gerar_visualizacoes(self, rotas_finais):
        plot_data = []
        rotas_df = []
        idx_veiculo = 1
        depot_idx = self.DEPOT_INDEX

        for rota_id, rota in rotas_finais.items():
            if len(rota) < 3: continue # Ignora rotas vazias ou só com o depósito

            # Cálculo de dados para linhas de rota
            for i in range(len(rota) - 1):
                origem = rota[i]
                destino = rota[i+1]
                ponto_a = self.df_clientes.iloc[origem]
                ponto_b = self.df_clientes.iloc[destino]
                
                # Dados para o mapa (linhas)
                plot_data.append({
                    'Veiculo': f"V{idx_veiculo}",
                    'Lat': [ponto_a['Lat'], ponto_b['Lat']],
                    'Lon': [ponto_a['Lon'], ponto_b['Lon']],
                    'Ordem': i
                })
                
            # Dados para o mapa (clientes)
            for no in rota[1:-1]:
                cliente = self.df_clientes.iloc[no].copy()
                cliente['Veiculo'] = f"V{idx_veiculo}"
                rotas_df.append(cliente)
                
            idx_veiculo += 1

        df_rotas = pd.DataFrame(rotas_df)
        
        if df_rotas.empty and not self.df_clientes.empty:
             # Adiciona o depósito se não houverem clientes nas rotas, mas existirem dados
             depot_data = self.depot.to_dict()
             depot_data['Veiculo'] = 'Depósito'
             df_rotas = pd.DataFrame([depot_data])
        elif df_rotas.empty:
            print("\nAVISO: Não foram encontradas rotas válidas com clientes para visualização. Arquivo HTML do mapa não gerado.")
            return

        fig = px.scatter_map(
            df_rotas[df_rotas['ID'] != 'C'], lat="Lat", lon="Lon", hover_name="ID", 
            color="Veiculo", size="Demanda", zoom=11,
            title="Rotas Otimizadas", map_style="carto-positron" )
        
        # Adiciona as linhas (rotas)
        cores_veiculos = fig.layout.coloraxis.color_scale if hasattr(fig.layout, 'coloraxis') else px.colors.qualitative.Plotly
        cores_rotas = {f"V{i+1}": cores_veiculos[i % len(cores_veiculos)] for i in range(idx_veiculo)}

        for linha in plot_data:
            cor = cores_rotas.get(linha['Veiculo'], 'gray')
            fig.add_trace(go.Scattermap( 
                lat=linha['Lat'], lon=linha['Lon'],
                mode='lines', line=dict(width=3, color=cor),
                name=f"Rota {linha['Veiculo']}", showlegend=False
            ))
            
        # Adiciona o depósito como estrela
        fig.add_trace(go.Scattermap( 
            lat=[self.depot['Lat']], lon=[self.depot['Lon']],
            mode='markers', marker=dict(size=15, symbol='star', color='black'),
            name="Depósito", hovertext=[f"Depósito ({self.depot['ID']})"], showlegend=True
        ))
        
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, showlegend=True)
        fig.write_html("rotas_otimizadas.html")
        
    # ----------------------------------------------------------
    # GERAÇÃO DO GRÁFICO DE GANTT
    # ----------------------------------------------------------
    def gerar_gantt(self, rotas_finais):
        """Calcula o cronograma e gera o gráfico de Gantt."""
        gantt_data = []
        depot_idx = self.DEPOT_INDEX

        for idx_veiculo, rota in enumerate(rotas_finais.values()):
            if len(rota) < 2: continue
            
            veiculo_id = f"V{idx_veiculo + 1}"
            tempo_atual_min = self.START_DEPOT_MIN # Tempo de início do depósito (em minutos)

            # Log de Saída inicial
            gantt_data.append({
                'Veiculo': veiculo_id, 'Tarefa': f"Saída (Depósito)", 
                'Start': tempo_atual_min, 'Finish': tempo_atual_min, 
                'Duration': 0, 'Tipo': 'Saída', 'Color': '#000000'
            })
            
            # Loop principal de cálculo
            for i in range(len(rota) - 1):
                origem_idx = rota[i]
                destino_idx = rota[i+1]
                
                distancia_km = self.dist_matrix[origem_idx, destino_idx]
                tempo_viagem_min = distancia_km / self.VELOCIDADE_MEDIA_KM_MIN
                tempo_chegada = tempo_atual_min + tempo_viagem_min
                
                # Log de Viagem
                gantt_data.append({
                    'Veiculo': veiculo_id,
                    'Tarefa': f"Viagem de {self.df_clientes.iloc[origem_idx]['ID']} para {self.df_clientes.iloc[destino_idx]['ID']}",
                    'Start': tempo_atual_min, 'Finish': tempo_chegada,
                    'Duration': tempo_viagem_min, 'Tipo': 'Viagem', 'Color': '#FFD700'
                })

                # Log de Serviço/Espera/Retorno no Destino
                if destino_idx != depot_idx:
                    cliente = self.df_clientes.iloc[destino_idx]
                    t_inicio_min = cliente['T_Inicio_h'] * 60
                    t_fim_min = cliente['T_Fim_h'] * 60
                    
                    tempo_inicio_servico = max(tempo_chegada, t_inicio_min)
                    tempo_espera = tempo_inicio_servico - tempo_chegada
                    
                    # Log da Espera (se houver)
                    if tempo_espera > 0.01:
                         gantt_data.append({
                              'Veiculo': veiculo_id, 'Tarefa': f"Espera em {cliente['ID']}",
                              'Start': tempo_chegada, 'Finish': tempo_inicio_servico,
                              'Duration': tempo_espera, 'Tipo': 'Espera', 'Color': '#ADD8E6'
                         })

                    tempo_fim_servico = tempo_inicio_servico + self.TEMPO_SERVICO_MIN

                    # Verifica violação de TW (já tratada na avaliação, mas bom para visualização)
                    servico_color = '#20B2AA'
                    if tempo_inicio_servico > t_fim_min:
                        servico_color = '#FF0000' # Vermelho se atrasado
                    
                    # Log do Serviço
                    gantt_data.append({
                        'Veiculo': veiculo_id,
                        'Tarefa': f"Serviço em {cliente['ID']} ({int(cliente['Demanda'])} un.)",
                        'Start': tempo_inicio_servico, 'Finish': tempo_fim_servico,
                        'Duration': self.TEMPO_SERVICO_MIN, 'Tipo': 'Serviço', 'Color': servico_color
                    })
                    
                    tempo_atual_min = tempo_fim_servico
                else:
                    # Depósito de Retorno
                    gantt_data.append({
                        'Veiculo': veiculo_id, 'Tarefa': f"Retorno ao Depósito",
                        'Start': tempo_chegada, 'Finish': tempo_chegada,
                        'Duration': 0, 'Tipo': 'Retorno', 'Color': '#808080'
                    })
                    tempo_atual_min = tempo_chegada

        # --- Passo 2: Geração do Plotly ---
        final_gantt_df = pd.DataFrame(gantt_data)
        
        if final_gantt_df.empty:
            print("\nAVISO: Não foi possível calcular o cronograma Gantt. Arquivo HTML do Gantt não gerado.")
            return

        final_gantt_df['Start_h'] = final_gantt_df['Start'] / 60
        final_gantt_df['Finish_h'] = final_gantt_df['Finish'] / 60
        
        color_map = {
            'Viagem': '#FFD700', 'Serviço': '#20B2AA', 'Espera': '#ADD8E6', 
            'Retorno': '#808080', 'Saída': '#000000'
        }
        
        # Garante que as cores de serviço atrasado (vermelho) sejam incluídas no mapa de cores.
        if '#FF0000' not in final_gantt_df['Color'].unique():
             color_map['Serviço Atrasado'] = '#FF0000'

        fig = px.timeline(final_gantt_df[final_gantt_df['Duration'] > 0.001], 
                          x_start="Start_h", x_end="Finish_h", y="Veiculo", 
                          color="Tipo", color_discrete_map=color_map, # Usa o campo Tipo para o mapeamento
                          title="Cronograma de Atendimento (Gráfico de Gantt)",
                          hover_data={'Duration': ':.1f', 'Start_h': ':.2f', 'Finish_h': ':.2f', 'Tarefa': True})
        
        fig.update_yaxes(categoryorder="category descending")
        fig.update_layout(xaxis_title="Tempo (Horas)")
        
        fig.write_html("gantt_schedule.html")
        
        return "gantt_schedule.html"

    # ----------------------------------------------------------
    # EXECUÇÃO COMPLETA
    # ----------------------------------------------------------
    def run_optimization(self, max_iterations=200, tabu_list_size=10):
        print("\n--- INÍCIO DA OTIMIZAÇÃO VRPTW ---")
        
        clientes_validos = self.df_clientes[self.df_clientes['ID'] != 'C']
        if clientes_validos.empty:
            print("AVISO: Nenhum cliente encontrado para otimização.")
            return {}

        print("1. Criando rotas iniciais (Clark & Wright Savings)...")
        rotas_iniciais_cw = self.criar_rotas_cw()
        custo_inicial_cw = self.calculate_total_cost(rotas_iniciais_cw)
        print(f"    Custo após Savings: R$ {custo_inicial_cw:.2f}")

        if custo_inicial_cw == 999999 or not rotas_iniciais_cw:
            print("AVISO: Solução inicial inválida/inviável. Não é possível otimizar.")
            self.gerar_visualizacoes({}) # Gera mapa vazio ou apenas com depósito
            return rotas_iniciais_cw

        print("2. Otimizando rotas iniciais com 2-opt intra-rota...")
        rotas_otimizadas_2opt = self.two_opt_optimize(rotas_iniciais_cw)
        custo_2opt = self.calculate_total_cost(rotas_otimizadas_2opt)
        print(f"    Custo após 2-opt: R$ {custo_2opt:.2f}")

        print(f"3. Executando Busca Tabu (Max Iter: {max_iterations})...") 
        rotas_finais = self.tabu_search_vrptw(rotas_otimizadas_2opt, 
                                             max_iterations=max_iterations, 
                                             tabu_list_size=tabu_list_size)

        print("4. Gerando visualizações (mapa e Gantt)...")
        self.gerar_visualizacoes(rotas_finais)
        self.gerar_gantt(rotas_finais)
        
        print("--- FIM DA OTIMIZAÇÃO VRPTW ---")
        return rotas_finais

# ==========================================================
# APLICAÇÃO FLASK
# ==========================================================
app = Flask(__name__)

# Parâmetros padrão do problema
DEFAULT_PARAMS = {
    'capacidade_veiculo': 20,
    'velocidade_media_km_min': 1.0,
    'tempo_servico_min': 15,
    'custo_km': 0.5,
    'custo_min_operacao': 1.0,
    'multa_atraso_por_min': 5.0,
    'max_iter': 200,
    'tabu_size': 10
}

# Inicializa o banco de dados e carrega dados iniciais
with app.app_context():
    garantir_tabela_clientes()
    inserir_clientes_iniciais()
    # Chama a otimização inicial para gerar os arquivos HTML de primeira vez
    try:
        df_clientes = carregar_clientes_do_banco()
        solver_init = VRPTWSolver(
            df_clientes, 
            DEFAULT_PARAMS['capacidade_veiculo'], 
            DEFAULT_PARAMS['velocidade_media_km_min'], 
            DEFAULT_PARAMS['tempo_servico_min'], 
            DEFAULT_PARAMS['custo_km'], 
            DEFAULT_PARAMS['custo_min_operacao'], 
            DEFAULT_PARAMS['multa_atraso_por_min']
        )
        solver_init.run_optimization(max_iterations=1, tabu_list_size=1) # Run rápido
    except Exception as e:
        print(f"Aviso: Falha na otimização inicial: {e}")

@app.route('/')
def index():
    """Renderiza a página HTML do dashboard. Assume 'dashboard.html' ou 'index.html' na pasta 'templates'."""
    # Usando 'index.html' por convenção do Flask, mas você pode renomear para 'dashboard.html'
    return render_template('index.html') 

@app.route('/optimize', methods=['POST'])
def optimize_routes():
    """Recebe parâmetros do frontend, executa a otimização."""
    try:
        data = request.get_json()
        
        df_clientes = carregar_clientes_do_banco()
        
        # Extrair parâmetros (usando defaults se faltar algo)
        capacidade = float(data.get('capacity', DEFAULT_PARAMS['capacidade_veiculo']))
        velocidade = float(data.get('speed', DEFAULT_PARAMS['velocidade_media_km_min']))
        custo_km = float(data.get('cost_km', DEFAULT_PARAMS['custo_km']))
        
        # Parâmetros fixos
        tempo_servico = DEFAULT_PARAMS['tempo_servico_min']
        custo_min_op = DEFAULT_PARAMS['custo_min_operacao']
        multa_atraso = DEFAULT_PARAMS['multa_atraso_por_min']
        max_iter = int(data.get('max_iter', DEFAULT_PARAMS['max_iter']))
        tabu_size = int(data.get('tabu_size', DEFAULT_PARAMS['tabu_size']))

        solver = VRPTWSolver(
            df_clientes, capacidade, velocidade, tempo_servico, 
            custo_km, custo_min_op, multa_atraso
        )
        
        rotas_finais = solver.run_optimization(max_iter, tabu_size)
        final_cost = solver.calculate_total_cost(rotas_finais)

        if final_cost >= 999999:
            return jsonify({'status': 'error', 'message': 'Nenhuma solução válida pôde ser encontrada com os parâmetros fornecidos.'}), 500
        
        return jsonify({
            'status': 'success',
            'final_cost': f"R$ {final_cost:.2f}",
            'routes_count': len(rotas_finais),
            'message': 'Otimização concluída com sucesso!'
        })

    except Exception as e:
        app.logger.error(f"Erro na otimização: {e}")
        return jsonify({'status': 'error', 'message': f"Erro interno do servidor: {str(e)}"}), 500

@app.route('/rotas_otimizadas')
def serve_map():
    """Serve o arquivo HTML do mapa gerado pelo Plotly."""
    # O arquivo gerado está na raiz da aplicação Flask (os.getcwd())
    return send_from_directory(os.getcwd(), 'rotas_otimizadas.html')

@app.route('/gantt_schedule')
def serve_gantt():
    """Serve o arquivo HTML do Gantt gerado pelo Plotly."""
    return send_from_directory(os.getcwd(), 'gantt_schedule.html')

@app.route('/api/novo_cliente', methods=['POST'])
def add_new_client():
    """Recebe dados de um novo cliente e o insere/atualiza no banco de dados, e reotimiza."""
    try:
        cliente_data = request.get_json()
        message = inserir_novo_cliente(cliente_data)
        
        # Dispara a otimização imediatamente após adicionar/atualizar
        # Não precisa retornar o custo, apenas informa que a otimização foi disparada
        
        # Simula a chamada para otimizar com parâmetros default (não retorna custo para o frontend)
        df_clientes = carregar_clientes_do_banco()
        solver = VRPTWSolver(
            df_clientes, 
            DEFAULT_PARAMS['capacidade_veiculo'], 
            DEFAULT_PARAMS['velocidade_media_km_min'], 
            DEFAULT_PARAMS['tempo_servico_min'], 
            DEFAULT_PARAMS['custo_km'], 
            DEFAULT_PARAMS['custo_min_operacao'], 
            DEFAULT_PARAMS['multa_atraso_por_min']
        )
        solver.run_optimization(max_iterations=10, tabu_list_size=1) # Otimização mais rápida

        return jsonify({'status': 'success', 'message': f"{message} Nova otimização disparada (10 iter). Favor atualizar o painel."}), 200

    except Exception as e:
        app.logger.error(f"Erro ao adicionar cliente e reotimizar: {e}")
        return jsonify({'status': 'error', 'message': f"Erro ao adicionar cliente: {str(e)}"}), 500

if __name__ == '__main__':
    print("Iniciando o servidor Flask VRPTW. Acesse http://127.0.0.1:5000/")
    
    # IMPORTANTE: Garanta que o arquivo HTML (index.html ou dashboard.html)
    # esteja na pasta 'templates' para que o 'render_template' funcione.
    
    app.run(debug=True)