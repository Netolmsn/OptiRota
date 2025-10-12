import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import copy 
import plotly.io as pio

# ✅ Importação do módulo db_utils (Certifique-se que db_utils.py está na mesma pasta)
# Você precisará criar o db_utils.py, que deve conter as funções:
# garantir_tabela_clientes, inserir_clientes, e carregar_clientes_do_banco
try:
    from db_utils import garantir_tabela_clientes, inserir_clientes, carregar_clientes_do_banco
except ImportError:
    print("AVISO: O arquivo 'db_utils.py' não foi encontrado. O código não rodará no bloco '__main__'.")
    # Para rodar isoladamente sem o banco, você precisará carregar um DataFrame df_clientes manualmente aqui.

# Define o renderizador do Plotly (Remova ou altere se não estiver usando VSCode/Jupyter)
pio.renderers.default = "vscode"


# ==========================================================
# CLASSE PRINCIPAL DE OTIMIZAÇÃO VRPTW
# ==========================================================
class VRPTWSolver:
    def __init__(self, df_clientes, capacidade_veiculo, velocidade_media_km_min, 
                 tempo_servico_min, custo_km, custo_min_operacao, 
                 multa_atraso_por_min):
        
        self.df_clientes = df_clientes
        
        # Acessa o depósito pelo ID 'C' de forma robusta e pega seu índice
        depot_row = df_clientes[df_clientes['ID'] == 'C'].iloc[0]
        self.DEPOT_INDEX = depot_row.name 
        self.depot = depot_row 
        
        self.CAPACIDADE_VEICULO = capacidade_veiculo
        # Garante que a velocidade não é zero para evitar divisão por zero
        self.VELOCIDADE_MEDIA_KM_MIN = max(0.001, velocidade_media_km_min) 
        self.TEMPO_SERVICO_MIN = tempo_servico_min
        self.CUSTO_KM = custo_km
        self.CUSTO_MIN_OPERACAO = custo_min_operacao
        self.MULTA_ATRASO_POR_MIN = multa_atraso_por_min
        
        # Usa o índice correto do depósito
        self.START_DEPOT_MIN = self.df_clientes.loc[self.DEPOT_INDEX, 'T_Inicio_h'] * 60
        self.END_DEPOT_MIN = self.df_clientes.loc[self.DEPOT_INDEX, 'T_Fim_h'] * 60

        self.dist_matrix = self._calcular_matriz_distancia()
        self.n_clientes = len(df_clientes)

    # ----------------------------------------------------------
    # MATRIZ DE DISTÂNCIA
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
    # AVALIAÇÃO DE ROTAS
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
            # Se qualquer rota for inválida (custo 999999 ou violação), o custo total é 999999
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
            # Garante que a rota individual é válida ANTES de incluí-la
            if self.avaliar_rota([depot_idx, i, depot_idx])[0] != 999999:
                rotas[i] = [depot_idx, i, depot_idx]
            else:
                 print(f"ATENÇÃO: Cliente {self.df_clientes.loc[i, 'ID']} não pode ser atendido em rota individual (TW ou Capacidade). Será ignorado.")


        economias = []
        indices_clientes = [i for i in rotas.keys()]

        # 2. Calcula todas as economias
        for i in indices_clientes:
            for j in indices_clientes:
                if i < j:
                    s_ij = self.dist_matrix[depot_idx, i] + self.dist_matrix[j, depot_idx] - self.dist_matrix[i, j]
                    economias.append((s_ij, i, j))

        economias.sort(key=lambda x: x[0], reverse=True)
        rotas_finais_cw = rotas.copy()
        
        # 3. Une rotas, verificando validade
        for _, i, j in economias:
            # Encontra a chave da rota que contém o cliente i e j
            rota_i_id = next((k for k, v in rotas_finais_cw.items() if i in v), None)
            rota_j_id = next((k for k, v in rotas_finais_cw.items() if j in v), None)

            if rota_i_id != rota_j_id and rota_i_id is not None and rota_j_id is not None:
                rota_i = rotas_finais_cw[rota_i_id]
                rota_j = rotas_finais_cw[rota_j_id]

                # Condição de junção: i deve ser o penúltimo na rota i, e j deve ser o segundo na rota j
                if rota_i[-2] == i and rota_j[1] == j:
                    nova_rota = rota_i[:-1] + rota_j[1:]
                    
                    _, capacidade_ok, tw_ok = self.avaliar_rota(nova_rota)

                    if capacidade_ok and tw_ok:
                        rotas_finais_cw[rota_i_id] = nova_rota
                        del rotas_finais_cw[rota_j_id]
                        
        return rotas_finais_cw

    # ----------------------------------------------------------
    # OTIMIZAÇÃO 2-OPT
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

                for i in range(1, n - 2):
                    for k in range(i + 1, n - 1):
                        # 2-opt: reverte a sub-rota entre i e k
                        new_route = best_route[:i] + best_route[k:i-1:-1] + best_route[k+1:]
                        
                        new_cost, capacidade_ok, tw_ok = self.avaliar_rota(new_route)
                        
                        if new_cost < best_cost and new_cost != 999999: # Verifica se é melhor e válido
                            best_route = new_route
                            best_cost = new_cost
                            melhoria_encontrada = True
                            break 
                    if melhoria_encontrada: break
            
            optimized_routes[route_id] = best_route
        return optimized_routes

    # ----------------------------------------------------------
    # BUSCA TABU (REVISADA)
    # ----------------------------------------------------------
    def tabu_search_vrptw(self, initial_routes, max_iterations=200, tabu_list_size=10):
        """Busca Tabu simplificada (usa 2-opt como movimento de vizinhança)."""
        # A implementação aqui é simplificada e não usa a lista tabu, 
        # mas foca na melhoria iterativa pelo 2-opt (melhoria local repetida).
        # Para uma Busca Tabu completa, seria necessário implementar movimentos inter-rota (troca de clientes)
        # e a lista tabu para evitar revisitar soluções.
        
        best_solution = self.copy_routes(initial_routes)
        best_cost = self.calculate_total_cost(best_solution)
        current_solution = self.copy_routes(initial_routes)
        
        print(f"Custo Inicial (TS): R$ {best_cost:.2f}")

        for iteration in range(max_iterations):
            # O vizinho é gerado pela melhoria local do 2-opt (intra-rota)
            # Isto serve como um mecanismo de "melhoria local" dentro do loop da Tabu Search
            new_solution_candidate = self.two_opt_optimize(current_solution) 
            current_cost = self.calculate_total_cost(new_solution_candidate)

            # A solução atual avança para o melhor vizinho encontrado
            current_solution = new_solution_candidate
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution
                print(f"Iteração {iteration+1}: NOVA MELHOR SOLUÇÃO! Custo: R$ {best_cost:.2f}")
            elif current_cost == 999999:
                 print(f"Iteração {iteration+1}: Solução inválida. Parando Busca Tabu.")
                 break
            
            # Condição de parada de estagnação simples
            if iteration > 0 and iteration % 50 == 0 and best_cost == self.calculate_total_cost(current_solution):
                 print(f"Iteração {iteration+1}: Nenhum progresso nas últimas 50 iterações. Parando...")
                 break

        print(f"\n--- Fim da Busca Tabu ---")
        print(f"Melhor custo encontrado: R$ {best_cost:.2f}")
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
                
            # Dados para o mapa (clientes): Apenas nós entre o depósito inicial e final
            for no in rota[1:-1]:
                cliente = self.df_clientes.iloc[no].copy()
                cliente['Veiculo'] = f"V{idx_veiculo}"
                rotas_df.append(cliente)
                
            idx_veiculo += 1

        df_rotas = pd.DataFrame(rotas_df)
        
        # Garante que o DataFrame não está vazio antes de plotar
        if df_rotas.empty:
            print("\nAVISO: Não foram encontradas rotas válidas com clientes para visualização. Arquivo HTML do mapa não gerado.")
            return

        # ✅ CORREÇÃO FINAL: Removido o 'asterisco duplo' (**) da frente do map_style
        fig = px.scatter_map(
            df_rotas, lat="Lat", lon="Lon", hover_name="ID", color="Veiculo", size="Demanda", zoom=11,
            title="Rotas Otimizadas", map_style="carto-positron" )
        
        # Adiciona as linhas (rotas)
        for linha in plot_data:
            fig.add_trace(go.Scattermap( 
                lat=linha['Lat'], lon=linha['Lon'],
                mode='lines', line=dict(width=2),
                name=f"Rota {linha['Veiculo']}", showlegend=False
            ))
            
        # Adiciona o Depósito (estrela preta)
        fig.add_trace(go.Scattermap( 
            lat=[self.depot['Lat']], lon=[self.depot['Lon']],
            mode='markers', marker=dict(size=15, symbol='star', color='black'),
            name="Depósito", hovertext=["Depósito"]
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
            veiculo_id = f"V{idx_veiculo + 1}"
            tempo_atual_min = self.START_DEPOT_MIN # Tempo de início do depósito (em minutos)

            # --- Passo 1: Calcular os pontos de Serviço e Espera ---
            
            # Adiciona o ponto de Saída inicial (tempo 0)
            gantt_data.append({
                'Veiculo': veiculo_id,
                'Tarefa': f"Saída (Depósito)",
                'Start': tempo_atual_min,
                'Finish': tempo_atual_min,
                'Duration': 0,
                'Tipo': 'Saída',
                'Color': '#000000'
            })
            
            # Loop principal de cálculo
            for i in range(len(rota) - 1): # Exclui o retorno ao depósito (já tratado no avaliador)
                origem_idx = rota[i]
                destino_idx = rota[i+1]
                
                distancia_km = self.dist_matrix[origem_idx, destino_idx]
                tempo_viagem_min = distancia_km / self.VELOCIDADE_MEDIA_KM_MIN
                tempo_chegada = tempo_atual_min + tempo_viagem_min
                
                # Log de Viagem
                gantt_data.append({
                    'Veiculo': veiculo_id,
                    'Tarefa': f"Viagem de {self.df_clientes.iloc[origem_idx]['ID']} para {self.df_clientes.iloc[destino_idx]['ID']}",
                    'Start': tempo_atual_min,
                    'Finish': tempo_chegada,
                    'Duration': tempo_viagem_min,
                    'Tipo': 'Viagem',
                    'Color': '#FFD700' # Amarelo (Viagem)
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
                              'Veiculo': veiculo_id,
                              'Tarefa': f"Espera em {cliente['ID']}",
                              'Start': tempo_chegada,
                              'Finish': tempo_inicio_servico,
                              'Duration': tempo_espera,
                              'Tipo': 'Espera',
                              'Color': '#ADD8E6' # Azul Claro (Espera)
                         })

                    tempo_fim_servico = tempo_inicio_servico + self.TEMPO_SERVICO_MIN

                    # Log do Serviço
                    gantt_data.append({
                        'Veiculo': veiculo_id,
                        'Tarefa': f"Serviço em {cliente['ID']} ({int(cliente['Demanda'])} un.)",
                        'Start': tempo_inicio_servico,
                        'Finish': tempo_fim_servico,
                        'Duration': self.TEMPO_SERVICO_MIN,
                        'Tipo': 'Serviço',
                        'Color': '#20B2AA' # Verde Azulado (Serviço)
                    })
                    
                    tempo_atual_min = tempo_fim_servico # Atualiza para o tempo de saída do serviço
                else:
                    # Depósito de Retorno (Pode ser considerado um 'Serviço' de encerramento)
                    gantt_data.append({
                        'Veiculo': veiculo_id,
                        'Tarefa': f"Retorno ao Depósito",
                        'Start': tempo_chegada,
                        'Finish': tempo_chegada,
                        'Duration': 0,
                        'Tipo': 'Retorno',
                        'Color': '#808080' # Cinza (Viagem Final)
                    })
                    tempo_atual_min = tempo_chegada # Tempo de chegada é o tempo final para o veículo

        # --- Passo 2: Geração do Plotly ---
        final_gantt_df = pd.DataFrame(gantt_data)
        
        # Garante que o DataFrame não está vazio
        if final_gantt_df.empty:
            print("\nAVISO: Não foi possível calcular o cronograma Gantt. Arquivo HTML do Gantt não gerado.")
            return

        # Prepara o Plotly
        final_gantt_df['Start_h'] = final_gantt_df['Start'] / 60
        final_gantt_df['Finish_h'] = final_gantt_df['Finish'] / 60
        
        # Mapeamento de Cores para o Gráfico
        color_map = {
            'Viagem': '#FFD700', 
            'Serviço': '#20B2AA', 
            'Espera': '#ADD8E6', 
            'Retorno': '#808080', 
            'Saída': '#000000' # Preto para Saída (Pode ser ignorado no plot, mas ajuda na legenda)
        }

        # Plota apenas os eventos com duração
        fig = px.timeline(final_gantt_df[final_gantt_df['Duration'] > 0.001], 
                          x_start="Start_h", x_end="Finish_h", y="Veiculo", 
                          color="Tipo", 
                          color_discrete_map=color_map,
                          title="Cronograma de Atendimento (Gráfico de Gantt)",
                          hover_data={'Duration': ':.1f', 'Start_h': ':.2f', 'Finish_h': ':.2f', 'Tarefa': True})
        
        fig.update_yaxes(categoryorder="category descending")
        fig.update_layout(xaxis_title="Tempo (Horas)")
        
        # Salva o gráfico
        fig.write_html("gantt_schedule.html")
        
        return "gantt_schedule.html"

    # ----------------------------------------------------------
    # EXECUÇÃO COMPLETA
    # ----------------------------------------------------------
    def run_optimization(self, max_iterations=200, tabu_list_size=10):
        print("1. Criando rotas iniciais (Clark & Wright Savings)...")
        rotas_iniciais_cw = self.criar_rotas_cw()
        custo_inicial_cw = self.calculate_total_cost(rotas_iniciais_cw)
        print(f"   Custo após Savings: R$ {custo_inicial_cw:.2f}")

        if custo_inicial_cw == 999999:
            print("AVISO: Solução inicial inválida. Não é possível otimizar.")
            self.gerar_visualizacoes(rotas_iniciais_cw)
            return rotas_iniciais_cw # Retorna a rota para o Flask lidar com o erro

        print("2. Otimizando rotas iniciais com 2-opt intra-rota...")
        rotas_otimizadas_2opt = self.two_opt_optimize(rotas_iniciais_cw)
        custo_2opt = self.calculate_total_cost(rotas_otimizadas_2opt)
        print(f"   Custo após 2-opt: R$ {custo_2opt:.2f}")

        print(f"3. Executando Busca Tabu (Max Iter: {max_iterations})...") 
        rotas_finais = self.tabu_search_vrptw(rotas_otimizadas_2opt, 
                                             max_iterations=max_iterations, 
                                             tabu_list_size=tabu_list_size)

        print("4. Gerando visualizações (mapa e Gantt)...")
        self.gerar_visualizacoes(rotas_finais)
        self.gerar_gantt(rotas_finais)

        return rotas_finais


# ==========================================================
# EXECUÇÃO PRINCIPAL (Para teste isolado, se rodar OptiRota.py diretamente)
# ==========================================================
if __name__ == '__main__':
    # Tenta importar as funções do db_utils
    if 'garantir_tabela_clientes' in globals() or 'garantir_tabela_clientes' in locals():
        # 1. Prepara o Banco de Dados
        garantir_tabela_clientes()
        inserir_clientes()
        df_clientes = carregar_clientes_do_banco()

        # 2. Parâmetros do problema
        CAPACIDADE_VEICULO = 20
        VELOCIDADE_MEDIA_KM_MIN = 1.0 # 60 km/h
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

        # 3. Executa a otimização
        rotas_finais = solver.run_optimization(MAX_ITER, TABU_SIZE)
        custo_final = solver.calculate_total_cost(rotas_finais)
        print(f"\nCUSTO TOTAL FINAL DA OTIMIZAÇÃO: R$ {custo_final:.2f}")
        print("\nArquivos de visualização (rotas_otimizadas.html e gantt_schedule.html) gerados na pasta de execução.")
    else:
        print("\nO bloco de execução principal foi ignorado. Por favor, crie o arquivo 'db_utils.py' e defina o DataFrame 'df_clientes'.")