from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from main import VRPTWSolver, carregar_clientes_do_banco
from db_utils import garantir_tabela_clientes, inserir_deposito_padrao
import os

app = Flask(__name__)

# Garante que o banco está pronto
garantir_tabela_clientes()
inserir_deposito_padrao()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        params = request.get_json()
        df_clientes = carregar_clientes_do_banco()
        solver = VRPTWSolver(
            df_clientes,
            capacidade_veiculo=params.get('capacity', 20),
            velocidade_media_km_min=params.get('speed', 1.0),
            tempo_servico_min=15,
            custo_km=params.get('cost_km', 0.5),
            custo_min_operacao=1.0,
            multa_atraso_por_min=5.0
        )
        custo_final = solver.run_optimization(max_iterations=200, tabu_list_size=10)
        return jsonify({'status':'success','final_cost':custo_final})
    except Exception as e:
        return jsonify({'status':'error','message':str(e)})

@app.route('/api/novo_cliente', methods=['POST'])
def novo_cliente():
    data = request.get_json()
    try:
        import sqlite3
        conn = sqlite3.connect('clientes.db')
        conn.execute('INSERT OR REPLACE INTO clientes (ID, Lat, Lon, Demanda, T_Inicio_h, T_Fim_h) VALUES (?, ?, ?, ?, ?, ?)',
                     (data['ID'], float(data['Lat']), float(data['Lon']), int(data['Demanda']), int(data['T_Inicio_h']), int(data['T_Fim_h'])))
        conn.commit()
        conn.close()
        return "✅ Cliente adicionado com sucesso!"
    except Exception as e:
        return f"❌ Erro ao adicionar cliente: {e}"

# Servir os HTMLs gerados
@app.route('/rotas_otimizadas')
def rotas_otimizadas():
    return send_file('rotas_otimizadas.html')

@app.route('/gantt_schedule')
def gantt_schedule():
    file_path = 'gantt_schedule_aprimorado.html'
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "<h3>Aguardando geração do gráfico de Gantt...</h3>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
