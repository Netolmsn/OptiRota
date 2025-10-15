from flask import Flask, jsonify, request, send_from_directory
import plotly.graph_objects as go
import pandas as pd
from db_utils import carregar_clientes_do_banco, salvar_cliente_no_banco

app = Flask(__name__)

# Servir interface principal (arquivo interface.html deve estar na mesma pasta)
@app.route('/')
def index():
    return send_from_directory('.', 'interface.html')

# permitir servir arquivos estáticos simples (rotas, html, js)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# API clientes (retorna JSON simples)
@app.route('/api/clientes', methods=['GET'])
def api_clientes():
    df = carregar_clientes_do_banco()
    # garantir colunas e tipos
    if df is None or df.empty:
        return jsonify([])  # lista vazia
    return df.to_json(orient='records', force_ascii=False)

# adicionar cliente
@app.route('/api/adicionar_cliente', methods=['POST'])
def adicionar_cliente():
    data = request.get_json()
    if not data:
        return jsonify({'erro': 'Nenhum dado recebido'}), 400
    try:
        novo_cliente = {
            'ID': data['ID'],
            'Lat': float(data['Lat']),
            'Lon': float(data['Lon']),
            'Demanda': int(data['Demanda']),
            'T_Inicio_h': float(data['T_Inicio_h']),
            'T_Fim_h': float(data['T_Fim_h'])
        }
        salvar_cliente_no_banco(novo_cliente)
        return jsonify({'mensagem': 'Cliente adicionado com sucesso!'})
    except Exception as e:
        return jsonify({'erro': str(e)}), 500

# GRÁFICO DEMANDA -> retorna figura em formato dicionário (serializável)
@app.route('/api/grafico_demanda')
def grafico_demanda():
    df = carregar_clientes_do_banco()
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem dados disponíveis")
        return jsonify(fig.to_dict())

    # agrega demanda por zona (exemplo por longitude)
    zonas = {"Oeste": 0, "Central": 0, "Leste": 0}
    for _, c in df.iterrows():
        if c.get("ID") == "C":
            continue
        lon = float(c.get("Lon", 0))
        demanda = int(c.get("Demanda", 0))
        if lon < -46.635:
            zonas["Oeste"] += demanda
        elif lon < -46.625:
            zonas["Central"] += demanda
        else:
            zonas["Leste"] += demanda

    fig = go.Figure(go.Bar(x=list(zonas.keys()), y=list(zonas.values()), marker=dict(color="#06b6d4")))
    fig.update_layout(
        title="Demanda por Zona",
        margin=dict(t=30, l=40, r=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfe7f7")
    )
    return jsonify(fig.to_dict())

# GRÁFICO GANTT -> retorna figura em formato dicionário
@app.route('/api/grafico_gantt')
def grafico_gantt():
    df = carregar_clientes_do_banco()
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem dados disponíveis")
        return jsonify(fig.to_dict())

    # filtrar depósito
    df = df[df["ID"] != "C"]

    # Criar barras horizontais (cada cliente uma linha)
    barras = []
    y_order = []  # para manter ordem
    for _, c in df.iterrows():
        try:
            start = float(c["T_Inicio_h"])
            end = float(c["T_Fim_h"])
            dur = end - start
            if dur <= 0:
                # evita barras inválidas
                continue
            barras.append(go.Bar(
                x=[dur],
                y=[c["ID"]],
                base=start,
                orientation="h",
                marker=dict(color="#3b82f6"),
                name=c["ID"]
            ))
            y_order.append(c["ID"])
        except Exception:
            continue

    if not barras:
        fig = go.Figure()
        fig.update_layout(title="Sem janelas válidas")
        return jsonify(fig.to_dict())

    fig = go.Figure(data=barras)
    fig.update_layout(
        title="Janela de Atendimento (Gantt)",
        xaxis=dict(title="Hora do dia (h)", range=[6, 20]),
        yaxis=dict(autorange='reversed'),  # mostra o primeiro no topo
        margin=dict(t=30, l=120, r=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfe7f7"),
        showlegend=False
    )
    return jsonify(fig.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
