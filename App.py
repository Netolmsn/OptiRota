from flask import Flask, request, jsonify
import OptiRota 

app = Flask(__name__)

@app.route('/optimize-routes', methods=['POST'])
def optimize_routes():
    data = request.get_json()
    
    try:
        capacity = float(data.get('capacity'))
        speed = float(data.get('speed'))
        cost_km = float(data.get('cost_km'))
        
    except (TypeError, ValueError):
        return jsonify({
            "error": "Dados inválidos",
            "message": "Certifique-se de que 'capacity', 'speed' e 'cost_km' são números válidos."
        }), 400

    try:
        final_cost = OptiRota.run_optimization(capacity, speed, cost_km)
        
        return jsonify({
            "success": True,
            "message": "Otimização concluída e arquivos HTML gerados.",
            "final_cost": f"R$ {final_cost:.2f}",
            "details": "Verifique os arquivos 'rotas_otimizadas.html' e 'gantt_schedule_aprimorado.html' no diretório do servidor."
        }), 200

    except Exception as e:
        print(f"Erro durante a otimização: {e}")
        return jsonify({
            "success": False,
            "message": f"Erro interno no servidor durante a otimização: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
