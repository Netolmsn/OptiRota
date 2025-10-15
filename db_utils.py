import sqlite3
import pandas as pd

DB_PATH = 'clientes.db'

# ==========================================================
# Garantir e inicializar a tabela
# ==========================================================
def garantir_tabela_clientes(db_path=DB_PATH):
    """Garante que a tabela 'clientes' existe e insere o depósito padrão."""
    conn = sqlite3.connect(db_path)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS clientes (
        ID TEXT PRIMARY KEY,
        Lat REAL,
        Lon REAL,
        Demanda INTEGER,
        T_Inicio_h REAL,
        T_Fim_h REAL
    )
    ''')
    # Insere o depósito padrão (ID 'C') se não existir
    conn.execute("""
        INSERT OR IGNORE INTO clientes (ID, Lat, Lon, Demanda, T_Inicio_h, T_Fim_h)
        VALUES ('C', -23.5505, -46.6333, 0, 8.0, 17.0)
    """)
    conn.commit()
    conn.close()


# ==========================================================
# Inserir dados iniciais (C1 a C10)
# ==========================================================
def inserir_clientes(db_path=DB_PATH):
    """Insere dados de exemplo (clientes de C1 a C10)."""
    conn = sqlite3.connect(db_path)
    clientes = [
        ('C1', -23.545, -46.640, 5, 9.0, 12.0),
        ('C2', -23.560, -46.625, 10, 10.0, 11.0),
        ('C3', -23.570, -46.638, 3, 9.0, 10.0),
        ('C4', -23.535, -46.650, 7, 11.0, 13.0),
        ('C5', -23.555, -46.615, 2, 10.0, 12.0),
        ('C6', -23.540, -46.630, 8, 8.0, 9.0),
        ('C7', -23.548, -46.620, 6, 9.0, 12.0),
        ('C8', -23.530, -46.635, 4, 10.0, 13.0),
        ('C9', -23.552, -46.637, 9, 8.0, 11.0),
        ('C10', -23.538, -46.628, 5, 9.0, 12.0)
    ]

    for cliente in clientes:
        conn.execute("""
            INSERT OR REPLACE INTO clientes (ID, Lat, Lon, Demanda, T_Inicio_h, T_Fim_h)
            VALUES (?, ?, ?, ?, ?, ?)
        """, cliente)
    conn.commit()
    conn.close()


# ==========================================================
# Carregar todos os clientes
# ==========================================================
def carregar_clientes_do_banco(db_path=DB_PATH):
    """Carrega todos os clientes do banco de dados."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM clientes ORDER BY ID", conn)
    conn.close()
    return df


# ==========================================================
# Salvar novo cliente (usado pela rota /api/adicionar_cliente)
# ==========================================================
def salvar_cliente_no_banco(cliente, db_path=DB_PATH):
    """Insere um novo cliente (exceto depósito 'C') no banco."""
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS clientes (
            ID TEXT PRIMARY KEY,
            Lat REAL,
            Lon REAL,
            Demanda INTEGER,
            T_Inicio_h REAL,
            T_Fim_h REAL
        )
    ''')
    if cliente['ID'] == 'C':
        raise ValueError("ID 'C' é reservado para o depósito principal.")
    conn.execute("""
        INSERT OR REPLACE INTO clientes (ID, Lat, Lon, Demanda, T_Inicio_h, T_Fim_h)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (cliente['ID'], cliente['Lat'], cliente['Lon'], cliente['Demanda'], cliente['T_Inicio_h'], cliente['T_Fim_h']))
    conn.commit()
    conn.close()


# ==========================================================
# Execução isolada (para testes)
# ==========================================================
if __name__ == '__main__':
    garantir_tabela_clientes()
    inserir_clientes()
    print("✅ Banco de dados 'clientes.db' configurado com sucesso.")
