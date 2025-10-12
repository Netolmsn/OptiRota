import sqlite3
import sqlite3
import pandas as pd

DB_PATH = 'clientes.db'


def garantir_tabela_clientes(db_path=DB_PATH):
    """Garante que a tabela de clientes existe no banco."""
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS clientes (
            ID TEXT PRIMARY KEY,
            Lat REAL,
            Lon REAL,
            Demanda INTEGER,
            T_Inicio_h INTEGER,
            T_Fim_h INTEGER
        )
    ''')
    conn.commit()
    conn.close()


def garantir_tabela_veiculos(db_path=DB_PATH):
    """Garante que a tabela de veículos existe no banco."""
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS veiculos (
            ID TEXT PRIMARY KEY,
            Capacidade INTEGER,
            Tipo TEXT
        )
    ''')
    conn.commit()
    conn.close()


def carregar_clientes_do_banco(db_path=DB_PATH):
    """Carrega a tabela de clientes como DataFrame pandas."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM clientes", conn)
    conn.close()
    return df


def inserir_deposito_padrao(db_path=DB_PATH):
    """Garante que o depósito padrão ('C') está presente no banco."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT OR IGNORE INTO clientes 
        (ID, Lat, Lon, Demanda, T_Inicio_h, T_Fim_h) 
        VALUES ('C', -23.5505, -46.6333, 0, 8, 17)
    """)
    conn.commit()
    conn.close()
