# OptiRota

OptiRota é uma solução para otimização de rotas de veículos com janelas de tempo, utilizando algoritmos heurísticos e metaheurísticos (Savings, 2-opt, Busca Tabu) e visualização interativa dos resultados.

## Funcionalidade

O sistema permite cadastrar clientes e veículos, armazenar os dados em banco SQLite, rodar a otimização das rotas e gerar visualizações interativas (mapa e gráfico de Gantt) em arquivos HTML.

## Passo a Passo para Utilizar

1. **Instale as dependências**
	- Python 3.8 ou superior
	- Instale as bibliotecas necessárias:
	  ```bash
	  pip install pandas numpy plotly
	  ```

2. **Configure o banco de dados**
	- Ao rodar o sistema, as tabelas `clientes` e `veiculos` serão criadas automaticamente se não existirem.
	- Cadastre pelo menos um cliente com `ID = 'C'` (depósito) e os demais clientes via formulário ou inserção direta no banco.

3. **Cadastro de clientes e veículos**
	- Use o formulário web (Interface.html) para cadastrar novos clientes.
	- Para cadastrar veículos, utilize função dedicada ou insira diretamente na tabela `veiculos`.

4. **Executando a otimização**
	- Execute o arquivo principal:
	  ```bash
	  python OptiRota.py
	  ```
	- Se não houver clientes cadastrados, o sistema exibirá uma mensagem informando.

5. **Visualização dos resultados**
	- Após a execução, serão gerados os arquivos `rotas_otimizadas.html` e `gantt_schedule_aprimorado.html` na pasta do projeto.
	- Abra esses arquivos no navegador para visualizar o mapa das rotas e o cronograma das entregas.

## Estrutura do Projeto

- `OptiRota.py`: Lógica principal de otimização e integração com banco de dados.
- `db_utils.py`: Funções utilitárias para manipulação do banco de dados.
- `Interface.html`: Interface web para cadastro de clientes.
- `clientes_inicial.html`, `rotas_otimizadas.html`, `gantt_schedule_aprimorado.html`: Visualizações geradas.
- `clientes.db`: Banco de dados SQLite.

## Observações

- Certifique-se de que o depósito (`ID = 'C'`) está cadastrado antes de rodar a otimização.
- Os parâmetros dos veículos e clientes podem ser ajustados conforme a necessidade.
- O sistema pode ser expandido para integração com APIs ou outras interfaces.
teste