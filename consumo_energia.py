import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sqlite3
import pandas as pd

# Função para criar um banco de dados SQLite e uma tabela para armazenar dados de consumo
def criar_banco_dados():
    try:
        conn = sqlite3.connect('consumo_energia.db')  # Conexão com o banco de dados
        cursor = conn.cursor()

        # Criar tabela
        cursor.execute('''CREATE TABLE IF NOT EXISTS consumo (
            id INTEGER PRIMARY KEY,
            tipo TEXT,
            valor REAL
        )''')

        # Inserir dados fictícios de consumo apenas se a tabela estiver vazia
        cursor.execute("SELECT COUNT(*) FROM consumo")
        if cursor.fetchone()[0] == 0:
            # Dados de uma família residencial
            consumo_familia_residencial = np.random.normal(loc=1.67, scale=0.3, size=12)  # Média de 1.67 kWh com variação
            consumo_familia_residencial = np.clip(consumo_familia_residencial, 0, None)  # Garantir que os valores sejam não negativos

            for valor in consumo_familia_residencial:
                cursor.execute('INSERT INTO consumo (tipo, valor) VALUES (?, ?)', ('residencial', valor))

            # Dados de consumo comercial (exemplo fictício)
            dados_comerciais = [
                ('comercial', 10), ('comercial', 12), ('comercial', 15),
                ('comercial', 20), ('comercial', 22), ('comercial', 18),
                ('comercial', 25), ('comercial', 20), ('comercial', 15),
                ('comercial', 14), ('comercial', 17), ('comercial', 19),
                ('comercial', 16), ('comercial', 18), ('comercial', 15),
                ('comercial', 13), ('comercial', 20), ('comercial', 21),
                ('comercial', 22), ('comercial', 20)
            ]

            cursor.executemany('INSERT INTO consumo (tipo, valor) VALUES (?, ?)', dados_comerciais)
            conn.commit()
        else:
            print("Dados já inseridos no banco de dados.")

    except Exception as e:
        print(f"Erro ao criar o banco de dados: {e}")
    finally:
        conn.close()

# Função para carregar os dados do banco de dados
def carregar_dados_banco():
    try:
        conn = sqlite3.connect('consumo_energia.db')
        query = "SELECT valor FROM consumo WHERE tipo='residencial'"
        consumo_residencial = pd.read_sql(query, conn).values.flatten()

        query = "SELECT valor FROM consumo WHERE tipo='comercial'"
        consumo_comercial = pd.read_sql(query, conn).values.flatten()

        return consumo_residencial, consumo_comercial

    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return np.array([]), np.array([])  # Retorna arrays vazios em caso de erro

    finally:
        conn.close()

# Função para treinar o modelo KNN
def treinar_modelo(X_train, y_train, n_neighbors=3):
    modelo_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    modelo_knn.fit(X_train, y_train)
    return modelo_knn

# Função para classificar novo consumo
def classificar_consumo(modelo_knn, novo_consumo):
    novo_consumo_array = np.array(novo_consumo).reshape(1, -1)  # Reshape para um único valor

    if novo_consumo_array.shape[1] != 1:
        print("Insira um valor de consumo.")
        return

    predicao = modelo_knn.predict(novo_consumo_array)
    resultado = 'Comercial' if predicao[0] == 2 else 'Residencial'
    print(f"Predição para o consumo de energia {novo_consumo[0]}: {resultado}")
    return resultado

# Função para visualizar dados de consumo
def visualizar_dados(consumo_residencial, consumo_comercial, novos_consumos):
    plt.figure(figsize=(12, 6))
    plt.hist(consumo_residencial, bins=10, alpha=0.5, color='blue', label='Residencial')
    plt.hist(consumo_comercial, bins=10, alpha=0.5, color='orange', label='Comercial')
    plt.title('Distribuição do Consumo de Energia')
    plt.xlabel('Valores de Consumo (kWh)')
    plt.ylabel('Frequência')

    if novos_consumos:
        plt.axvline(x=np.mean(novos_consumos), color='red', linestyle='--', label='Média dos Novos Consumidores')

    plt.text(0.5, 0.9, 'Dados coletados em um intervalo de 2 horas',
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12, color='black')
    plt.legend()
    plt.show()

# Criar o banco de dados e tabela
criar_banco_dados()

# Carregar os dados do banco de dados
consumo_residencial, consumo_comercial = carregar_dados_banco()

# Combine os dados comerciais e residenciais
X = np.concatenate([consumo_residencial, consumo_comercial]).reshape(-1, 1)

# Rótulos: 1 para residencial e 2 para comercial
y = np.array([1] * len(consumo_residencial) + [2] * len(consumo_comercial))

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo KNN
modelo_knn = treinar_modelo(X_train, y_train)

# Avaliar a precisão do modelo
y_pred = modelo_knn.predict(X_test)
precisao = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo KNN: {precisao * 100:.2f}%')

# Testar diferentes valores de k
for k in range(1, 11):
    modelo_knn = treinar_modelo(X_train, y_train, n_neighbors=k)
    y_pred = modelo_knn.predict(X_test)
    precisao = accuracy_score(y_test, y_pred)
    print(f'Acurácia com k={k}: {precisao * 100:.2f}%')

# Listas para armazenar os novos consumos e resultados
novos_consumos = []
resultados = []

# Coletar dados do usuário
while True:
    try:
        novo_consumo = float(input("Insira um valor de consumo de energia (ou digite -1 para sair): "))
        if novo_consumo == -1:
            break  # Sair do loop se o usuário digitar -1
        novos_consumos.append(novo_consumo)
        resultado = classificar_consumo(modelo_knn, [novo_consumo])
        resultados.append((novo_consumo, resultado))
    except ValueError:
        print("Valor inválido. Tente novamente.")

# Exibir os resultados de cada valor digitado
print("\nResultados das classificações:")
for consumo, resultado in resultados:
    print(f"Consumo: {consumo} kWh - Classificação: {resultado}")

# Visualização final dos dados de consumo
visualizar_dados(consumo_residencial, consumo_comercial, novos_consumos)
