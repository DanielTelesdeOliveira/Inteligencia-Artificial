# Customer Segmentation - K-Means Clustering

**Disciplina:** Inteligência Artificial - Ciência da Computação (PUC-SP)  
**Equipe:** João Victor Torres Soares e Daniel Teles de Oliveira

## Objetivo do Projeto
Este mini-projeto de laboratório tem como objetivo explorar conceitos de **Aprendizado Não Supervisionado** aplicando o algoritmo de clusterização **K-Means**. A análise foi realizada sobre um dataset de clientes de um shopping, visando identificar diferentes perfis de consumidores com base em sua renda anual e pontuação de gastos (Spending Score).

## Dataset Utilizado
Os dados foram extraídos da plataforma Kaggle:
* **Dataset:** [Mall Customer Segmentation Data](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
* **Importação:** Para garantir a reprodutibilidade deste repositório, o dataset é importado dinamicamente durante a execução do notebook utilizando a biblioteca `kagglehub`, dispensando o download manual.

## Tecnologias e Bibliotecas
* Python 3
* Pandas & NumPy (Manipulação de dados)
* Matplotlib & Seaborn (Análise Exploratória de Dados - EDA e Visualização)
* Scikit-Learn (Machine Learning - KMeans)
* Google Colab (Ambiente de desenvolvimento)

## Metodologia
1. **EDA (Análise Exploratória de Dados):** Inspeção da estrutura do dataset, verificação de dados nulos e visualização preliminar da distribuição (Renda x Gastos).
2. **Método do Cotovelo (Elbow Method):** Utilizado para calcular a Soma dos Quadrados Intra-Cluster (WCSS) e definir o número ideal de clusters, que convergiu para **K=5**.
3. **Treinamento e Clusterização:** Aplicação do algoritmo K-Means para segmentar a base em 5 grupos distintos.

## Principais Insights
O modelo conseguiu segmentar os clientes em 5 perfis comportamentais claros, gerando insumos estratégicos para campanhas de marketing:
* **Cluster 0:** Renda Média / Gasto Médio (Público padrão)
* **Cluster 1:** Alta Renda / Alto Gasto (Clientes premium/alvo)
* **Cluster 2:** Baixa Renda / Alto Gasto (Clientes impulsivos)
* **Cluster 3:** Alta Renda / Baixo Gasto (Clientes conservadores/cuidadosos)
* **Cluster 4:** Baixa Renda / Baixo Gasto 

## Como executar
Como o notebook foi configurado para importar os dados via API, basta abrir o arquivo `notebook_eda.ipynb` no Google Colab ou Jupyter Notebook e executar todas as células consecutivamente.