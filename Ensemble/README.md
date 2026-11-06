# Ensemble Learning: Bagging e Boosting

## Inteligência Artificial — Machine Learning com Scikit-Learn

Este projeto tem como objetivo explorar e implementar técnicas de *Ensemble Learning*, especificamente **Bagging** e **Boosting**, utilizando a biblioteca Scikit-Learn em Python. A atividade utiliza o conjunto de dados `countries`, disponibilizado pelo pacote Vega Datasets, para desenvolver e comparar modelos de regressão voltados à estimativa da expectativa de vida.

## Integrantes

| Integrante | Nome | RA |
|---|---|---|
| 1 | Daniel Teles de Oliveira | RA00334427 |
| 2 | João Victor Torres Soares | RA00330701 |

## Tecnologias e bibliotecas

- **Python:** linguagem utilizada no desenvolvimento.
- **Pandas:** manipulação e análise dos dados.
- **NumPy:** operações numéricas.
- **Matplotlib e Seaborn:** visualização de dados.
- **Vega Datasets:** disponibilização do conjunto de dados.
- **Scikit-Learn:** implementação, treinamento e avaliação dos modelos.
- **Jupyter Notebook:** ambiente de desenvolvimento e execução.

## Conjunto de dados

Foi utilizado o conjunto de dados `countries`, pertencente ao pacote Vega Datasets, que reúne informações demográficas de diferentes países ao longo dos anos.

Entre as variáveis disponíveis, destacam-se:

| Variável | Descrição |
|---|---|
| `country` | País ao qual o registro pertence. |
| `year` | Ano de observação. |
| `life_expect` | Expectativa de vida. |
| `fertility` | Taxa de fertilidade. |
| `n_fertility` | Indicador relacionado à taxa de fertilidade. |
| `n_life_expect` | Indicador relacionado à expectativa de vida. |
| `p_fertility` | Valor anterior da taxa de fertilidade. |
| `p_life_expect` | Valor anterior da expectativa de vida. |

## Etapas de desenvolvimento

O projeto foi organizado nas seguintes etapas:

### 1. Análise exploratória dos dados (EDA)

Inicialmente, foi realizada uma análise da estrutura e das características do conjunto de dados, contemplando:

- Verificação da quantidade de registros e atributos.
- Análise estatística descritiva.
- Identificação de valores ausentes.
- Visualização da evolução da expectativa de vida ao longo dos anos.
- Visualização da evolução da taxa de fertilidade.
- Comparação dos países com maiores valores registrados de expectativa de vida e fertilidade.
- Investigação da relação entre fertilidade e expectativa de vida.

### 2. Verificação de outliers

Foi utilizado o método do Intervalo Interquartil (IQR) para identificar possíveis valores atípicos nas variáveis numéricas.

Os valores identificados, principalmente aqueles relacionados às baixas expectativas de vida, foram preservados por representarem fenômenos demográficos e históricos relevantes, e não necessariamente inconsistências nos dados.

### 3. Análise de correlação

Foi construída uma matriz de correlação para investigar as relações entre as variáveis do conjunto de dados, utilizando um mapa de calor (*heatmap*) para facilitar a interpretação.

### 4. Preparação dos dados

Para o desenvolvimento dos modelos de regressão, foram realizadas as seguintes operações:

- Seleção das variáveis de interesse.
- Remoção de registros com valores ausentes.
- Codificação numérica da variável categórica `country`.
- Definição dos atributos preditores e da variável-alvo.

**Variáveis preditoras:**
- `p_fertility`
- `year`
- `country`

**Variável-alvo:**
- `life_expect`

A variável `life_expect` representa a expectativa de vida que os modelos deverão estimar.

### 5. Divisão dos dados

O conjunto de dados foi dividido em duas partes:

- **80%:** dados destinados ao treinamento.
- **20%:** dados destinados ao teste.

Foi utilizado `random_state=42` para garantir a reprodutibilidade da divisão.

## Modelos de Ensemble Learning

### Bagging

Foi implementado o algoritmo `BaggingRegressor`, utilizando árvores de decisão (`DecisionTreeRegressor`) como estimadores individuais.

A técnica consiste em treinar múltiplos modelos de forma independente, utilizando diferentes amostras dos dados, e combinar suas previsões por meio de agregação.

**Principais parâmetros utilizados:**

| Parâmetro | Valor |
|---|---|
| `n_estimators` | 200 |
| `max_samples` | 0.8 |
| `max_features` | 1.0 |
| `max_depth` (árvore) | 4 |
| `criterion` | squared_error |
| `random_state` | 42 |

### Boosting

Foi implementado o algoritmo `GradientBoostingRegressor`, que constrói árvores de decisão sequencialmente. Cada novo estimador busca corrigir os erros residuais produzidos pelas etapas anteriores, aprimorando progressivamente as previsões.

**Principais parâmetros utilizados:**

| Parâmetro | Valor |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.05 |
| `max_depth` | 3 |
| `subsample` | 0.8 |
| `max_features` | 0.8 |
| `loss` | squared_error |
| `random_state` | 42 |

## Avaliação dos modelos

Para avaliar o desempenho dos modelos, foram utilizadas quatro métricas de regressão:

- **MSE (Mean Squared Error):** calcula a média dos erros de previsão elevados ao quadrado, penalizando mais intensamente erros maiores.
- **RMSE (Root Mean Squared Error):** representa a raiz quadrada do MSE, expressando o erro na mesma unidade da variável-alvo.
- **MAE (Mean Absolute Error):** calcula a média dos erros absolutos das previsões.
- **R² (Coeficiente de Determinação):** indica a proporção da variabilidade da variável-alvo explicada pelo modelo.

### Resultados obtidos

| Métrica | Bagging | Gradient Boosting |
|---|---:|---:|
| MSE | 15,3010 | 8,0364 |
| RMSE | 3,9116 | 2,8349 |
| MAE | 2,8856 | 2,0005 |
| R² | 0,8338 | 0,9127 |

### Análise comparativa

Os resultados demonstraram que o **Gradient Boosting apresentou desempenho superior ao Bagging em todas as métricas avaliadas**.

Em comparação ao Bagging, o Gradient Boosting apresentou:

- Redução de aproximadamente 47,5% no MSE.
- Redução de aproximadamente 27,5% no RMSE.
- Redução de aproximadamente 30,7% no MAE.
- Aumento de aproximadamente 7,97 pontos percentuais no R².

Esses resultados indicam que o Gradient Boosting produziu previsões mais próximas dos valores reais, apresentando menores erros e maior capacidade de explicar a variabilidade da expectativa de vida.

O Bagging também demonstrou resultados relevantes, destacando a capacidade da combinação de múltiplas árvores de decisão em produzir estimativas consistentes.

Portanto, considerando os resultados obtidos nesta configuração experimental, o Gradient Boosting apresentou a melhor capacidade preditiva para a estimativa da expectativa de vida.

## Como executar o projeto

### 1. Pré-requisitos

É necessário possuir Python instalado e um ambiente compatível com Jupyter Notebook.

### 2. Instalação das dependências

Execute o comando:

```bash
pip install pandas numpy matplotlib seaborn missingno scikit-learn vega-datasets jupyter
```

### 3. Execução

Clone ou baixe este repositório e abra o arquivo `Ensemble.ipynb` utilizando Jupyter Notebook ou JupyterLab.

Execute as células sequencialmente para reproduzir as análises exploratórias, o processamento dos dados, o treinamento dos modelos e a avaliação dos resultados.

## Considerações finais

A atividade permitiu explorar duas importantes técnicas de *Ensemble Learning*: Bagging e Boosting, aplicadas a um problema de regressão utilizando dados demográficos.

A comparação evidenciou diferenças entre as abordagens, demonstrando que o Gradient Boosting obteve melhores resultados nesta aplicação, especialmente na redução dos erros de previsão e no aumento do coeficiente de determinação.

O desenvolvimento também proporcionou a aplicação prática de conceitos de análise exploratória, preparação de dados, aprendizado supervisionado e avaliação de modelos de Machine Learning.