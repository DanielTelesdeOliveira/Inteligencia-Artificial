# Random Forest — Análise de dados demográficos

Atividade da disciplina de **Inteligência Artificial**, do curso de **Ciência da Computação da PUC-SP**, que explora modelos de *Random Forest* para classificação e regressão utilizando Python, Scikit-learn e o conjunto de dados `countries`, disponibilizado pelo pacote `vega\_datasets`.

## Integrantes

|Nome|GitHub|
|-|-|
|Daniel Teles de Oliveira|[DanielTelesdeOlivera](https://github.com/DanielTelesdeOlivera)|
|João Victor Torres Soares|[JTSoares](https://github.com/JTSoares)|

## Objetivo

Explorar o processo de análise e preparação de dados e aplicar os algoritmos `RandomForestClassifier` e `RandomForestRegressor`. O notebook investiga indicadores demográficos de diferentes países, com destaque para a expectativa de vida e a taxa de fertilidade, e avalia os resultados dos modelos por meio de visualizações e métricas.

## Conjunto de dados

O dataset `countries`, carregado com `data.countries()`, reúne observações por país e ano. Entre as variáveis utilizadas no notebook estão:

* `country`: país;
* `year`: ano da observação;
* `life\_expect`: expectativa de vida;
* `fertility`: taxa de fertilidade;
* `n\_life\_expect` e `n\_fertility`: indicadores referentes à próxima observação temporal, conforme a estrutura do dataset;
* `p\_life\_expect` e `p\_fertility`: indicadores referentes à observação temporal anterior, conforme a estrutura do dataset.

## Etapas desenvolvidas

1. **Exploração inicial:** inspeção da estrutura, quantidade de registros, estatísticas descritivas e amostras do dataset.
2. **Visualização dos dados:** gráficos de linha para tendências temporais, tabelas e gráficos de barras para comparar países e gráfico de dispersão para observar a relação entre fertilidade e expectativa de vida.
3. **Análise de valores ausentes:** identificação da quantidade e distribuição de dados ausentes com tabelas e visualizações.
4. **Verificação de outliers:** investigação de valores extremos por meio do intervalo interquartil (IQR), com discussão sobre a preservação de observações demográficas que podem representar situações reais.
5. **Correlação entre variáveis:** codificação de colunas categóricas para explorar associações por meio de uma matriz de correlação.
6. **Preparação para modelagem:** definição das variáveis de entrada, criação da variável-alvo de classificação, tratamento de valores ausentes e separação dos dados em conjuntos de treino e teste.
7. **Classificação:** treinamento de um `RandomForestClassifier` para prever se a expectativa de vida aumentará em relação ao valor atual. A avaliação inclui acurácia, relatório de classificação e matriz de confusão.
8. **Regressão:** treinamento de um `RandomForestRegressor` para estimar a expectativa de vida histórica (`p\_life\_expect`), com avaliação por métricas de regressão.

## Tecnologias e bibliotecas

* Python
* pandas e NumPy — manipulação dos dados;
* Matplotlib e Seaborn — visualizações;
* missingno — análise visual de valores ausentes;
* Vega Datasets — obtenção do dataset;
* Scikit-learn — divisão dos dados, modelos e métricas de avaliação.

## Como executar

### 1\. Pré-requisitos

Tenha o Python instalado e, preferencialmente, utilize um ambiente virtual.

### 2\. Instale as dependências

```bash
pip install pandas numpy matplotlib seaborn missingno vega\_datasets scikit-learn jupyter
```

### 3\. Abra o notebook

Na pasta que contém `Random\_Forest.ipynb`, execute:

```bash
jupyter notebook Random\_Forest.ipynb
```

Execute as células em ordem para carregar os dados, realizar as análises e treinar os modelos. Também é possível abrir o arquivo no JupyterLab ou no Visual Studio Code com suporte a notebooks.

## Resultados apresentados

O notebook registra os seguintes resultados para os modelos avaliados:

* **Classificação:** acurácia reportada de **95,24%**, acompanhada de relatório por classe e matriz de confusão.
* **Regressão:** (R^2 = 0{,}9793) e erro absoluto médio (*MAE*) de aproximadamente **0,7924 ano**, conforme a avaliação registrada no notebook.

Esses valores correspondem aos resultados documentados na atividade e podem variar se os dados, as dependências ou as etapas de processamento forem alterados.

## Limitação metodológica

Antes de interpretar os resultados como evidência de capacidade preditiva generalizável, é importante ressaltar a definição das variáveis de entrada. Na classificação, a variável-alvo `expectativa\_vai\_aumentar` é construída a partir de `n\_life\_expect`, que também aparece entre as variáveis de entrada. Essa relação pode causar **vazamento de dados (*data leakage*)** e inflar as métricas. 

## Estrutura do projeto

```text
.
├── Random\_Forest.ipynb
└── README.md
```

\---

Atividade acadêmica desenvolvida para explorar análise exploratória de dados e modelos de Random Forest com Scikit-learn.

