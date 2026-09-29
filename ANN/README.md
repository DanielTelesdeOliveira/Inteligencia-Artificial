# Previsão de Preços de Conjuntos LEGO com Redes Neurais Artificiais

Projeto desenvolvido para a disciplina de **Inteligência Artificial /
Machine Learning** do curso de Ciência da Computação, com o objetivo de
explorar **Redes Neurais Artificiais (ANN)** utilizando Python e
Scikit-learn para prever o preço de conjuntos LEGO.

## Integrantes

- Daniel: [github.com/DanielTelesdeOliveira](https://github.com/DanielTelesdeOliveira)
- João Victor: [github.com/JTSoares](https://github.com/JTSoares)

## Sobre o projeto

O projeto utiliza uma base de dados de conjuntos LEGO lançados entre
**1970 e 2022**, contendo informações como tema, quantidade de peças,
quantidade de minifiguras, idade mínima recomendada e preço de varejo
nos Estados Unidos.

O objetivo principal é utilizar um modelo de **Multi-Layer Perceptron
Regressor (MLPRegressor)** para estimar o preço (`US_retailPrice`) dos
conjuntos e criar uma nova coluna `prices` com os valores previstos.

A base de dados utilizada está disponível no [Maven Analytics Data
Playground](https://mavenanalytics.io/data-playground/lego-sets).

## Estrutura dos dados

A base `lego_sets.csv` contém, entre outras, as seguintes variáveis:

-   `set_id`: identificador do conjunto;
-   `name`: nome do conjunto;
-   `year`: ano de lançamento;
-   `theme`: tema;
-   `subtheme`: subtema;
-   `themeGroup`: grupo do tema;
-   `category`: categoria;
-   `pieces`: quantidade de peças;
-   `minifigs`: quantidade de minifiguras;
-   `agerange_min`: idade mínima recomendada;
-   `US_retailPrice`: preço de varejo nos Estados Unidos;
-   `bricksetURL`: URL do conjunto;
-   `thumbnailURL`: URL da miniatura;
-   `imageURL`: URL da imagem.

## Tecnologias e bibliotecas

-   Python 3
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Missingno
-   Scikit-learn
    -   `LabelEncoder`
    -   `StandardScaler`
    -   `OneHotEncoder`
    -   `ColumnTransformer`
    -   `MLPRegressor`
    -   métricas de regressão

## Execução

O notebook foi desenvolvido para execução em ambiente Python/Google
Colab.

### 1. Obter os dados

O notebook realiza o download do arquivo compactado diretamente da fonte
utilizada:

``` bash
wget https://maven-datasets.s3.amazonaws.com/LEGO+Sets/LEGO+Sets.zip
```

Em seguida, o arquivo é descompactado:

``` bash
unzip *.zip
```

São obtidos os arquivos:

-   `lego_sets.csv`
-   `lego_sets_data_dictionary.csv`

### 2. Carregar a base

A base é carregada com Pandas:

``` python
lego_df = pd.read_csv("lego_sets.csv")
```

## Análise exploratória

Inicialmente são investigados os valores presentes nas variáveis e a
quantidade de dados ausentes.

Entre as colunas com valores ausentes estão:

-   `subtheme`
-   `themeGroup`
-   `pieces`
-   `minifigs`
-   `agerange_min`
-   `US_retailPrice`
-   `thumbnailURL`
-   `imageURL`

A distribuição dos valores ausentes é visualizada utilizando a
biblioteca `missingno`, por meio de:

-   matriz de valores ausentes;
-   gráfico de barras;
-   heatmap.

Também é construído um **heatmap de correlação** envolvendo variáveis
categóricas e numéricas. Para possibilitar a correlação das variáveis
categóricas, o notebook utiliza `LabelEncoder` e remove, nessa análise,
as observações que possuem valores nulos nas variáveis consideradas.

A partir dessa análise são selecionadas as variáveis utilizadas
posteriormente no modelo:

-   `pieces`;
-   `minifigs`;
-   `agerange_min`;
-   `theme`.

A variável-alvo é:

-   `US_retailPrice`.

## Tratamento dos dados

### Remoção de outliers

Os outliers são identificados nas variáveis numéricas:

-   `pieces`;
-   `minifigs`;
-   `agerange_min`;
-   `US_retailPrice`.

É utilizado o método do **Intervalo Interquartil (IQR)**, considerando
os limites:

``` text
Limite inferior = Q1 - 1,5 × IQR
Limite superior = Q3 + 1,5 × IQR
```

Na primeira etapa de tratamento, após a remoção dos outliers, o conjunto
passa a possuir **18.457 linhas**.

### Imputação de valores ausentes

Os valores ausentes restantes são preenchidos utilizando a **moda** de
cada coluna.

Após a imputação, é realizada uma verificação para confirmar a ausência
de valores nulos.

## Modelo de previsão

O modelo utilizado é o `MLPRegressor` do Scikit-learn.

As variáveis numéricas utilizadas como entrada são:

``` python
["pieces", "minifigs", "agerange_min"]
```

A variável categórica utilizada é:

``` python
["theme"]
```

O pré-processamento utiliza:

-   `StandardScaler` para as variáveis numéricas;
-   `OneHotEncoder(handle_unknown="ignore")` para `theme`;
-   `ColumnTransformer` para combinar as transformações.

A primeira preparação resultou em **157 atributos processados**.

### Configuração do MLP

``` python
MLPRegressor(
    hidden_layer_sizes=(100, 100),
    activation="relu",
    solver="adam",
    max_iter=250,
    random_state=42
)
```

A arquitetura possui duas camadas ocultas, cada uma com 100 neurônios.

## Primeira avaliação

Inicialmente, o modelo foi treinado e utilizado para realizar previsões
sobre o próprio conjunto utilizado no treinamento.

As previsões foram armazenadas na nova coluna:

``` text
prices
```

A comparação entre `US_retailPrice` e `prices` foi realizada por meio de
gráficos de dispersão, incluindo uma linha de referência `y = x`.

Para os conjuntos que já possuíam preço original, foram encontrados
**6.982 conjuntos** para comparação direta entre o preço original e o
preço previsto.

### Métricas

| Métrica | Resultado |
|---|---:|
| R² | 0,886 |
| MAE | 9,455 |
| MSE | 338,331 |
| RMSE | 18,394 |


Esse resultado não deve ser interpretado como uma medida de
generalização do modelo, pois a avaliação foi realizada sobre dados
utilizados durante o treinamento. O próprio notebook identifica essa
limitação e realiza posteriormente uma nova avaliação com separação
entre treino e teste.

## Segunda avaliação: treino e teste

Para obter uma avaliação mais adequada e evitar **data leakage**, o
notebook refaz o processo seguindo esta sequência:

1.  divisão da base original em treino e teste;
2.  cálculo dos limites de outliers somente no conjunto de treino;
3.  cálculo das modas somente no conjunto de treino;
4.  aplicação desses parâmetros ao conjunto de teste;
5.  treinamento do MLP apenas com os dados de treino;
6.  avaliação das previsões no conjunto de teste.

A divisão utilizada foi de **80% para treinamento e 20% para teste**,
com `random_state=42`.


| Conjunto | Linhas |
|---|---:|
| Treino | 14.765 |
| Teste | 3.692|


No segundo processamento, nenhum registro foi removido por outliers,
tanto no treino quanto no teste, utilizando os limites calculados a
partir do conjunto de treinamento.

Após o pré-processamento, o `OneHotEncoder` e o `StandardScaler`
produziram **156 atributos** para treino e teste.

## Resultados no conjunto de teste

O segundo modelo utiliza a mesma configuração do MLP:

``` python
MLPRegressor(
    hidden_layer_sizes=(100, 100),
    activation="relu",
    solver="adam",
    max_iter=250,
    random_state=42
)
```

Os resultados obtidos no conjunto de teste foram:

| Métrica | Resultado |
|---|---:|
| R² | 0,665 |
| MAE | 5,699 |
| MSE | 373,590 |
| RMSE | 19,328 |

O **R² de 0,665** indica que o modelo explicou aproximadamente 66,5% da
variabilidade observada nos preços do conjunto de teste.

O **MAE de 5,699** representa a diferença absoluta média entre os preços
previstos e os valores reais no teste.

O MSE e o RMSE atribuem maior peso aos erros elevados, e seus valores
foram 373,590 e 19,328, respectivamente.

Essa segunda avaliação é a principal referência para observar o
comportamento do modelo em dados que não participaram do treinamento.

## Visualizações

O notebook produz diferentes visualizações para analisar os dados e as
previsões, incluindo:

-   visualização dos valores ausentes;
-   heatmap de correlação;
-   preço real versus preço previsto;
-   quantidade de peças versus preço real;
-   quantidade de peças versus preço previsto;
-   quantidade de minifiguras versus preço real;
-   quantidade de minifiguras versus preço previsto.

Na etapa final, essas comparações são realizadas especificamente sobre o
conjunto de teste.

## Fluxo do projeto

``` text
Base LEGO
   │
   ├── Análise exploratória
   │     ├── Valores ausentes
   │     └── Correlações
   │
   ├── Seleção das variáveis
   │     ├── pieces
   │     ├── minifigs
   │     ├── agerange_min
   │     └── theme
   │
   ├── Divisão treino/teste
   │
   ├── Tratamento no treino
   │     ├── Outliers (IQR)
   │     └── Imputação pela moda
   │
   ├── Aplicação dos parâmetros no teste
   │
   ├── Pré-processamento
   │     ├── StandardScaler
   │     └── OneHotEncoder
   │
   ├── MLPRegressor
   │
   └── Avaliação
         ├── MAE
         ├── MSE
         ├── RMSE
         └── R²
```

## Observação sobre os resultados

O notebook apresenta duas avaliações com objetivos diferentes. O
primeiro resultado apresenta um R² de 0,886, mas foi obtido sobre dados
utilizados no treinamento e, portanto, pode fornecer uma estimativa
otimista do desempenho.

A segunda avaliação, utilizando uma separação de 80%/20% e realizando o
tratamento a partir dos parâmetros calculados no treino, apresenta R² de
0,665 no conjunto de teste. Por utilizar dados não empregados no
treinamento, essa avaliação fornece uma visão mais adequada da
capacidade de generalização do modelo.

## Arquivo principal

O projeto é apresentado no notebook:

``` text
ANN_LEGO.ipynb
```

O notebook contém todas as etapas de download, análise exploratória,
tratamento dos dados, treinamento da rede neural, geração das previsões,
visualizações e avaliação do modelo.
