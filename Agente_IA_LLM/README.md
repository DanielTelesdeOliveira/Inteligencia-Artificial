\# Gerador Automático de Flashcards com Inteligência Artificial



Projeto desenvolvido para a disciplina de \*\*Inteligência Artificial\*\*, do curso de Ciência da Computação da Pontifícia Universidade Católica de São Paulo (PUC-SP).



\## Integrantes



| Nome | RA |

|---|---|

| Daniel Teles de Oliveira | RA00334427 |

| João Victor Torres Soares | RA00330701 |



\## Sobre o projeto



O \*\*Gerador Automático de Flashcards\*\* é uma aplicação desenvolvida em Python que utiliza modelos de linguagem (\*Large Language Models – LLMs\*) para transformar artigos científicos em materiais de estudo.



A aplicação realiza buscas de artigos científicos na plataforma \[arXiv](https://arxiv.org/), recupera seus arquivos PDF e os encaminha ao modelo Gemini, do Google. A partir do conteúdo dos documentos, o modelo identifica conceitos relevantes e gera flashcards estruturados em JSON.



O objetivo é facilitar a assimilação e a revisão de conhecimentos científicos, produzindo diferentes tipos de perguntas e respostas que podem ser utilizados em ferramentas de aprendizagem, como o Anki.



\## Funcionalidades



\- \*\*Busca de artigos científicos:\*\* pesquisa artigos na API oficial do arXiv utilizando palavras-chave ou expressões definidas pelo usuário.

\- \*\*Download de documentos:\*\* recupera os arquivos PDF dos artigos encontrados, permitindo o processamento de seus conteúdos.

\- \*\*Integração com Google Gemini:\*\* utiliza um modelo de linguagem para analisar os documentos e gerar flashcards fundamentados exclusivamente nas informações dos artigos.

\- \*\*Geração de diferentes tipos de flashcards:\*\* produz três modalidades de perguntas e respostas:

&#x20; - \*\*Pergunta direta:\*\* verifica a compreensão de conceitos científicos específicos.

&#x20; - \*\*Cloze:\*\* apresenta uma frase contextualizada com uma lacuna a ser preenchida.

&#x20; - \*\*Comparação:\*\* solicita a identificação de semelhanças e diferenças entre conceitos, métodos ou técnicas.

\- \*\*Estruturação e validação dos dados:\*\* utiliza Pydantic para definir um esquema padronizado e validar a resposta gerada pelo modelo.

\- \*\*Visualização dos resultados:\*\* converte os flashcards em um DataFrame do Pandas, facilitando sua consulta e organização.



\## Tecnologias utilizadas



| Tecnologia | Aplicação |

|---|---|

| Python | Linguagem de programação principal. |

| Google Gemini | Modelo de linguagem utilizado para analisar os artigos e gerar os flashcards. |

| Google Gen AI SDK | Integração com os serviços de inteligência artificial do Google. |

| arXiv API | Pesquisa e recuperação de artigos científicos. |

| Requests | Realização de requisições HTTP para obtenção dos documentos. |

| Feedparser | Interpretação das respostas XML da API do arXiv. |

| Pydantic | Definição e validação da estrutura dos flashcards. |

| Pandas | Organização e visualização dos resultados em DataFrames. |

| Google Colab | Ambiente de desenvolvimento e execução do notebook. |



\## Funcionamento



O projeto é organizado em um pipeline que integra a busca de artigos científicos ao processamento de seus conteúdos por um modelo de linguagem.



\### 1. Configuração do ambiente



Inicialmente, são instaladas e importadas as bibliotecas necessárias para realizar as requisições, processar documentos e integrar os serviços de inteligência artificial.



A chave de acesso à API do Google Gemini é recuperada por meio do sistema de Secrets do Google Colab.



\### 2. Busca de artigos no arXiv



A função `arxiv\_query(queryString)` realiza uma requisição à API oficial do arXiv, utilizando a expressão de busca fornecida pelo usuário.



A pesquisa está configurada para recuperar até 25 artigos, ordenados por relevância em ordem decrescente.



Os resultados são interpretados pelo Feedparser, que disponibiliza os metadados dos artigos em uma estrutura de fácil manipulação.



\### 3. Download e envio dos artigos



A função `download\_arxiv\_pdfs(entries, limit=3)` identifica os links dos arquivos PDF dos artigos recuperados e realiza seus downloads.



Os documentos são armazenados temporariamente e enviados individualmente à API do Google Gemini, que recebe os arquivos como contexto para a geração dos flashcards.



Por padrão, são processados os três primeiros artigos encontrados.



\### 4. Geração dos flashcards



A função `generate\_flashcards\_arxiv(query)` integra as etapas anteriores:



1\. Pesquisa os artigos científicos no arXiv.

2\. Recupera os arquivos PDF dos três primeiros resultados.

3\. Envia os documentos ao modelo Gemini.

4\. Solicita a geração de flashcards individuais para cada artigo.

5\. Recupera a resposta em formato JSON.

6\. Valida os dados utilizando o modelo Pydantic.



A geração utiliza temperatura `0.2`, buscando respostas mais consistentes, e configura o retorno no formato JSON por meio de um esquema previamente definido.



O modelo utilizado no notebook é `gemini-4-hydrogen`.



\### 5. Estrutura dos flashcards



Cada flashcard possui os seguintes atributos:



| Atributo | Descrição |

|---|---|

| `conceito\_principal` | Principal conceito científico abordado. |

| `artigo\_origem` | Título do artigo de origem. |

| `fonte` | Seção ou localização da informação utilizada. |

| `pagina` | Número da página do PDF correspondente. |

| `pergunta\_direta` | Pergunta objetiva sobre o conceito. |

| `resposta\_direta` | Resposta concisa e fundamentada no artigo. |

| `pergunta\_cloze` | Frase contextualizada contendo uma lacuna. |

| `resposta\_cloze` | Conceito que preenche a lacuna. |

| `pergunta\_comparacao` | Pergunta para comparação entre conceitos. |

| `resposta\_comparacao` | Explicação das semelhanças e diferenças. |



A estrutura é definida utilizando duas classes:



\- `Flashcard`: representa individualmente cada cartão de estudo.

\- `FlashcardCollection`: representa a coleção de flashcards gerados.



\### 6. Visualização dos resultados



Após a geração, os dados podem ser apresentados em diferentes formatos.



O notebook demonstra a impressão da coleção em JSON, utilizando `model\_dump\_json()`, e sua conversão para um DataFrame do Pandas, facilitando a visualização dos cartões e de seus respectivos atributos.



\## Configuração e execução



\### Pré-requisitos



\- Conta Google com acesso ao Google Colab.

\- Chave de API do Google Gemini.

\- Acesso à internet para consultar o arXiv e utilizar a API do Gemini.



\### 1. Configuração da chave de API



No Google Colab, acesse a área \*\*Secrets\*\* e cadastre uma chave com o seguinte nome:



`GEMINI\_API\_KEY`



Insira como valor a chave de API obtida no Google AI Studio e habilite seu acesso ao notebook.



O código recupera a chave utilizando:



```python

from google.colab import userdata



GOOGLE\_API\_KEY = userdata.get('GEMINI\_API\_KEY')

```



\### 2. Instalação das dependências



Execute a seguinte célula no notebook:



```python

%pip install requests feedparser google-genai

```



\### 3. Execução



Execute as células do notebook sequencialmente, respeitando a ordem de configuração, importação, definição das funções e estruturas de dados.



Para realizar uma pesquisa, utilize:



```python

flashcards = generate\_flashcards\_arxiv(

&#x20;   "The impact of IA in software"

)

```



O argumento recebido pela função representa o tema ou a expressão utilizada para pesquisar artigos científicos.



\### 4. Visualização em JSON



Para visualizar a coleção de flashcards no formato JSON:



```python

print(flashcards.model\_dump\_json(indent=2))

```



\### 5. Visualização em DataFrame



Para converter os resultados em um DataFrame:



```python

dados\_cards = flashcards.model\_dump()



df\_flashcards = pd.DataFrame(

&#x20;   dados\_cards\["flashcards"]

)



display(df\_flashcards)

```



\## Considerações e limitações



\- A pesquisa recupera até 25 artigos por consulta, mas o processamento está limitado aos três primeiros documentos.

\- A geração depende da disponibilidade da API do Google Gemini e de uma chave de acesso válida.

\- A qualidade dos flashcards depende do conteúdo dos artigos recuperados e da capacidade do modelo de interpretar os documentos.

\- As informações geradas são estruturadas e validadas conforme o esquema Pydantic, mas isso não garante, por si só, a ausência de erros conceituais.

\- Embora o projeto produza dados estruturados que podem ser utilizados em ferramentas como o Anki, a exportação direta ou integração automática com essa plataforma ainda não faz parte da implementação atual.



\## Objetivo acadêmico



A atividade tem como objetivo explorar a integração de aplicações Python com serviços de inteligência artificial disponibilizados por Web APIs, utilizando modelos de linguagem para processamento de documentos e geração estruturada de informações.



O projeto demonstra, na prática, a combinação de diferentes tecnologias para desenvolver um fluxo automatizado de recuperação de informações científicas, interpretação de documentos e geração de materiais de apoio ao aprendizado.

