# Desafio Kognita

# Table of Contents
1. [Introdução](#introdução)
    1. [Como usar o código](#uso)
2. [Análise dos dados](#analise)
3. [Modelagem](#modelo)
4. [Api](#api)
5. [Considerações futuras](#consideracoes)


# 1. Introdução
A tarefa de MLOps vem pra ajudar desde a extração de dados até o serviço de um modelo de machine learning, passando pela construção deste modelo. Neste desafio procurei me concentrar em alguma coisas que possam fazer a diferença neste processo, como focar na qualidade de código para que outras pessoas possam trabalhar em cima deste projeto e melhorar o que esta escrito, com isso em mente separei as etapas em entrypoints. Note que por conta do desafio pedir que tudo esteja num mesmo repositório temos de fazer isso do melhor jeito possível, o ideal neste caso seria um projeto separado pra cada ação, ou seja, um projeto para análise e etl outro para modelagem e outro para a api, outro fator seria todo o fluxo acontecendo em um cluster kubernetes através de jobs rodando num orquestrador de pipelines como por exemplo o Kubeflow (isso será tratado na seção final deste relatório) assim o melhor possível seria colocar entrypoints separados para cada uma destas ações, outro ponto de suma importância no mundo de machine learning é o reuso dos modelos, um modelo bem treinado para uma classificação por exemplo pode ser reusado ara outros tipos de dados, ou no máximo refatorado pra outro problema de negócio fazendo com que o tempo de modelagem caia bastante, por conta deste fator todo modelo produzido neste desafio gera um artefato que pode ser consumido depois pra inferência ou servido para uma api este artefato pode ser versionado garantindo um rollback se necessário, por fim temos uma API que irá servir o modelo salvo realizando apenas a inferência, isto é outra facilidade de salvar os artefatos.

Este desafio esta estruturado o mais próximo possível de todo o fluxo de machine learning em produção.
## 1.1 Como usar o código
Abra um terminal no diretório que deseja salvar o projeto e clone este através do comando

```bash
$ git clone 
```

A estrutura do projeto sera detalhada a cada passo da explicação. A instalação do projeto pode ser feita a partir do ```Makefile``` localizado na raiz do projeto

```
├── desafio
│   ├── api
│   ├── exploratory_data_analysis
│   ├── model
│   ├── persistence
├── Makefile
├── README.md
├── resources
├── setup.py
```
neste arquivo vc encontrará o comando

```bash
$ make install
``` 

pu pode usar 
```bash
$ pip install -e .
```
a partir do diretório raiz. Assim teremos o projeto instalado como pacote python junto com todas as dependências que estão localizadas no arquivo ```setup.py```.

Antes lembre-se de criar um ambiente virtual para este projeto que pode ser feito seguindo este comando
```bash
python3 -m venv /path/to/new/virtual/environment
```

# 2. Análise dos dados
O módulo de análise esta organizado da forma
```
├── desafio
│   ├── exploratory_data_analysis
│   │   ├── graphic.py
│   │   ├── info.py
│   │   ├── __init__.py
│   │   ├── prepare.py
│   │   └── stats.py
```
Neste módulo temos três endpoints.
## 2.2 info e stats
### 2.2.1 Rodando os entrypoint
O entrypoint ```info``` será o primeiro a ser executado, uma vez o projeto instalado basta digitarmos 
```bash
$ run_info --help
```
para sabermos o que deverá ser passado como argumento, encontraremos os seguintes argumentos:

```
Options:
  --dataset-path TEXT                       The path of the dataset to be analyzed
  --separator TEXT    default="\t"          The separator of the the file
  --encoding TEXT     default="utf-8"       The encoding of the file
  --replace BOOLEAN   default=True          Flag for replacing or not some value in the dataset
  --replace-ref TEXT  default="missing"     The value to replace
  --replace-for FLOAT                       The replacement will be performed bi this value
  --output-folder TEXT                      The folder to save the results
```

O segundo entrypoint sera o stats e rodará do mesmo jeito do anterior porém chamando 
```bash
$ run_stats
```
assim como antes use 
```bash
$ run_info --help
```
para ver a relação de parâmetros a serem passados
### 2.2.2 Analisando os resultados
O retorno deste endpoint será um dataframe de informações úteis
![dataframe_info](resources/info/info_dataframe.jpg)

e o dataframe original que foi lido. Abaixo temos os gráficos gerados pelos métodos de padronização dos dados.

**Data set original**
![original_dataset](resources/graphs/unchanged_features.jpg)

**MinMaxScaler**
![original_dataset](resources/graphs/min_max_scaler.jpg)

**Normalizer**
![original_dataset](resources/graphs/normalizer.jpg)

**PowerTransformer**
![original_dataset](resources/graphs/power_transformer.jpg)


Podemos ver a partir do info que existem alguns problemas nos dados, o primeiro e mais aparente é na coluna ```participacao_falencia_valor``` não tem nenhuma variância sendo composta por somente um único valor, isso será o suficiente pra descarta-lá mais a frente quando entrarmos no processo de feature engineering. Todas as colunas categóricas apresentam valores faltantes, a correção ou eliminação destes será importante pra reduzir o viés do modelo. Temos métodos comuns em data science para tratar estes valores em colunas categóricas, 
- **Deletar observações**:
podemos simplesmente deletar as observações com valores faltantes entretanto como pode ser visto nas figuras acima teremos um total de $43\%$ das observações eliminadas, tornando esta uma opção ruim.
- **Valores mais comuns**:
 Podemos então pensar em trocar os valores faltantes pelos mais comuns, entretanto,podemos ver na figura original do data set que a maioria dos valores se concentram no mesmo tipo, uma investigação maior nestas colunas nos mostra que 
```atividade_principal``` tem $20\%$ concentrado em ```com de equipamentos de informatica``` esta coluna em particular esta concentração não deveria importar tanto a nível de negócio, uma vez que a maioria das empresas que compram da xhealth tem a ver com informática (apesar da minha estranheza quanto a por exemplo termos uma empresa de de móveis e estofados comprando produtos eletrônicos para saúde), as outras colunas categóricas também apresentam o mesmo comportamento chegando a termos $89\%$ do valores em ```opcao_tributaria``` como ```simples nacional```, o que pode fazer sentido dado o modelo de negócio, mas para termos uma observação melhor deveríamos ter mais conhecimento do negócio o que neste momento não é o caso. Isto mostra que a opção de trocar os valores faltantes pelo mais comum iria dar um viés gigantesco ao modelo
- **Modelo capaz de lidar com esse problema**:
Podemos pensar num modelo capaz de olhar pra estas variáveis como independentes, assim excluiríamos estas considerando todas as outras menos a coluna alvo e usar um k-means por exemplo e depois tentar categorizar estas colunas, ou usar one-hot-encoder (uma vez que temos variáveis categóricas nominais) Esta seria uma opção esperta, porém com a distribuição dos datasets e a quantidade de valores faltantes nestas colunas fazem com que uma categorias para valores faltantes iria representar uma boa quantidade do dataset nas colunas ```opcao_tributaria``` e ```forma_pagamento```, já nas outras duas colunas temos menos de $1\%$ de valores faltantes entretanto com o conhecimento atual do problema e como estas se apresentam na distribuição dos valores esta opção não seria tão boa

Por estes pontos a melhor opção seria eliminar estas colunas do dataset.

As colunas ``month`` e ``year`` podem ser jogadas fora uma vez que neste dataset não temos como representar um cliente univocamente através de um id, uma vez que cada obervação do dataset é uma compra e não temos informação pra atrelar esta compra a um cliente único. Uma vez que pudéssemos fazer esta identificação poderíamos de alguma forma realizar uma agregação e ter a partir destas colunas uma linha temporal de compras, porém não é este o cenário assim estas colunas não apresentam valor pra predição.

Sabemos que um passo importante na construção de um modelo é dar uma escala que faça sentido aos dados, para isso escolhemos ```Normalizer(), MinMaxScaler()``` e ```PowerTransformer()``` para testarmos e decidirmos qual transformação se aplica melhor as dados existentes. Escolhemos ```PowerTransformer()``` por tentar dar uma distribuição mais gaussiana aos dados e isso aconteceu nas colunas onde não tínhamos uma concentração grande de dados em poucos valores (o que basicamente acontece com todas as features do dataset) porém como podemos ver esta apresentou alguns outliers o que não é bom. ```MinMaxScaler()``` por ser capaz de deixar por default os valores das colunas entre zero e um, fazendo com que possamos compara-los, tivemos um resultado "menos" gaussiano que o ```PowerTransformer()```  o que era de se esperar uma vez que este serve pra isso e o ```MinMaxScaler()``` não tem essa pretensão, observamos também uma transformação que nos deixa com menos outliers. Por fim pode ver a transformação pelo método ```Normalizer()``` que diferente dos outros acima atua nos valores das observações tratando-as como um vetor e como o objetivo de normalizar cada observação, observamos nesta um pouco mais de outliers que no resultado de ```MinMaxScaler()``` , fazendo com que escolhemos esta última como a transformação adequada a ser feita.

# Modelagem
O módulo onde estão os entrypoints dos modelos é organizado da seguinte forma
```
── model
│   │   ├── __init__.py
│   │   ├── logistic_regression_model
│   │   │   ├── __init__.py
│   │   │   └── main.py
│   │   ├── prediction
│   │   │   ├── __init__.py
│   │   │   └── prediction.py
│   │   ├── random_forest_model
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   └── utils
│   │       ├── grid_search.py
│   │       ├── __init__.py
│   │       └── results.py
```
Para rodar os modelos basta 
```bash
$ run_random_forest
```
e
```bash
$ run_logistic_regression
```

use o ```--help``` para ver a relação de parâmetros

Observe que temos dois modelos, foram estes dois que escolhemos para realizar a classificação de uma compra como calote ou não-calote. Como o problema será de classificação da compra escolhemos um modelo de regressão e outro de árvore de decisão e vamos comparar estes dois após aplicarmos a ambos um tunning de hiperparâmetros usando o ```Grid Search```.

Para o modelo de regressão logística já temo um dataset com a aplicação de um padronizador de features escolhido pela análise da seção anterior, já jogamos fora colunas com muito pouca variância, isso faz com que possamos tentar garantir a convergência.

Como podemos ver o modelo random forest performou muito melhor que o de regressão. Usaremos este modelo para predição.

**Random forest**

| Precision          | Recall             | F1 score          | Accuracy         | ROC AUC          |
|--------------------| ------------------ |------------------ |------------------|------------------|
| 0.8515850144092219 | 0.40892579138557344|0.5525300923220755 |0.8911659371269399|0.9009990463982573|
| 0.4974022422750889 | 0.6293028887735685 | 0.5556319205803741|0.8346029219487238|0.8492962414902909|

**Logistic Regression**

| Precision       | Recall             | F1 score          | Accuracy         | ROC AUC          |
| -----------     | ------------------ |------------------ |------------------|------------------|
|0.33994334277620397|0.3321224701608718|0.33598740047248227|0.7842931044284008|0.6822882677568207|
|0.3214844973379267|0.3551288704376405|0.3374702062957179|0.7708771530896481|0.6828364176964257|

**Roc curve for logistic regression**

![roc_lr](resources/graphs/lr_roc_auc.png)

**Roc curve for random forest**

![rf_roc](resources/graphs/rf_roc_auc.png)

Como temos o 0 como negativo e 1 como positivo e no nosso caso 0 significa pago e 1 calote e dado o resultado do modelo escolhi continuar com o random forest  antes da otimização pois com o precision maior temos menos falsos positivos que no nosso caso erraríamos pouco em relação a classificar como calote alguém que irá pagar, além de praticamente todas as métricas estarem melhor nesse modelo.

## 3.1 Predições

O entrypoint de predição esta dentro do pacote de modelos, teremos então este lendo um modelo salvo e recebendo um json como entrada e devolvendo este mesmo json com a coluna predita default com seu valor predito

# 4. Api
O pacote da api esta organizado da seguinte maneira

```
├── desafio
│   ├── api
│   │   ├── blueprints
│   │   │   ├── base.py
│   │   │   ├── create_app.py
│   │   │   ├── __init__.py
│   │   │   ├── prediction_blueprint.py
│   │   │   └── root_blueprint.py
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model
│   │   │   ├── __init__.py
│   │   │   ├── predict.py
│   │   ├── requests
│   │   │   └── validation.rest
│   │   └── resources
│   │       └── random_forest_model.sav
```
Para servir este modelo escrevi uma api muito simples que recebe como requisição um json com os parâmetros descrito na seção anterior e devolve estes parâmetros com a chave ```default``` com o valor da predição do modelo. No arquivo validation.rest temos um exemplo de dados a serem mandado para a api, que responderá no endpoint. 
```bash
POST http://0.0.0.0:8080/predict/ 
```
O dado de entrada do método post será da forma

```
{
    "default_3months": 0,
    "ioi_36months": 58.0,
    "ioi_3months": 18.236092447,
    "valor_por_vencer": 0.0,
    "valor_vencido": 0.0,
    "valor_quitado": 242100.7,
    "quant_protestos": 0,
    "valor_protestos": 0.0,
    "quant_acao_judicial": 0,
    "acao_judicial_valor": 0.0,
    "dividas_vencidas_valor": 0.0,
    "dividas_vencidas_qtd": 0,
    "falencia_concordata_qtd": 0,
    "valor_total_pedido": 34665.6749381255
}
```
Para colocar este serviço no ar basta declarar variáveis de ambiente 
```bash
HTTP_HOST=0.0.0.0
HTTP_RESPONSE=8080
```
Com estes valores sendo o default pra rodar o serviço localmente, e rodar no terminal

```
run_api
```
# 5. Considerações futuras
Como considerações finais deixarei alguns pensamentos sobre como melhorar este projeto e uma arquitetura desde a extração de dados até o serviço e possíveis ferramentas pra cada etapa

- A primeira etapa seria a de extração de dados de um banco ou de um object storage de alguma nuvem (AWS, GCP, et cetera)
um job poderá ser feito pra isso extraindo do banco ou object storage e jogando em tabelas destas nuvem, um job que rode diariamente pra alimentar os dados por exemplo, estes jobs rodariam em um orquestrador como o Kubeflow por exemplo, que roda em cima de um cluster kubernetes
- A seguir pode ser feito o etl usando o sql direto de uma nuvem pra isso poderia ser usado o Redash como ferramente que pluga na nuvem e realiza o sql junto de algumas visualizações possíveis dentro do Redash, ou pode sr feito um job que através de um template roda uma consulta dentro da nuvem e grava o resultado num object storage e pode ser lido pra realizar o etl usando o spark por exemplo e após feito o etl usando alguma ferramenta de visualização, isto pode ser feito localmente dentro de um notebook, porém o recomendado seria usar o orquestrador pra rodar num cluster onde o ambiente seja isolado e usando os recursos necessários, sem que haja recursos como memória e cpu jogados fora visando também o reuso do algoritmo ou do template sql para outros jobs
- A modelagem é feita e o modelo treina num local como o orquestrador com todas as vantagens ditas ao longo deste texto, podendo expor as métricas  de treino pra uma tabela podendo ser consultada comparando diferentes versões do modelo, e vendo a performance deste sempre gerando um artefato a ser consumido
- Por fim teremos uma api que irá consumir o artefato da etapa anterior e servir ao cliente.