# 🦟 ArchiZikDen Dataset Downloader

**Info Dengue - Download automatizado de dados epidemiológicos de Dengue, Chikungunya e Zika para todos os municípios brasileiros, utilizando a API pública do InfoDengue e a base municipal do IBGE.**



\

---

## 📌 Sobre o projeto

O **ArchiZikDen Dataset Downloader** é um programa em Python desenvolvido para automatizar o download de dados epidemiológicos relacionados às principais arboviroses monitoradas no Brasil:

* 🦟 **Dengue**
* 🦠 **Chikungunya**
* 🧬 **Zika**

O sistema consulta a API pública do **InfoDengue** e utiliza a API de localidades do **IBGE** para obter automaticamente a lista oficial de municípios brasileiros, permitindo coletas em escala municipal, estadual, nacional ou apenas para capitais.

O projeto foi pensado para apoiar pesquisas em **Ciência de Dados**, **Epidemiologia Computacional**, **Saúde Pública**, **Vigilância Epidemiológica**, **Modelagem Preditiva** e **Análise Territorial de Arboviroses**.

---

## 🎯 Objetivo

O objetivo principal do programa é facilitar a construção de bases de dados epidemiológicas organizadas, padronizadas e reutilizáveis para análises computacionais sobre arboviroses no Brasil.

Com ele, é possível:

* baixar dados por município, estado, capitais ou Brasil inteiro;
* escolher a arbovirose de interesse;
* definir recortes temporais por ano, mês ou semana epidemiológica;
* salvar os dados em formato **CSV** ou **JSON**;
* gerar logs automáticos da execução;
* realizar análise exploratória inicial dos dados baixados;
* estruturar automaticamente os arquivos em diretórios organizados.

---

## 🧠 Contexto de uso

Este projeto pode ser utilizado em estudos sobre:

* evolução temporal da dengue, chikungunya e zika;
* análise espacial de arboviroses;
* construção de dashboards epidemiológicos;
* estudos de sazonalidade;
* modelagem preditiva de surtos;
* análise de risco por município;
* cruzamento com variáveis climáticas, populacionais e territoriais;
* projetos acadêmicos em Ciência de Dados e Saúde Pública.

---

## 🗂️ Arboviroses disponíveis

O programa permite selecionar uma das três doenças disponíveis no menu interativo:

```text
📁 Arbovirose
├── [1] 🦠 Chikungunya
├── [2] 🦟 Dengue
└── [3] 🧬 Zika
```

Cada opção é convertida internamente para o identificador utilizado pela API do InfoDengue.

---

## 🌎 Escopos geográficos disponíveis

O sistema permite quatro tipos de seleção territorial:

```text
📁 Base de dados
├── [1] 🌎 Nacional
├── [2] 🗺️ Estadual
├── [3] 🏛️ Capitais
└── [4] 📍 Municipal
```

### 🌎 Nacional

Executa a coleta para todos os municípios brasileiros disponíveis na base do IBGE.

Esse modo pode envolver aproximadamente **5.570 municípios**, dependendo da resposta atual da API do IBGE.

### 🗺️ Estadual

Permite selecionar uma Unidade da Federação, como:

```text
MS, SP, RJ, MG, PR, MT, GO, BA, PE, CE...
```

O programa carrega todos os municípios pertencentes à UF escolhida.

### 🏛️ Capitais

Executa a coleta apenas para as 27 capitais brasileiras.

Exemplos:

| UF | Capital        | Código IBGE |
| -- | -------------- | ----------- |
| MS | Campo Grande   | 5002704     |
| SP | São Paulo      | 3550308     |
| RJ | Rio de Janeiro | 3304557     |
| DF | Brasília       | 5300108     |
| MG | Belo Horizonte | 3106200     |

### 📍 Municipal

Permite selecionar um município de duas formas:

1. pelo **código IBGE**;
2. pelo **nome do município**.

Exemplo:

```text
Código IBGE: 5002704
Município: Campo Grande/MS
```

---

## 📅 Recortes temporais

O programa permite três formas de seleção temporal:

```text
📁 Data Epidemiológica
├── [1] 📅 Ano completo
├── [2] 📆 Mês
└── [3] 🗓️ Semana epidemiológica
```

### 📅 Ano completo

Permite baixar dados de um ou mais anos completos.

Exemplo:

```text
Ano início: 2022
Ano fim: 2024
```

O programa consulta da semana epidemiológica 01 até a 53.

### 📆 Mês

Permite escolher mês inicial e mês final.

O programa converte automaticamente o intervalo mensal em semanas epidemiológicas aproximadas.

Exemplo:

```text
Mês início: 1
Ano início: 2024
Mês fim: 5
Ano fim: 2024
```

### 🗓️ Semana epidemiológica

Permite definir diretamente o intervalo de semanas epidemiológicas.

Exemplo:

```text
Ano início: 2024
SE início: 1
Ano fim: 2024
SE fim: 20
```

---

## 📁 Formatos de saída

O programa permite salvar os dados em dois formatos:

```text
📁 Formato de saída
├── [1] 📄 CSV
└── [2] 🗂️ JSON
```

### 📄 CSV

Formato recomendado para:

* Excel;
* LibreOffice Calc;
* pandas;
* R;
* Power BI;
* dashboards;
* bancos relacionais;
* análise tabular.

### 🗂️ JSON

Formato recomendado para:

* APIs;
* aplicações web;
* armazenamento estruturado;
* integração com sistemas;
* pipelines de dados semiestruturados.

---

## 🏗️ Estrutura de diretórios gerada

Os arquivos são organizados automaticamente dentro da pasta `Base_de_dados`.

Exemplo de estrutura:

```text
Base_de_dados/
└── Arbovirose/
    ├── DENGUE/
    │   ├── Nacional/
    │   │   ├── ANO/
    │   │   │   ├── CSV/
    │   │   │   └── JSON/
    │   │   ├── MÊS/
    │   │   │   ├── CSV/
    │   │   │   └── JSON/
    │   │   └── SEMANA/
    │   │       ├── CSV/
    │   │       └── JSON/
    │   ├── Estadual/
    │   ├── Capitais/
    │   └── Municipal/
    ├── CHIKUNGUNYA/
    └── ZIKA/
```

Os nomes dos arquivos seguem o padrão:

```text
doenca_escopo_timestamp.extensao
```

Exemplo:

```text
dengue_nacional_20260528_143022.csv
zika_municipal_20260528_151245.json
chikungunya_estadual_20260528_160300.csv
```

---

## 🧾 Logs automáticos

A cada execução, o programa gera um arquivo de log com informações detalhadas sobre a operação.

Exemplo de nome de log:

```text
infodengue_csv_20260528_143022.log
infodengue_json_20260528_151245.log
```

O log registra:

* nome do script;
* versão;
* sistema operacional;
* versão do Python;
* host da máquina;
* arbovirose selecionada;
* escopo geográfico;
* filtro temporal;
* formato de saída;
* arquivo gerado;
* fonte dos códigos IBGE;
* quantidade de municípios processados;
* amostra de geocodes;
* data e hora de início;
* data e hora de fim;
* tempo total da execução;
* número de requisições;
* número de respostas com dados;
* número de erros ou municípios sem dados;
* taxa de sucesso;
* total de registros baixados;
* estatísticas exploratórias, quando disponíveis.

---

## 📊 Análise exploratória automática

Após o download, se a biblioteca `pandas` estiver instalada, o programa realiza uma análise exploratória inicial dos dados baixados.

A análise pode incluir:

* quantidade de linhas e colunas;
* estatísticas descritivas;
* valores mínimos;
* médias;
* medianas;
* valores máximos;
* quantidade de valores nulos;
* distribuição dos níveis de alerta;
* período coberto pelos dados.

### Colunas numéricas analisadas

O programa tenta analisar automaticamente colunas como:

```text
casos
casos_est
cases_est_min
cases_est_max
p_rt1
p_inc100k
Rt
pop
tempmin
tempmed
tempmax
umidmin
umidmed
umidmax
```

### Níveis de alerta

Os níveis de alerta são interpretados da seguinte forma:

| Nível | Classificação |
| ----- | ------------- |
| 1     | 🟢 Verde      |
| 2     | 🟡 Amarelo    |
| 3     | 🟠 Laranja    |
| 4     | 🔴 Vermelho   |

---

## 📚 Dicionário de dados

O programa exibe um dicionário de dados com os principais campos retornados pela API do InfoDengue.

| Campo              | Descrição                                                     |
| ------------------ | ------------------------------------------------------------- |
| `data_ini_SE`      | Primeiro dia da semana epidemiológica                         |
| `SE`               | Número da semana epidemiológica                               |
| `casos_est`        | Casos estimados por nowcasting                                |
| `cases_est_min`    | Limite inferior do intervalo de confiança dos casos estimados |
| `cases_est_max`    | Limite superior do intervalo de confiança dos casos estimados |
| `casos`            | Casos notificados por semana                                  |
| `p_rt1`            | Probabilidade de Rt ser maior que 1                           |
| `p_inc100k`        | Incidência estimada por 100 mil habitantes                    |
| `nivel`            | Nível de alerta epidemiológico                                |
| `Rt`               | Número reprodutivo efetivo                                    |
| `pop`              | População municipal estimada                                  |
| `tempmin`          | Temperatura mínima média semanal                              |
| `tempmed`          | Temperatura média semanal                                     |
| `tempmax`          | Temperatura máxima média semanal                              |
| `umidmin`          | Umidade relativa mínima média                                 |
| `umidmed`          | Umidade relativa média                                        |
| `umidmax`          | Umidade relativa máxima média                                 |
| `receptivo`        | Receptividade climática                                       |
| `transmissao`      | Indicador de transmissão                                      |
| `nivel_inc`        | Nível de incidência                                           |
| `notif_accum_year` | Casos acumulados no ano                                       |
| `Localidade_id`    | Divisão submunicipal, quando aplicável                        |

---

## 🔌 Fontes de dados utilizadas

O programa utiliza duas fontes principais:

### 🦟 InfoDengue

Fonte dos dados epidemiológicos de dengue, chikungunya e zika.

Endpoint utilizado:

```text
https://info.dengue.mat.br/api/alertcity
```

### 🗺️ IBGE Localidades

Fonte dos municípios brasileiros e seus códigos oficiais.

Endpoint utilizado:

```text
https://servicodados.ibge.gov.br/api/v1/localidades/municipios
```

---

## ⚙️ Tecnologias utilizadas

O programa foi desenvolvido em **Python** e utiliza as seguintes bibliotecas:

| Biblioteca    | Finalidade                          |
| ------------- | ----------------------------------- |
| `requests`    | Requisições HTTP para APIs          |
| `tqdm`        | Barra de progresso no terminal      |
| `texttable`   | Exibição de tabelas no console      |
| `pandas`      | Análise exploratória dos dados      |
| `csv`         | Escrita de arquivos CSV             |
| `json`        | Leitura e escrita de arquivos JSON  |
| `logging`     | Registro de logs                    |
| `datetime`    | Controle de datas e timestamps      |
| `pathlib`     | Manipulação de caminhos             |
| `platform`    | Informações do sistema operacional  |
| `socket`      | Identificação do host               |
| `unicodedata` | Normalização de nomes de municípios |

---

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/archizikden-dataset-downloader.git
cd archizikden-dataset-downloader
```

### 2. Crie um ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install requests tqdm texttable pandas
```

Ou, se houver um arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 📄 Sugestão de `requirements.txt`

```txt
requests
tqdm
texttable
pandas
```

---

## ▶️ Como executar

Execute o programa no terminal:

```bash
python archizikden_dataset_downloader_v1.0.py
```

Em alguns sistemas Linux/macOS, também é possível executar:

```bash
python3 archizikden_dataset_downloader_v1.0.py
```

---

## 🧭 Fluxo de execução

Ao iniciar o programa, o usuário passa por cinco etapas principais:

```text
PASSO 1 / 5 — Seleção da arbovirose
PASSO 2 / 5 — Seleção do escopo geográfico
PASSO 3 / 5 — Seleção do recorte temporal
PASSO 4 / 5 — Seleção do formato de saída
PASSO 5 / 5 — Download dos dados
```

Antes do download, o programa exibe um resumo da operação para confirmação.

---

## 💻 Exemplo de uso

Exemplo de operação:

```text
Arbovirose: Dengue
Escopo: Estadual — MS
Recorte temporal: Ano 2024
Formato: CSV
```

Resultado esperado:

```text
Base_de_dados/
└── Arbovirose/
    └── DENGUE/
        └── Estadual/
            └── ANO/
                └── CSV/
                    └── dengue_estadual_20260528_143022.csv
```

Também será gerado um arquivo de log:

```text
infodengue_csv_20260528_143022.log
```

---

## 🧠 Cache de municípios do IBGE

Para evitar consultas repetidas à API do IBGE, o programa cria um arquivo de cache local:

```text
ibge_municipios_cache.json
```

Esse cache é considerado válido por até **30 dias**.

Se o cache estiver expirado ou ausente, o programa tenta consultar novamente a API do IBGE.

Caso a API esteja indisponível, o sistema utiliza uma lista de fallback com capitais e cidades relevantes.

---

## 🛡️ Tratamento de erros

O programa possui mecanismos para lidar com falhas comuns, como:

* ausência de dependências obrigatórias;
* indisponibilidade temporária da API do IBGE;
* respostas vazias da API do InfoDengue;
* timeouts;
* erro HTTP 404;
* seleção inválida de menu;
* código IBGE inválido;
* município não encontrado;
* interrupção pelo usuário.

As requisições à API do InfoDengue utilizam:

```text
API_RETRIES = 3
API_TIMEOUT = 30
API_DELAY = 0.08
```

Isso reduz o risco de falhas por instabilidade temporária e evita excesso de requisições em curto intervalo.

---

## 🔍 Principais funcionalidades

* ✅ Download de dados de dengue, chikungunya e zika;
* ✅ Cobertura nacional com municípios obtidos via API IBGE;
* ✅ Seleção por Brasil, UF, capitais ou município;
* ✅ Busca municipal por nome ou código IBGE;
* ✅ Recorte temporal por ano, mês ou semana epidemiológica;
* ✅ Exportação em CSV e JSON;
* ✅ Criação automática de diretórios;
* ✅ Cache local da base de municípios;
* ✅ Logs detalhados por execução;
* ✅ Análise exploratória automática;
* ✅ Dicionário de dados no terminal;
* ✅ Interface interativa em linha de comando;
* ✅ Barra de progresso com `tqdm`;
* ✅ Tabelas formatadas com `texttable`.

---

## 🧪 Aplicações em Ciência de Dados

Os dados baixados podem ser utilizados em pipelines de:

### 📊 Análise exploratória

* distribuição temporal de casos;
* comparação entre municípios;
* identificação de sazonalidade;
* análise de incidência por 100 mil habitantes;
* avaliação de níveis de alerta.

### 🗺️ Análise espacial

* mapas epidemiológicos;
* comparação por UF;
* identificação de clusters territoriais;
* análise de capitais e regiões metropolitanas.

### 🤖 Machine Learning

* previsão de casos;
* classificação de risco epidemiológico;
* detecção de padrões sazonais;
* modelos de alerta antecipado;
* análise de importância de variáveis climáticas.

### 📈 Séries temporais

* modelagem de tendência;
* previsão semanal;
* análise de ciclos epidêmicos;
* estudos de defasagem climática.

### 🏥 Saúde Pública

* apoio à vigilância epidemiológica;
* priorização territorial;
* planejamento de ações preventivas;
* monitoramento de surtos;
* avaliação de cenários de risco.

---

## 📌 Exemplo de análise com pandas

Após gerar um arquivo CSV, é possível analisá-lo com `pandas`:

```python
import pandas as pd

df = pd.read_csv("Base_de_dados/Arbovirose/DENGUE/Estadual/ANO/CSV/dengue_estadual_20260528_143022.csv")

print(df.head())
print(df.info())
print(df.describe())
```

Exemplo de agrupamento por semana epidemiológica:

```python
casos_por_semana = df.groupby("SE")["casos"].sum()
print(casos_por_semana)
```

Exemplo de análise por nível de alerta:

```python
alertas = df["nivel"].value_counts().sort_index()
print(alertas)
```

---

## 📈 Possíveis visualizações

Os dados podem ser usados para gerar:

* gráficos de linha por semana epidemiológica;
* mapas coropléticos por município;
* dashboards em Streamlit, Dash ou Power BI;
* séries temporais por UF;
* ranking de municípios com maior incidência;
* comparação entre dengue, zika e chikungunya;
* mapas de calor;
* gráficos de sazonalidade.

---

## 🧱 Estrutura interna do programa

O programa é organizado em blocos funcionais:

```text
IMPORTS
CONSTANTES GLOBAIS
CORES ANSI
TEXTTABLE HELPERS
LOGGING
IBGE — CARREGAMENTO DE MUNICÍPIOS
BUSCA POR NOME
MENU — ARBOVIROSE
MENU — ESCOPO GEOGRÁFICO
MENU — RECORTE TEMPORAL
MENU — FORMATO DE SAÍDA
ESTRUTURA DE PASTAS
DOWNLOAD CORE
ANÁLISE EXPLORATÓRIA
DICIONÁRIO DE DADOS
VERIFICAÇÃO DE DEPENDÊNCIAS
MAIN
```

---

## 🧩 Funções principais

| Função                   | Finalidade                                               |
| ------------------------ | -------------------------------------------------------- |
| `main()`                 | Controla o fluxo principal do programa                   |
| `banner()`               | Exibe o cabeçalho inicial                                |
| `_check_deps()`          | Verifica dependências obrigatórias                       |
| `load_ibge_municipios()` | Carrega municípios do IBGE ou do cache                   |
| `_fetch_ibge_api()`      | Consulta a API de municípios do IBGE                     |
| `_parse_uf_map()`        | Organiza municípios por UF                               |
| `ask_disease()`          | Permite selecionar a arbovirose                          |
| `ask_scope()`            | Permite selecionar o escopo geográfico                   |
| `ask_date_range()`       | Permite selecionar o recorte temporal                    |
| `ask_output_format()`    | Permite escolher CSV ou JSON                             |
| `build_output_path()`    | Cria o caminho de saída dos arquivos                     |
| `_fetch_one()`           | Consulta a API InfoDengue para um município              |
| `download_all()`         | Executa o download para todos os municípios selecionados |
| `_save_csv()`            | Salva registros em CSV                                   |
| `_save_json()`           | Salva registros em JSON                                  |
| `explore()`              | Realiza análise exploratória automática                  |
| `show_data_dict()`       | Exibe o dicionário de dados                              |
| `write_log_entry()`      | Gera o log detalhado da execução                         |

---

## ⚠️ Observações importantes

1. O modo nacional pode demorar bastante, pois consulta milhares de municípios.
2. Alguns municípios podem retornar dados vazios para determinados períodos ou doenças.
3. A conversão de mês para semana epidemiológica é aproximada.
4. O programa depende da disponibilidade das APIs externas.
5. A quantidade de municípios pode variar conforme atualizações da base do IBGE.
6. Os dados retornados devem ser validados antes de uso em pesquisas, relatórios ou publicações.
7. Para análises científicas, recomenda-se documentar data de coleta, parâmetros e fonte dos dados.

---

## 🧾 Boas práticas recomendadas

Para uso acadêmico ou científico, recomenda-se:

* manter os arquivos de log junto com os datasets;
* registrar a data de execução;
* versionar os scripts no GitHub;
* separar dados brutos de dados tratados;
* documentar filtros aplicados;
* validar campos ausentes;
* verificar inconsistências temporais;
* citar adequadamente as fontes de dados;
* evitar sobrescrever arquivos antigos;
* usar ambientes virtuais Python.

---

## 📁 Sugestão de organização do repositório

```text
archizikden-dataset-downloader/
├── archizikden_dataset_downloader_v1.0.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── docs/
│   └── dicionario_dados.md
├── examples/
│   ├── exemplo_uso_csv.py
│   └── exemplo_analise_pandas.ipynb
├── Base_de_dados/
│   └── .gitkeep
└── logs/
    └── .gitkeep
```

---

## 🚫 Sugestão de `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
env/

# Logs
*.log

# Cache
ibge_municipios_cache.json

# Dados gerados
Base_de_dados/
*.csv
*.json

# Sistema operacional
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
```

Caso os datasets sejam grandes, recomenda-se utilizar **Git LFS** ou disponibilizar os arquivos em repositórios próprios de dados.

---

## 🧪 Sugestão de uso com Git LFS

Para versionar arquivos grandes:

```bash
git lfs install
git lfs track "*.csv"
git lfs track "*.json"
git add .gitattributes
git add .
git commit -m "Adiciona downloader ArchiZikDen e estrutura inicial"
git push origin main
```

---

## 🏷️ Sugestão de tópicos para GitHub

Adicione os seguintes tópicos ao repositório:

```text
python
infodengue
ibge
dengue
zika
chikungunya
arboviroses
epidemiologia
saude-publica
ciencia-de-dados
dados-abertos
vigilancia-epidemiologica
machine-learning
series-temporais
```

---

## 🧑‍💻 Autor

**VIANA**

Projeto desenvolvido para apoiar estudos em **Ciência de Dados**, **Epidemiologia Computacional** e **Análise de Arboviroses no Brasil**.

---

## 📚 Como citar este projeto

Caso utilize este programa em trabalhos acadêmicos, relatórios técnicos ou projetos científicos, recomenda-se citar o repositório da seguinte forma:

```text
VIANA. ArchiZikDen Dataset Downloader: ferramenta Python para download de dados epidemiológicos de Dengue, Chikungunya e Zika a partir do InfoDengue e IBGE. GitHub, 2026. Disponível em: https://github.com/SEU-USUARIO/archizikden-dataset-downloader. Acesso em: dia mês ano.
```

Modelo em ABNT:

```text
VIANA. ArchiZikDen Dataset Downloader: ferramenta Python para download de dados epidemiológicos de Dengue, Chikungunya e Zika a partir do InfoDengue e IBGE. GitHub, 2026. Disponível em: https://github.com/SEU-USUARIO/archizikden-dataset-downloader. Acesso em: 28 maio 2026.
```

---

## 📜 Licença

Este projeto pode ser distribuído sob a licença **MIT**, caso o autor deseje permitir uso, modificação e redistribuição do código.

Exemplo:

```text
MIT License

Copyright (c) 2026 VIANA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 🚀 Roadmap

Possíveis melhorias futuras:

* [ ] adicionar argumentos via linha de comando;
* [ ] permitir execução sem modo interativo;
* [ ] gerar metadados em YAML;
* [ ] salvar logs em pasta dedicada;
* [ ] exportar também em Parquet;
* [ ] adicionar suporte a DuckDB;
* [ ] gerar dashboards automáticos;
* [ ] criar mapas por município;
* [ ] integrar com Streamlit;
* [ ] adicionar testes unitários;
* [ ] adicionar validação automática dos dados;
* [ ] criar documentação técnica em `/docs`;
* [ ] adicionar notebooks de exemplo;
* [ ] criar pipeline para GitHub Actions.

---

## ✅ Conclusão

O **ArchiZikDen Dataset Downloader** é uma ferramenta prática e extensível para coleta automatizada de dados epidemiológicos de arboviroses no Brasil. Ao integrar dados do InfoDengue com a base territorial do IBGE, o programa facilita a criação de bases organizadas para análises em larga escala, contribuindo para estudos de saúde pública, vigilância epidemiológica, ciência de dados e modelagem preditiva.

Com suporte a múltiplos escopos geográficos, formatos de saída, logs detalhados e análise exploratória inicial, o projeto oferece uma base sólida para pesquisas, dashboards e sistemas inteligentes de monitoramento epidemiológico.
