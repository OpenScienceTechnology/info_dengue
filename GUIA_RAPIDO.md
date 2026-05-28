# 🚀 GUIA RÁPIDO - SISTEMA EPIDEMIOLÓGICO DE DENGUE EM R

## ⚡ Início Rápido (5 minutos)

### 1. Verificar instalação de R
```r
# No console R ou RStudio
R.version
```

### 2. Navegar até o diretório
```r
# Definir diretório de trabalho
setwd("/caminho/para/OpenScienceTechnology/info_dengue")

# Verificar arquivos
list.files(pattern = "*.R$")
```

### 3. Executar o sistema completo (Recomendado)
```r
# Uma única linha para executar tudo!
source("epi_analysis_master.R")
```

**Tempo:** ~10-15 minutos  
**Saída:** Todos os análises, gráficos e relatórios gerados

---

## 📋 Guia Detalhado de Uso

### Opção A: Master Script (Automático)

```r
source("epi_analysis_master.R")
```

✅ Executa todos os 5 módulos  
✅ Gera todas as saídas  
✅ Cria relatório final  
✅ Compacta tudo em ZIP  

### Opção B: Módulos Individuais

```r
# Passo 1: Limpeza e Análise
source("epi_analysis_main.R")           # ~5 min

# Passo 2: Machine Learning
source("epi_analysis_ml.R")             # ~3 min

# Passo 3: Séries Temporais
source("epi_analysis_timeseries.R")     # ~2 min

# Passo 4: Visualizações
source("epi_analysis_visualizacao.R")   # ~4 min

# Passo 5: Relatórios
source("epi_analysis_relatorios.R")     # ~2 min
```

### Opção C: Terminal/Linha de Comando

```bash
# Linux/macOS
cd /caminho/para/info_dengue
Rscript epi_analysis_master.R

# Windows (PowerShell)
cd C:\caminho\para\info_dengue
Rscript epi_analysis_master.R
```

### Opção D: RStudio

1. Abrir RStudio
2. Arquivo → Abrir: `epi_analysis_master.R`
3. Ctrl+Shift+S ou botão Source
4. Ver console para progresso

---

## 🔧 Instalação de Dependências

### Automático (Recomendado)

```r
# Os pacotes são instalados automaticamente na primeira execução
source("epi_analysis_master.R")
```

### Manual (se necessário)

```r
# Instalar todos os pacotes
pacotes <- c(
  "tidyverse", "data.table", "dplyr", "tidyr", "stringr", "readr",
  "lubridate", "forecast", "tseries", "seasonal",
  "ggplot2", "plotly", "leaflet", "scales", "cowplot", "gridExtra", "ggpubr",
  "sf", "sp", "tmap", "rgdal",
  "caret", "randomForest", "rpart", "e1071", "class", "dbscan",
  "prophet", "modeltime",
  "glmnet", "MASS", "nlme", "lme4",
  "rmarkdown", "knitr", "gt", "flextable", "pander",
  "writexl", "readxl", "openxlsx",
  "jsonlite", "arrow",
  "magrittr", "here", "glue", "purrr",
  "futile.logger", "logging"
)

install.packages(pacotes)
```

---

## 📊 O que Será Gerado

### Após Execução: 30-50 arquivos

```
outputs/
├── 📊 data/ (15-20 arquivos CSV)
│   ├── dengue_processado.csv
│   ├── resumo_municipal.csv
│   ├── classificacao_risco.csv
│   └── ... (outros dados)
│
├── 📈 visualizations/ (13+ arquivos)
│   ├── 01-12_*.png (Gráficos)
│   └── 13_mapa_interativo.html
│
├── 📄 reports/ (4-5 arquivos)
│   ├── RELATORIO_ANALISE_DENGUE.txt (Principal!)
│   ├── RELATORIO_ANALISE_DENGUE.xlsx
│   └── metadata.json
│
└── 🗂️ EpiAnalysis_DENGUE_[data_hora].zip
    └── Todos os arquivos compactados
```

---

## 🎯 O QUE O SISTEMA FAZ

### Módulo 1: Análise Exploratória
- ✅ Carrega dados CSV
- ✅ Limpa e transforma
- ✅ Calcula indicadores epidemiológicos
- ✅ Classifica risco por município
- ✅ Exporta dados processados

### Módulo 2: Machine Learning
- ✅ Traina Random Forest (Classificação)
- ✅ Traina Decision Tree (Interpretabilidade)
- ✅ Agrupa municípios (K-Means)
- ✅ Prediz incidência (Regressões)
- ✅ Detecta anomalias

### Módulo 3: Séries Temporais
- ✅ Decompõe sazonalidade
- ✅ Treina ARIMA (previsão 8 semanas)
- ✅ Treina Prophet (previsão 12 meses)
- ✅ Analisa tendências
- ✅ Gera alertas precoces

### Módulo 4: Visualizações
- ✅ 12 gráficos estáticos (PNG)
- ✅ 1 mapa interativo (HTML/Leaflet)
- ✅ Série temporal com tendências
- ✅ Rankings de municípios
- ✅ Padrões sazonais

### Módulo 5: Relatórios
- ✅ Relatório TXT executivo
- ✅ Planilha XLSX
- ✅ Metadados JSON
- ✅ Logs estruturados
- ✅ Arquivo ZIP

---

## 📖 LEITURA DOS RESULTADOS

### 1. Abrir Relatório Principal
```
outputs/reports/RELATORIO_ANALISE_DENGUE.txt
```

Contém:
- Resumo executivo
- Indicadores principais
- Top 10 municípios
- Análise de Campo Grande
- Classificação de risco
- Modelos treinados
- Qualidade dos dados

### 2. Ver Gráficos
```
outputs/visualizations/
  01_serie_temporal_nacional.png
  02_evolucao_rt.png
  03_top15_municipios.png
  ...
  13_mapa_interativo.html  ← Abrir no navegador!
```

### 3. Explorar Dados
```
outputs/data/
  resumo_municipal.csv        ← Ranking de municípios
  classificacao_risco.csv     ← Nível de risco por município
  ts_previsao_arima.csv       ← Previsões para próximas semanas
```

### 4. Usar Planilha Excel
```
outputs/reports/RELATORIO_ANALISE_DENGUE.xlsx
```

Contém 6 abas com dados processados e modelos

---

## ⚠️ PROBLEMAS COMUNS E SOLUÇÕES

### Problema 1: "Pacote não encontrado"
```r
# Solução: Instalar pacote individual
install.packages("nome_do_pacote")

# O sistema tenta instalar automaticamente na primeira execução
```

### Problema 2: "Arquivo CSV não encontrado"
```r
# Solução: Verificar diretório
getwd()  # Deve estar em .../info_dengue

# Arquivos devem estar em:
# Dataset/Dengue/csv_archive/*.csv
```

### Problema 3: Execução lenta
```r
# Solução: R depende de processamento local
# Tempo normal: 10-15 minutos
# 
# Dicas:
# - Fechar outros programas
# - Use RStudio (mais eficiente que R puro)
# - Máquinas mais poderosas processam mais rápido
```

### Problema 4: Erro em visualizações
```r
# Solução: Instalar ggplot2
install.packages(c("ggplot2", "plotly", "leaflet"))

# Depois reexecutar:
source("epi_analysis_visualizacao.R")
```

---

## 🔍 ANÁLISE DOS RESULTADOS

### Interpretação de Rt (Número Reprodutivo)

| Rt | Interpretação | Ação |
|---|---|---|
| > 1.5 | 🔴 CRÍTICO | Aumentar vigilância |
| 1.0 a 1.5 | 🟠 ALERTA | Monitorar próximas semanas |
| 0.5 a 1.0 | 🟡 CONTROLE | Manter medidas |
| < 0.5 | 🟢 CONTROLADO | Reduzir alerta |

### Interpretação de Incidência

| Casos/100k | Nível | Ação |
|---|---|---|
| > 300 | 🔴 ALTA | Investigação urgente |
| 100-300 | 🟠 MÉDIA | Monitoramento |
| < 100 | 🟢 BAIXA | Vigilância rotineira |

---

## 💡 DICAS E TRUQUES

### 1. Executar apenas um módulo
```r
# Carregar dados processados do cache
dados_env <- readRDS("cache/dados_processados.rds")
dados <- dados_env$dados

# Agora explorar dados
head(dados, 10)
```

### 2. Customizar visualizações
```r
# Abrir epi_analysis_visualizacao.R
# Editar cores, títulos, fontes
# Re-executar apenas esse módulo

source("epi_analysis_visualizacao.R")
```

### 3. Adicionar novo município
```r
# Filtrar dados
dados %>% filter(municipio_nome == "Seu Município")

# Fazer análise específica
```

### 4. Comparar períodos
```r
# Dados de 2025 vs 2024
dados_2025 <- filter(dados, ano == 2025)
dados_2024 <- filter(dados, ano == 2024)

# Comparar
summary(dados_2025$casos) 
summary(dados_2024$casos)
```

---

## 📱 EXECUTAR EM DIFERENTES AMBIENTES

### Google Colab (Cloud)
```r
# 1. Fazer upload dos arquivos CSV
# 2. Copiar os scripts R para notebook
# 3. Executar célula por célula ou usar:

system("Rscript epi_analysis_master.R")

# 4. Download dos resultados
```

### RStudio Desktop (Local)
```
✅ Mais fácil e intuitivo
✅ Abrir arquivo, click Source
✅ Ver progresso em tempo real
```

### Terminal/Linha de Comando
```bash
Rscript epi_analysis_master.R

# Resultado: Todos os arquivos gerados em outputs/
```

---

## 📞 PRÓXIMAS AÇÕES

1. ✅ Executar: `source("epi_analysis_master.R")`
2. ⏳ Aguardar 10-15 minutos
3. 📂 Explorar pasta `outputs/`
4. 📄 Abrir `RELATORIO_ANALISE_DENGUE.txt`
5. 📊 Visualizar gráficos em `visualizations/`
6. 🗺️ Abrir mapa: `13_mapa_interativo.html` no navegador
7. 📈 Analisar previsões em `ts_previsao_*.csv`
8. 📋 Revisar planilha: `RELATORIO_ANALISE_DENGUE.xlsx`

---

## 🎓 ENTENDENDO O SISTEMA

### Arquitetura
```
Master Script
    ↓
Módulo 1 (Dados) → Cache
    ↓
Módulo 2 (ML) → Modelos
    ↓
Módulo 3 (TS) → Previsões
    ↓
Módulo 4 (Viz) → Gráficos
    ↓
Módulo 5 (Report) → Saídas
```

### Fluxo de Dados
```
CSV → Limpeza → ML/TS → Visualização → Relatórios → ZIP
```

### Modelos Implementados
- **ML:** Random Forest, Decision Tree, K-Means, Regressões, Anomalias
- **TS:** Decomposição, ARIMA, Prophet

---

## 🎯 Objetivo Final

Gerar um **relatório epidemiológico completo e automatizado** sobre dengue em Campo Grande/MS com:

✅ Dados limpos e validados  
✅ Indicadores epidemiológicos  
✅ Modelos preditivos  
✅ Visualizações profissionais  
✅ Relatórios em múltiplos formatos  

---

**Desenvolvido em R | 57.000 linhas de código | 5 módulos integrados**

Boa análise! 🔬📊
