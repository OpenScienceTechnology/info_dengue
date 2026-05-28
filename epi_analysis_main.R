# ============================================================================
# SISTEMA INTEGRADO DE ANÁLISE EPIDEMIOLÓGICA DE DENGUE
# Análise completa de dados de dengue com ML, DL e NN em R
# ============================================================================
#
# Projeto: InfoDengue Epidemiological Analysis System
# Foco: Campo Grande/MS (IBGE: 5002704)
# Período: 2015-2026
# Idioma: R (equivalente a sistema Python/Jupyter)
# Saídas: CSV, XLSX, TXT, LOG, PDF, PNG, HTML, JSON, PARQUET
#
# ============================================================================
# PARTE 1: CONFIGURAÇÃO INICIAL E CARREGAMENTO DE PACOTES
# ============================================================================

cat("\n")
cat("═" %rep% 80, "\n")
cat("  SISTEMA DE ANÁLISE EPIDEMIOLÓGICA DE DENGUE - Campo Grande/MS\n")
cat("═" %rep% 80, "\n")
cat("\n")

# Registro de execução
timestamp_inicio <- Sys.time()
cat("[LOG]", format(timestamp_inicio, "%Y-%m-%d %H:%M:%S"), "- Iniciando análise\n\n")

# Criar diretórios necessários
dirs_necessarios <- c(
  "outputs/reports",
  "outputs/data",
  "outputs/visualizations",
  "outputs/dashboards",
  "outputs/logs",
  "cache"
)

for (dir in dirs_necessarios) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  }
}

# ============================================================================
# INSTALAÇÃO E CARREGAMENTO DE PACOTES
# ============================================================================

# Lista de pacotes necessários
pacotes_necessarios <- c(
  # Data manipulation
  "tidyverse", "data.table", "dplyr", "tidyr", "stringr", 
  
  # Date/time
  "lubridate", "forecast", "tseries", "seasonal",
  
  # Visualization
  "ggplot2", "plotly", "leaflet", "ggmap", "scales",
  "cowplot", "gridExtra", "ggpubr",
  
  # Spatial
  "sf", "sp", "tmap", "rgdal",
  
  # Machine Learning
  "caret", "randomForest", "rpart", "e1071", "class",
  "dbscan", "kmeans", "isolation",
  
  # Time series
  "prophet", "forecastML", "modeltime",
  
  # Statistical modeling
  "glmnet", "MASS", "nlme", "lme4",
  
  # Reporting
  "rmarkdown", "knitr", "gt", "flextable", "pander",
  
  # Excel/CSV export
  "writexl", "readxl", "openxlsx",
  
  # JSON/Parquet
  "jsonlite", "arrow",
  
  # Utilities
  "lubridate", "glue", "purrr", "magrittr", "here",
  
  # Logging
  "futile.logger", "logging"
)

# Função para instalar e carregar pacotes
instalar_carregar_pacotes <- function(pacotes) {
  novos_pacotes <- pacotes[!(pacotes %in% installed.packages()[, "Package"])]
  
  if (length(novos_pacotes) > 0) {
    cat("[INFO] Instalando pacotes faltantes...\n")
    install.packages(novos_pacotes, quiet = TRUE, verbose = FALSE)
  }
  
  # Carregar todos os pacotes
  for (pacote in pacotes) {
    suppressWarnings(suppressMessages(library(pacote, character.only = TRUE)))
  }
  
  cat("[OK] Todos os pacotes carregados com sucesso\n\n")
}

# Executar instalação e carregamento
instalar_carregar_pacotes(pacotes_necessarios)

# ============================================================================
# PARTE 2: FUNÇÕES UTILITÁRIAS
# ============================================================================

# Função para logging estruturado
log_msg <- function(nivel, mensagem) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  msg_formatada <- sprintf("[%s] %s - %s", timestamp, nivel, mensagem)
  cat(msg_formatada, "\n")
  
  # Salvar em arquivo de log
  write(msg_formatada, file = "outputs/logs/execucao.log", append = TRUE)
}

# Função para loading bar
progress_bar <- function(iteracao, total) {
  percentual <- (iteracao / total) * 100
  barras <- round(percentual / 2)
  barra <- paste0("[", strrep("=", barras), strrep(" ", 50 - barras), "]")
  cat("\r", barra, sprintf("%.1f%%", percentual))
  if (iteracao == total) cat("\n")
}

# Função para cálculo de incidência por 100 mil
calcular_incidencia <- function(casos, populacao) {
  if (is.na(casos) | is.na(populacao) | populacao == 0) {
    return(NA)
  }
  (casos / populacao) * 100000
}

# Função para cálculo de taxa de crescimento
calcular_taxa_crescimento <- function(valor_final, valor_inicial) {
  if (valor_inicial == 0 | is.na(valor_inicial)) return(NA)
  ((valor_final - valor_inicial) / valor_inicial) * 100
}

# Função para normalização de dados
normalizar_dados <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

# ============================================================================
# PARTE 3: CARREGAMENTO E PREPARAÇÃO DE DADOS
# ============================================================================

log_msg("INFO", "Iniciando carregamento de dados")

# Caminho dos dados
data_dir <- "Dataset/Dengue/csv_archive"

# Listar arquivos CSV
arquivos_csv <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)

cat("Arquivos encontrados:\n")
print(arquivos_csv)
cat("\n")

# Função para carregar e combinar dados
carregar_dados_dengue <- function(diretorio) {
  files <- list.files(diretorio, pattern = "\\.csv$", full.names = TRUE)
  
  if (length(files) == 0) {
    log_msg("ERROR", "Nenhum arquivo CSV encontrado")
    stop("Nenhum arquivo CSV encontrado")
  }
  
  dados_lista <- list()
  
  for (i in seq_along(files)) {
    cat("Carregando arquivo", i, "de", length(files), "...\n")
    progress_bar(i, length(files))
    
    arquivo <- files[i]
    dados_lista[[i]] <- tryCatch(
      read.csv(arquivo, encoding = "UTF-8", stringsAsFactors = FALSE),
      error = function(e) {
        log_msg("WARN", paste("Erro ao ler", arquivo, ":", e$message))
        NULL
      }
    )
  }
  
  # Combinar todos os dados
  dados_combinados <- bind_rows(dados_lista)
  
  return(dados_combinados)
}

# Carregar dados
dados_brutos <- carregar_dados_dengue(data_dir)

log_msg("OK", sprintf("Dados carregados: %d linhas, %d colunas", 
                       nrow(dados_brutos), ncol(dados_brutos)))

# ============================================================================
# PARTE 4: LIMPEZA E TRANSFORMAÇÃO DE DADOS
# ============================================================================

log_msg("INFO", "Iniciando limpeza e transformação de dados")

dados <- dados_brutos %>%
  # Converter timestamp para data
  mutate(
    data = as.POSIXct(data_iniSE / 1000, origin = "1970-01-01", tz = "UTC"),
    ano = year(data),
    mes = month(data),
    semana_epi = week(data),
    trimestre = quarter(data),
    data_formatada = as.Date(data),
    
    # Preenchimento de valores faltantes
    casos = ifelse(is.na(casos), 0, casos),
    casos_est = ifelse(is.na(casos_est), 0, casos_est),
    Rt = ifelse(is.na(Rt), 0, Rt),
    
    # Conversão de valores
    incidencia_100k = calcular_incidencia(casos_est, pop),
    
    # Classificação de risco
    nivel_risco = case_when(
      Rt > 1.5 ~ "CRÍTICO",
      Rt > 1.0 ~ "ALERTA",
      Rt > 0.5 ~ "CONTROLE",
      TRUE ~ "CONTROLADO"
    ),
    
    # Classificação de incidência
    nivel_incidencia = case_when(
      p_inc100k > 300 ~ "ALTA",
      p_inc100k > 100 ~ "MÉDIA",
      TRUE ~ "BAIXA"
    )
  ) %>%
  # Remover duplicatas
  distinct() %>%
  # Ordenar por data
  arrange(data_formatada)

log_msg("OK", "Dados limpos e transformados")

# Estatísticas de limpeza
cat("\n=== RESUMO DA LIMPEZA DE DADOS ===\n")
cat("Total de registros original:", nrow(dados_brutos), "\n")
cat("Total de registros após limpeza:", nrow(dados), "\n")
cat("Colunas disponíveis:", ncol(dados), "\n")
cat("Período coberto:", min(dados$data_formatada), "a", max(dados$data_formatada), "\n")
cat("Municípios únicos:", n_distinct(dados$municipio_nome), "\n\n")

# ============================================================================
# PARTE 5: ANÁLISE EXPLORATÓRIA
# ============================================================================

log_msg("INFO", "Realizando análise exploratória")

# 5.1 Estatísticas gerais
cat("\n=== ESTATÍSTICAS GERAIS ===\n")
cat("Total de casos notificados:", sum(dados$casos, na.rm = TRUE), "\n")
cat("Total de casos estimados:", sum(dados$casos_est, na.rm = TRUE), "\n")
cat("Rt médio:", round(mean(dados$Rt, na.rm = TRUE), 3), "\n")
cat("Incidência média (por 100k):", round(mean(dados$p_inc100k, na.rm = TRUE), 2), "\n")

# 5.2 Resumo descritivo das variáveis numéricas
cat("\n=== RESUMO DESCRITIVO ===\n")
vars_numericas <- c("casos", "casos_est", "Rt", "pop", "p_inc100k", 
                    "tempmin", "tempmax", "tempmed", "umidmax", "umidmin", "umidmed")
vars_numericas <- vars_numericas[vars_numericas %in% names(dados)]

resumo_stats <- dados %>%
  select(all_of(vars_numericas)) %>%
  summarise(across(everything(), list(
    n = ~sum(!is.na(.)),
    media = ~mean(., na.rm = TRUE),
    mediana = ~median(., na.rm = TRUE),
    desvio_padrao = ~sd(., na.rm = TRUE),
    minimo = ~min(., na.rm = TRUE),
    maximo = ~max(., na.rm = TRUE)
  )))

print(resumo_stats)

# ============================================================================
# PARTE 6: ANÁLISE POR MUNICÍPIO
# ============================================================================

log_msg("INFO", "Análise por município")

resumo_municipal <- dados %>%
  group_by(municipio_nome) %>%
  summarise(
    total_casos = sum(casos, na.rm = TRUE),
    media_casos = mean(casos, na.rm = TRUE),
    max_casos = max(casos, na.rm = TRUE),
    media_rt = mean(Rt, na.rm = TRUE),
    media_incidencia = mean(p_inc100k, na.rm = TRUE),
    populacao = first(pop),
    num_semanas = n_distinct(semana_epi),
    nivel_risco_predominante = names(table(nivel_risco))[which.max(table(nivel_risco))],
    .groups = 'drop'
  ) %>%
  arrange(desc(total_casos))

cat("\n=== TOP 15 MUNICÍPIOS (por total de casos) ===\n")
print(head(resumo_municipal, 15))

# ============================================================================
# PARTE 7: ANÁLISE TEMPORAL
# ============================================================================

log_msg("INFO", "Análise temporal")

# 7.1 Série temporal por mês
serie_temporal_mensal <- dados %>%
  group_by(ano, mes) %>%
  summarise(
    total_casos = sum(casos, na.rm = TRUE),
    media_casos = mean(casos, na.rm = TRUE),
    media_rt = mean(Rt, na.rm = TRUE),
    media_incidencia = mean(p_inc100k, na.rm = TRUE),
    num_municipios = n_distinct(municipio_nome),
    .groups = 'drop'
  ) %>%
  mutate(ano_mes = as.Date(paste(ano, mes, "01", sep = "-")))

# 7.2 Série temporal por semana
serie_temporal_semanal <- dados %>%
  group_by(ano, semana_epi) %>%
  summarise(
    total_casos = sum(casos, na.rm = TRUE),
    media_casos = mean(casos, na.rm = TRUE),
    media_rt = mean(Rt, na.rm = TRUE),
    .groups = 'drop'
  )

cat("\n=== SÉRIE TEMPORAL MENSAL (Últimos 12 meses) ===\n")
print(tail(serie_temporal_mensal, 12))

# ============================================================================
# PARTE 8: ANÁLISE CAMPO GRANDE (MUNICÍPIO FOCAL)
# ============================================================================

log_msg("INFO", "Análise específica de Campo Grande/MS")

dados_campo_grande <- dados %>%
  filter(municipio_nome == "Campo Grande")

if (nrow(dados_campo_grande) > 0) {
  cat("\n=== CAMPO GRANDE/MS - ANÁLISE FOCADA ===\n")
  cat("Total de casos:", sum(dados_campo_grande$casos, na.rm = TRUE), "\n")
  cat("Casos estimados:", sum(dados_campo_grande$casos_est, na.rm = TRUE), "\n")
  cat("Rt médio:", round(mean(dados_campo_grande$Rt, na.rm = TRUE), 3), "\n")
  cat("Taxa de incidência média:", round(mean(dados_campo_grande$p_inc100k, na.rm = TRUE), 2), "\n")
  cat("População:", first(dados_campo_grande$pop), "\n")
  
  # Distribuição por período epidemiológico
  distribuicao_cg <- dados_campo_grande %>%
    group_by(ano) %>%
    summarise(
      total_casos = sum(casos, na.rm = TRUE),
      media_rt = mean(Rt, na.rm = TRUE),
      .groups = 'drop'
    )
  
  cat("\nDistribuição de casos por ano:\n")
  print(distribuicao_cg)
}

# ============================================================================
# PARTE 9: ANÁLISE DE RISCO E CLASSIFICAÇÃO
# ============================================================================

log_msg("INFO", "Análise de risco e classificação")

# Criar matriz de risco
risco_municipios <- dados %>%
  group_by(municipio_nome) %>%
  summarise(
    rt_medio = mean(Rt, na.rm = TRUE),
    incidencia_media = mean(p_inc100k, na.rm = TRUE),
    casos_totais = sum(casos, na.rm = TRUE),
    populacao = first(pop),
    .groups = 'drop'
  ) %>%
  mutate(
    classificacao_rt = case_when(
      rt_medio > 1.5 ~ "CRÍTICO",
      rt_medio > 1.0 ~ "ALERTA",
      rt_medio > 0.5 ~ "CONTROLE",
      TRUE ~ "CONTROLADO"
    ),
    classificacao_incidencia = case_when(
      incidencia_media > 300 ~ "ALTA",
      incidencia_media > 100 ~ "MÉDIA",
      TRUE ~ "BAIXA"
    ),
    risco_geral = case_when(
      (rt_medio > 1.5 & incidencia_media > 100) ~ "MUITO ALTO",
      (rt_medio > 1.0 & incidencia_media > 50) ~ "ALTO",
      (rt_medio > 0.5 | incidencia_media > 50) ~ "MÉDIO",
      TRUE ~ "BAIXO"
    )
  ) %>%
  arrange(desc(risco_geral), desc(rt_medio))

cat("\n=== MUNICÍPIOS POR NÍVEL DE RISCO ===\n")
print(table(risco_municipios$risco_geral))

cat("\nMunicípios em RISCO MUITO ALTO:\n")
print(filter(risco_municipios, risco_geral == "MUITO ALTO"))

# ============================================================================
# PARTE 10: ANÁLISE DE QUALIDADE DE DADOS
# ============================================================================

log_msg("INFO", "Análise de qualidade dos dados")

qualidade_dados <- data.frame(
  Campo = names(dados),
  Preenchimento_Pct = colSums(!is.na(dados)) / nrow(dados) * 100,
  Valores_Unicos = sapply(dados, function(x) n_distinct(x, na.rm = TRUE))
)

cat("\n=== QUALIDADE DOS DADOS ===\n")
print(head(qualidade_dados[order(qualidade_dados$Preenchimento_Pct), ], 20))

# ============================================================================
# PARTE 11: EXPORTAÇÃO PRELIMINAR
# ============================================================================

log_msg("INFO", "Exportando dados processados")

# Exportar dados processados
write.csv(dados, "outputs/data/dengue_processado.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(resumo_municipal, "outputs/data/resumo_municipal.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(risco_municipios, "outputs/data/classificacao_risco.csv", row.names = FALSE, fileEncoding = "UTF-8")

# Exportar para Excel
write_xlsx(list(
  Dados_Processados = dados,
  Resumo_Municipal = resumo_municipal,
  Classificacao_Risco = risco_municipios,
  Serie_Temporal_Mensal = serie_temporal_mensal
), "outputs/data/dengue_analise_completa.xlsx")

log_msg("OK", "Dados exportados para CSV e XLSX")

# ============================================================================
# PARTE 12: SALVAR AMBIENTE PARA PRÓXIMAS ETAPAS
# ============================================================================

# Salvar dados processados para uso em scripts subsequentes
saveRDS(list(
  dados = dados,
  resumo_municipal = resumo_municipal,
  risco_municipios = risco_municipios,
  serie_temporal_mensal = serie_temporal_mensal,
  dados_campo_grande = dados_campo_grande
), "cache/dados_processados.rds")

log_msg("OK", "Dados salvos em cache para próximas análises")

# ============================================================================
# RESUMO FINAL
# ============================================================================

timestamp_fim <- Sys.time()
duracao <- difftime(timestamp_fim, timestamp_inicio, units = "mins")

cat("\n")
cat("═" %rep% 80, "\n")
cat("  RESUMO DA EXECUÇÃO - PARTE 1 (LIMPEZA E ANÁLISE EXPLORATÓRIA)\n")
cat("═" %rep% 80, "\n")
cat("Tempo total:", round(duracao, 2), "minutos\n")
cat("Registros processados:", nrow(dados), "\n")
cat("Municípios analisados:", n_distinct(dados$municipio_nome), "\n")
cat("Período de análise:", min(dados$ano), "-", max(dados$ano), "\n")
cat("\nArquivos gerados em: outputs/\n")
cat("Logs em: outputs/logs/execucao.log\n")
cat("═" %rep% 80, "\n\n")

cat("[PRÓXIMOS PASSOS]\n")
cat("1. Executar: epi_analysis_ml.R - Análise com Machine Learning\n")
cat("2. Executar: epi_analysis_timeseries.R - Análise de Séries Temporais\n")
cat("3. Executar: epi_analysis_visualizacao.R - Gráficos e Mapas\n")
cat("4. Executar: epi_analysis_relatorios.R - Gerar Relatórios Finais\n\n")

# ============================================================================
# FIM DA PARTE 1
# ============================================================================
