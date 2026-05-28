# ============================================================================
# Análise de Dados sobre Dengue - InfoDengue Dataset
# ============================================================================
# 
# Programa em R para análise exploratória e temporal dos dados de dengue
# do repositório Info Dengue
#
# Autor: Data Analysis Script
# Data: 2026
# ============================================================================

# Instalação e carregamento de pacotes necessários
packages <- c("dplyr", "ggplot2", "tidyr", "readr", "lubridate", "gridExtra")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) {
  install.packages(new_packages)
}

library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(lubridate)
library(gridExtra)

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================

# Definir diretório de dados
data_dir <- "Dataset/Dengue/csv_archive"

# Listar arquivos CSV disponíveis
csv_files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)
cat("Arquivos encontrados:\n")
print(csv_files)
cat("\n")

# Função para carregar e combinar dados de múltiplos arquivos
load_dengue_data <- function(directory) {
  files <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  
  if (length(files) == 0) {
    stop("Nenhum arquivo CSV encontrado no diretório especificado")
  }
  
  data_list <- lapply(files, function(file) {
    cat("Carregando:", file, "\n")
    read_csv(file, show_col_types = FALSE)
  })
  
  # Combinar todos os arquivos
  combined_data <- bind_rows(data_list)
  
  return(combined_data)
}

# Carregar dados
dengue_data <- load_dengue_data(data_dir)

# ============================================================================
# 2. EXPLORAÇÃO INICIAL DOS DADOS
# ============================================================================

cat("\n========== EXPLORAÇÃO INICIAL DOS DADOS ==========\n\n")

# Dimensões do dataset
cat("Dimensões do dataset:\n")
cat("Linhas:", nrow(dengue_data), "\n")
cat("Colunas:", ncol(dengue_data), "\n\n")

# Nomes das colunas
cat("Colunas disponíveis:\n")
print(names(dengue_data))
cat("\n")

# Tipos de dados
cat("Tipos de dados:\n")
print(str(dengue_data))
cat("\n")

# Primeiras linhas
cat("Primeiras linhas dos dados:\n")
print(head(dengue_data, 10))
cat("\n")

# ============================================================================
# 3. ESTATÍSTICAS DESCRITIVAS
# ============================================================================

cat("\n========== ESTATÍSTICAS DESCRITIVAS ==========\n\n")

# Selecionar variáveis numéricas principais
numeric_vars <- c("casos", "casos_est", "Rt", "pop", "tempmin", "tempmax", 
                   "tempmed", "umidmax", "umidmin", "umidmed", "p_inc100k")

# Filtrar apenas colunas que existem
numeric_vars <- numeric_vars[numeric_vars %in% names(dengue_data)]

# Resumo estatístico
cat("Resumo estatístico das variáveis principais:\n\n")
summary_stats <- dengue_data %>%
  select(all_of(numeric_vars)) %>%
  summarise(across(everything(), list(
    n = ~sum(!is.na(.)),
    media = ~mean(., na.rm = TRUE),
    mediana = ~median(., na.rm = TRUE),
    desvio_padrao = ~sd(., na.rm = TRUE),
    minimo = ~min(., na.rm = TRUE),
    maximo = ~max(., na.rm = TRUE)
  )))

print(summary_stats)
cat("\n")

# ============================================================================
# 4. ANÁLISE POR MUNICÍPIO/LOCALIDADE
# ============================================================================

cat("\n========== ANÁLISE POR MUNICÍPIO ==========\n\n")

# Estatísticas por município
municipal_summary <- dengue_data %>%
  group_by(municipio_nome) %>%
  summarise(
    num_registros = n(),
    total_casos = sum(casos, na.rm = TRUE),
    media_casos = mean(casos, na.rm = TRUE),
    max_casos = max(casos, na.rm = TRUE),
    media_rt = mean(Rt, na.rm = TRUE),
    populacao = first(pop),
    .groups = 'drop'
  ) %>%
  arrange(desc(total_casos))

cat("Top 10 municípios com maior número de casos estimados:\n")
print(head(municipal_summary, 10))
cat("\n")

# ============================================================================
# 5. ANÁLISE TEMPORAL
# ============================================================================

cat("\n========== ANÁLISE TEMPORAL ==========\n\n")

# Converter timestamp em data (assumindo que está em millisegundos)
dengue_data <- dengue_data %>%
  mutate(
    data = as.POSIXct(data_iniSE / 1000, origin = "1970-01-01", tz = "UTC"),
    ano = year(data),
    mes = month(data),
    semana = week(data)
  )

# Série temporal agregada
temporal_summary <- dengue_data %>%
  group_by(ano, mes) %>%
  summarise(
    total_casos = sum(casos, na.rm = TRUE),
    media_casos = mean(casos, na.rm = TRUE),
    media_rt = mean(Rt, na.rm = TRUE),
    num_municipios = n_distinct(municipio_nome),
    .groups = 'drop'
  ) %>%
  arrange(ano, mes)

cat("Resumo temporal (por ano e mês):\n")
print(head(temporal_summary, 20))
cat("\n")

# ============================================================================
# 6. CRIAR VISUALIZAÇÕES
# ============================================================================

cat("\n========== CRIANDO VISUALIZAÇÕES ==========\n\n")

# Criar diretório para gráficos
dir.create("outputs", showWarnings = FALSE)

# Gráfico 1: Série temporal de casos estimados
if (nrow(temporal_summary) > 0) {
  p1 <- ggplot(temporal_summary, aes(x = factor(ano), y = total_casos, fill = factor(mes))) +
    geom_col(position = "dodge") +
    labs(
      title = "Total de Casos de Dengue por Ano e Mês",
      x = "Ano",
      y = "Total de Casos Estimados",
      fill = "Mês"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("outputs/01_casos_por_ano_mes.png", plot = p1, width = 12, height = 6)
  cat("Gráfico salvo: outputs/01_casos_por_ano_mes.png\n")
}

# Gráfico 2: Evolução do Rt (número reprodutivo)
temporal_rt <- dengue_data %>%
  group_by(data) %>%
  summarise(media_rt = mean(Rt, na.rm = TRUE), .groups = 'drop') %>%
  filter(!is.na(media_rt))

if (nrow(temporal_rt) > 0) {
  p2 <- ggplot(temporal_rt, aes(x = data, y = media_rt)) +
    geom_line(color = "darkred", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
    labs(
      title = "Evolução do Número Reprodutivo (Rt) ao Longo do Tempo",
      x = "Data",
      y = "Rt (Média)",
      subtitle = "Linha tracejada indica Rt = 1 (controle da epidemia)"
    ) +
    theme_minimal()
  
  ggsave("outputs/02_evolucao_rt.png", plot = p2, width = 12, height = 6)
  cat("Gráfico salvo: outputs/02_evolucao_rt.png\n")
}

# Gráfico 3: Top 10 municípios com mais casos
top_10_municipal <- municipal_summary %>%
  slice_max(total_casos, n = 10)

if (nrow(top_10_municipal) > 0) {
  p3 <- ggplot(top_10_municipal, aes(x = reorder(municipio_nome, total_casos), 
                                      y = total_casos, fill = media_rt)) +
    geom_col() +
    coord_flip() +
    labs(
      title = "Top 10 Municípios com Maior Número de Casos de Dengue",
      x = "Município",
      y = "Total de Casos Estimados",
      fill = "Rt Médio"
    ) +
    theme_minimal() +
    scale_fill_gradient(low = "lightgreen", high = "darkred")
  
  ggsave("outputs/03_top10_municipios.png", plot = p3, width = 10, height = 8)
  cat("Gráfico salvo: outputs/03_top10_municipios.png\n")
}

# Gráfico 4: Correlação entre temperatura e casos
correlation_data <- dengue_data %>%
  group_by(ano, mes) %>%
  summarise(
    media_casos = mean(casos, na.rm = TRUE),
    media_temp = mean(tempmed, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  filter(!is.na(media_temp) & !is.na(media_casos))

if (nrow(correlation_data) > 0) {
  p4 <- ggplot(correlation_data, aes(x = media_temp, y = media_casos)) +
    geom_point(size = 3, alpha = 0.6, color = "darkblue") +
    geom_smooth(method = "loess", se = TRUE, color = "red") +
    labs(
      title = "Relação entre Temperatura Média e Casos de Dengue",
      x = "Temperatura Média (°C)",
      y = "Média de Casos Estimados",
      subtitle = "Com linha de tendência LOESS"
    ) +
    theme_minimal()
  
  ggsave("outputs/04_temperatura_vs_casos.png", plot = p4, width = 10, height = 6)
  cat("Gráfico salvo: outputs/04_temperatura_vs_casos.png\n")
  
  # Calcular correlação
  corr <- cor(correlation_data$media_temp, correlation_data$media_casos, 
              use = "complete.obs")
  cat("Correlação entre temperatura e casos:", round(corr, 3), "\n")
}

# ============================================================================
# 7. RELATÓRIO FINAL
# ============================================================================

cat("\n========== RELATÓRIO DE ANÁLISE ==========\n\n")

# Estatísticas gerais
total_casos <- sum(dengue_data$casos, na.rm = TRUE)
media_rt <- mean(dengue_data$Rt, na.rm = TRUE)
num_municipios <- n_distinct(dengue_data$municipio_nome)
periodo_ini <- min(dengue_data$data, na.rm = TRUE)
periodo_fim <- max(dengue_data$data, na.rm = TRUE)

cat("RESUMO EXECUTIVO\n")
cat("===============\n\n")
cat("Total de casos registrados:", total_casos, "\n")
cat("Número de municípios:", num_municipios, "\n")
cat("Período de análise:", as.Date(periodo_ini), "a", as.Date(periodo_fim), "\n")
cat("Número reprodutivo médio (Rt):", round(media_rt, 3), "\n\n")

# Classificação de situação epidemiológica
if (media_rt > 1.5) {
  situacao <- "CRÍTICA - Epidemia em aceleração"
} else if (media_rt > 1) {
  situacao <- "ALERTA - Epidemia em expansão"
} else if (media_rt > 0.5) {
  situacao <- "CONTROLE - Epidemia em declínio"
} else {
  situacao <- "CONTROLADA - Sem transmissão"
}

cat("Situação epidemiológica geral:", situacao, "\n\n")

# Ranking de municípios
cat("RANKING DE MUNICÍPIOS (Top 5)\n")
cat("==============================\n\n")
for (i in 1:min(5, nrow(municipal_summary))) {
  row <- municipal_summary[i, ]
  cat(i, ".", row$municipio_nome, "\n")
  cat("   - Total de casos: ", row$total_casos, "\n")
  cat("   - Média de casos: ", round(row$media_casos, 1), "\n")
  cat("   - Número reprodutivo: ", round(row$media_rt, 3), "\n\n")
}

cat("\n========== ANÁLISE CONCLUÍDA COM SUCESSO ==========\n\n")
cat("Gráficos e resultados salvos em: outputs/\n")

# ============================================================================
# 8. SALVAR DADOS PROCESSADOS
# ============================================================================

# Salvar dados processados como CSV para análises futuras
write_csv(municipal_summary, "outputs/resumo_municipal.csv")
write_csv(temporal_summary, "outputs/resumo_temporal.csv")

cat("Dados processados salvos:\n")
cat("  - outputs/resumo_municipal.csv\n")
cat("  - outputs/resumo_temporal.csv\n\n")

# ============================================================================
# FIM DO SCRIPT
# ============================================================================
