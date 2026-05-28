# ============================================================================
# Análise Simplificada de Dados de Dengue
# ============================================================================
# 
# Script R compacto para análise básica de dados de dengue
# ============================================================================

# Carregamento de pacotes
library(dplyr)
library(readr)

# Carregar dados
data_dir <- "Dataset/Dengue/csv_archive"
files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)

dengue <- bind_rows(lapply(files, read_csv, show_col_types = FALSE))

# Conversão de timestamp
dengue <- mutate(dengue, data = as.POSIXct(data_iniSE / 1000, origin = "1970-01-01"))

# Análise básica
cat("===== ANÁLISE DE DENGUE =====\n\n")

# 1. Estatísticas gerais
cat("ESTATÍSTICAS GERAIS:\n")
cat("Total de casos:", sum(dengue$casos, na.rm = TRUE), "\n")
cat("Municípios:", n_distinct(dengue$municipio_nome), "\n")
cat("Período:", min(dengue$data, na.rm = TRUE), "a", max(dengue$data, na.rm = TRUE), "\n")
cat("Rt médio:", round(mean(dengue$Rt, na.rm = TRUE), 3), "\n\n")

# 2. Top 10 municípios
cat("TOP 10 MUNICÍPIOS (por total de casos):\n")
top_10 <- dengue %>%
  group_by(municipio_nome) %>%
  summarise(
    total = sum(casos, na.rm = TRUE),
    media = mean(casos, na.rm = TRUE),
    rt = mean(Rt, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(desc(total)) %>%
  head(10)

print(top_10)

# 3. Análise temporal
cat("\n\nRESUMO MENSAL:\n")
temporal <- dengue %>%
  mutate(ano_mes = format(data, "%Y-%m")) %>%
  group_by(ano_mes) %>%
  summarise(
    casos = sum(casos, na.rm = TRUE),
    rt_medio = mean(Rt, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(ano_mes)

print(temporal)

# 4. Estatísticas descritivas
cat("\n\nESTATÍSTICAS DESCRITIVAS DE CASOS:\n")
print(summary(dengue$casos))

cat("\n\n=== Análise concluída ===\n")
