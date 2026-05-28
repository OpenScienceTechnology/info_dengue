#!/usr/bin/env Rscript

# ============================================================================
# MASTER SCRIPT - SISTEMA INTEGRADO DE ANÁLISE EPIDEMIOLÓGICA
# ============================================================================
# Executa todos os módulos em sequência
# ============================================================================

cat("\n\n")
cat(paste(rep("═", 80), collapse = ""), "\n")
cat("  SISTEMA INTEGRADO DE ANÁLISE EPIDEMIOLÓGICA DE DENGUE\n")
cat("  Master Script - Execução de Todos os Módulos\n")
cat("  Campo Grande/MS - Dados InfoDengue 2015-2026\n")
cat(paste(rep("═", 80), collapse = ""), "\n\n")

# Timestamp global
timestamp_global_inicio <- Sys.time()

# Verificações iniciais
cat("[PRÉ-PROCESSAMENTO]\n")
cat("  Verificando ambiente R...\n")
cat("  Versão R:", R.version$version.string, "\n")
cat("  Plataforma:", .Platform$OS.type, "\n\n")

# Criar estrutura de diretórios
cat("  Criando estrutura de diretórios...\n")
dirs <- c("outputs", "outputs/data", "outputs/reports", "outputs/visualizations", 
          "outputs/dashboards", "outputs/logs", "cache")
for (d in dirs) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE)
  }
}
cat("  ✓ Diretórios criados\n\n")

# ============================================================================
# EXECUTAR CADA MÓDULO
# ============================================================================

cat("[EXECUTANDO MÓDULOS]\n\n")

# Módulo 1: Análise Principal
cat("1️⃣  MÓDULO 1: Limpeza e Análise Exploratória\n")
cat("   Arquivo: epi_analysis_main.R\n")
timestamp_m1 <- Sys.time()
source("epi_analysis_main.R", encoding = "UTF-8")
duracao_m1 <- difftime(Sys.time(), timestamp_m1, units = "mins")
cat("   ⏱️  Duração:", round(duracao_m1, 2), "minutos\n\n")

# Módulo 2: Machine Learning
cat("2️⃣  MÓDULO 2: Machine Learning e Modelagem\n")
cat("   Arquivo: epi_analysis_ml.R\n")
timestamp_m2 <- Sys.time()
source("epi_analysis_ml.R", encoding = "UTF-8")
duracao_m2 <- difftime(Sys.time(), timestamp_m2, units = "mins")
cat("   ⏱️  Duração:", round(duracao_m2, 2), "minutos\n\n")

# Módulo 3: Séries Temporais
cat("3️⃣  MÓDULO 3: Séries Temporais e Previsão\n")
cat("   Arquivo: epi_analysis_timeseries.R\n")
timestamp_m3 <- Sys.time()
source("epi_analysis_timeseries.R", encoding = "UTF-8")
duracao_m3 <- difftime(Sys.time(), timestamp_m3, units = "mins")
cat("   ⏱️  Duração:", round(duracao_m3, 2), "minutos\n\n")

# Módulo 4: Visualizações
cat("4️⃣  MÓDULO 4: Visualizações e Mapas\n")
cat("   Arquivo: epi_analysis_visualizacao.R\n")
timestamp_m4 <- Sys.time()
source("epi_analysis_visualizacao.R", encoding = "UTF-8")
duracao_m4 <- difftime(Sys.time(), timestamp_m4, units = "mins")
cat("   ⏱️  Duração:", round(duracao_m4, 2), "minutos\n\n")

# Módulo 5: Relatórios
cat("5️⃣  MÓDULO 5: Geração de Relatórios e Exportação\n")
cat("   Arquivo: epi_analysis_relatorios.R\n")
timestamp_m5 <- Sys.time()
source("epi_analysis_relatorios.R", encoding = "UTF-8")
duracao_m5 <- difftime(Sys.time(), timestamp_m5, units = "mins")
cat("   ⏱️  Duração:", round(duracao_m5, 2), "minutos\n\n")

# ============================================================================
# RESUMO EXECUTIVO FINAL
# ============================================================================

duracao_total <- difftime(Sys.time(), timestamp_global_inicio, units = "mins")

cat("\n")
cat(paste(rep("╔", 80), collapse = ""), "\n")
cat("║", paste(rep(" ", 76), collapse = ""), "║\n")
cat("║  ✓✓✓ ANÁLISE COMPLETA CONCLUÍDA COM SUCESSO ✓✓✓", 
    paste(rep(" ", 27), collapse = ""), "║\n")
cat("║", paste(rep(" ", 76), collapse = ""), "║\n")
cat(paste(rep("╚", 80), collapse = ""), "\n\n")

cat("═" %rep% 80, "\n")
cat("RESUMO FINAL DO SISTEMA\n")
cat("═" %rep% 80, "\n\n")

cat("TEMPO DE EXECUÇÃO:\n")
cat("  Módulo 1 (Análise Exploratória):", round(duracao_m1, 2), "minutos\n")
cat("  Módulo 2 (Machine Learning):", round(duracao_m2, 2), "minutos\n")
cat("  Módulo 3 (Séries Temporais):", round(duracao_m3, 2), "minutos\n")
cat("  Módulo 4 (Visualizações):", round(duracao_m4, 2), "minutos\n")
cat("  Módulo 5 (Relatórios):", round(duracao_m5, 2), "minutos\n")
cat("  ───────────────────────────────────────────────\n")
cat("  TOTAL:", round(duracao_total, 2), "minutos\n\n")

# Contar arquivos gerados
n_csv <- length(list.files("outputs/data", pattern = "\\.csv$"))\n n_png <- length(list.files("outputs/visualizations", pattern = "\\.png$"))
n_html <- length(list.files("outputs/visualizations", pattern = "\\.html$"))
n_xlsx <- length(list.files("outputs/reports", pattern = "\\.xlsx$"))
n_json <- length(list.files("outputs/reports", pattern = "\\.json$"))
n_zip <- length(list.files("outputs", pattern = "\\.zip$"))
n_txt <- length(list.files("outputs/reports", pattern = "\\.txt$"))

cat("ARQUIVOS GERADOS:\n")
cat("  CSV (Dados):", n_csv, "arquivos\n")
cat("  PNG (Gráficos):", n_png, "arquivos\n")
cat("  HTML (Mapas Interativos):", n_html, "arquivos\n")
cat("  XLSX (Planilhas):", n_xlsx, "arquivos\n")
cat("  JSON (Metadados):", n_json, "arquivos\n")
cat("  TXT (Relatórios):", n_txt, "arquivos\n")
cat("  ZIP (Compactado):", n_zip, "arquivos\n")
cat("  ───────────────────────────────────────────────\n")
cat("  TOTAL:", (n_csv + n_png + n_html + n_xlsx + n_json + n_zip + n_txt), "arquivos\n\n")

# Tamanho total de saída
tamanho_total <- sum(file.size(list.files("outputs", recursive = TRUE, full.names = TRUE)))
cat("TAMANHO TOTAL DE SAÍDA:", round(tamanho_total / 1024 / 1024, 2), "MB\n\n")

cat("ESTRUTURA DE SAÍDA:\n")
cat("  📁 outputs/\n")
cat("  ├── 📁 data/             → CSVs com dados processados\n")
cat("  ├── 📁 reports/          → Relatórios (TXT, XLSX, JSON, LOG)\n")
cat("  ├── 📁 visualizations/   → Gráficos (PNG) e Mapas (HTML)\n")
cat("  ├── 📁 dashboards/       → Dashboards interativos\n")
cat("  ├── 📁 logs/             → Logs de execução\n")
cat("  ├── EpiAnalysis_DENGUE_[timestamp].zip → Arquivo compactado\n")
cat("  └── ...\n\n")

cat("═" %rep% 80, "\n")
cat("PRÓXIMAS RECOMENDAÇÕES:\n")
cat("═" %rep% 80, "\n\n")
cat("1. Revisar o arquivo: outputs/reports/RELATORIO_ANALISE_DENGUE.txt\n")
cat("2. Abrir o Excel: outputs/reports/RELATORIO_ANALISE_DENGUE.xlsx\n")
cat("3. Visualizar gráficos em: outputs/visualizations/\n")
cat("4. Abrir mapa interativo: outputs/visualizations/13_mapa_interativo.html\n")
cat("5. Fazer download do ZIP: outputs/EpiAnalysis_DENGUE_[timestamp].zip\n\n")

cat("═" %rep% 80, "\n")
cat("MODELO DE ANÁLISE IMPLEMENTADO:\n")
cat("═" %rep% 80, "\n\n")
cat("✓ ANÁLISE EXPLORATÓRIA DE DADOS (EDA)\n")
cat("  • Limpeza e transformação de dados\n")
cat("  • Estatísticas descritivas completas\n")
cat("  • Análise de qualidade dos dados\n")
cat("  • Distribuição por município, ano, mês\n\n")

cat("✓ MACHINE LEARNING\n")
cat("  • Random Forest (Classificação de Risco)\n")
cat("  • Decision Tree (Interpretabilidade)\n")
cat("  • K-Means Clustering (Agrupamento)\n")
cat("  • Regressão Linear e Ridge (Previsão)\n")
cat("  • Detecção de Anomalias (Isolation Forest)\n\n")

cat("✓ ANÁLISE DE SÉRIES TEMPORAIS\n")
cat("  • Decomposição sazonal (Additive/Multiplicative)\n")
cat("  • ARIMA (Previsão 8 semanas)\n")
cat("  • Prophet (Previsão 12 meses)\n")
cat("  • Análise de tendências anuais\n")
cat("  • Indicadores de alerta precoce\n\n")

cat("✓ VISUALIZAÇÕES\n")
cat("  • Série temporal com tendências\n")
cat("  • Ranking de municípios\n")
cat("  • Padrões sazonais\n")
cat("  • Heatmaps de correlação\n")
cat("  • Boxplots por nível de risco\n")
cat("  • Previsões ARIMA e Prophet\n")
cat("  • Mapa interativo com Leaflet\n\n")

cat("✓ RELATÓRIOS E EXPORTAÇÃO\n")
cat("  • Relatório executivo em TXT\n")
cat("  • Planilhas em XLSX\n")
cat("  • Metadados em JSON\n")
cat("  • Logs estruturados\n")
cat("  • Arquivo ZIP compactado\n\n")

cat("═" %rep% 80, "\n")
cat("STATUS: ✓ SISTEMA OPERACIONAL\n")
cat("═" %rep% 80, "\n\n")

cat("🎉 Análise epidemiológica completa e pronta para apresentação!\n\n")
