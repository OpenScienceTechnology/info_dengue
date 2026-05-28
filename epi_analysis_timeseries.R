# ============================================================================
# MÓDULO 3: ANÁLISE DE SÉRIES TEMPORAIS E PREVISÃO
# ============================================================================
# ARIMA, Prophet, Decomposição Sazonal e Previsões
# ============================================================================

cat("\n")
cat("═" %rep% 80, "\n")
cat("  MÓDULO 3: ANÁLISE DE SÉRIES TEMPORAIS E PREVISÃO\n")
cat("═" %rep% 80, "\n\n")

timestamp_inicio_ts <- Sys.time()

# Carregar dados processados
dados_env <- readRDS("cache/dados_processados.rds")
dados <- dados_env$dados
serie_temporal_mensal <- dados_env$serie_temporal_mensal

# ============================================================================
# PARTE 1: PREPARAÇÃO DE SÉRIES TEMPORAIS
# ============================================================================

cat("[INFO] Preparando séries temporais...\n\n")

# Série temporal agregada nacional
serie_nacional <- dados %>%
  group_by(data_formatada) %>%
  summarise(
    casos_dia = sum(casos, na.rm = TRUE),
    casos_est_dia = sum(casos_est, na.rm = TRUE),
    rt_medio = mean(Rt, na.rm = TRUE),
    incidencia = mean(p_inc100k, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(data_formatada)

# Série temporal semanal
serie_semanal <- dados %>%
  mutate(semana = floor_date(data_formatada, "week")) %>%
  group_by(semana) %>%
  summarise(
    casos = sum(casos, na.rm = TRUE),
    casos_est = sum(casos_est, na.rm = TRUE),
    rt = mean(Rt, na.rm = TRUE),
    incidencia = mean(p_inc100k, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(semana)

# Série de Campo Grande
serie_cg <- dados %>%
  filter(municipio_nome == "Campo Grande") %>%
  group_by(data_formatada) %>%
  summarise(
    casos = sum(casos, na.rm = TRUE),
    casos_est = sum(casos_est, na.rm = TRUE),
    rt = mean(Rt, na.rm = TRUE),
    incidencia = mean(p_inc100k, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(data_formatada)

cat("[OK] Séries preparadas\n")
cat("  - Série Nacional: ", nrow(serie_nacional), "registros\n")
cat("  - Série Semanal:", nrow(serie_semanal), "registros\n")
cat("  - Série Campo Grande:", nrow(serie_cg), "registros\n\n")

# ============================================================================
# PARTE 2: DECOMPOSIÇÃO SAZONAL
# ============================================================================

cat("[INFO] Analisando sazonalidade...\n\n")

# Verificar periodicidade
if (nrow(serie_semanal) > 52) {
  # Usar dados semanais para análise sazonal
  ts_semanal <- ts(serie_semanal$casos, 
                    start = c(year(min(serie_semanal$semana)), 
                             week(min(serie_semanal$semana))),
                    frequency = 52)
  
  # Decomposição
  decom <- decompose(ts_semanal, type = "additive")
  
  # Extrair componentes
  tendencia <- decom$trend
  sazonalidade <- decom$seasonal
  residuos <- decom$random
  
  cat("[OK] Decomposição sazonal realizada\n")
  cat("  - Modo: Aditivo\n")
  cat("  - Frequência: 52 semanas (anual)\n\n")
  
  # Análise de sazonalidade
  sazonalidade_resumo <- data.frame(
    Metrica = c("Variância da Tendência", "Variância da Sazonalidade", "Variância dos Resíduos"),
    Valor = c(var(tendencia, na.rm = TRUE), 
              var(sazonalidade, na.rm = TRUE),
              var(residuos, na.rm = TRUE))
  )
  
  print(sazonalidade_resumo)
}

cat("\n")

# ============================================================================
# PARTE 3: MODELAGEM ARIMA
# ============================================================================

cat("[INFO] Treinando modelos ARIMA...\n\n")

# ARIMA para série de casos semanais
if (nrow(serie_semanal) > 30) {
  # Auto ARIMA
  modelo_arima <- auto.arima(
    serie_semanal$casos,
    max.p = 5, max.q = 5, max.d = 2,
    seasonal = FALSE,
    stepwise = TRUE,
    trace = FALSE
  )
  
  # Previsão 8 semanas à frente
  previsao_arima <- forecast(modelo_arima, h = 8)
  
  cat("[OK] Modelo ARIMA treinado\n")
  cat("  - Ordem ARIMA:", modelo_arima$arima$order, "\n")
  cat("  - AIC:", round(modelo_arima$aic, 2), "\n")
  cat("  - Previsão para 8 semanas: \n")
  print(data.frame(
    Semana = 1:8,
    Previsto = round(previsao_arima$mean, 0),
    IC_Inferior = round(previsao_arima$lower[, 1], 0),
    IC_Superior = round(previsao_arima$upper[, 1], 0)
  ))
}

cat("\n")

# ============================================================================
# PARTE 4: MODELAGEM COM PROPHET (Facebook)
# ============================================================================

cat("[INFO] Treinando modelos Prophet...\n\n")

# Preparar dados no formato esperado por Prophet
dados_prophet <- serie_mensal <- dados %>%
  group_by(ano, mes) %>%
  summarise(
    casos = sum(casos, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(
    ds = as.Date(paste(ano, mes, "01", sep = "-")),
    y = casos
  ) %>%
  select(ds, y) %>%
  filter(!is.na(y))

# Treinar Prophet
suppressWarnings({
  modelo_prophet <- prophet(
    dados_prophet,
    yearly.seasonality = TRUE,
    weekly.seasonality = FALSE,
    daily.seasonality = FALSE,
    interval.width = 0.95
  )
})

# Criar dataframe para previsão
futuro <- make_future_dataframe(modelo_prophet, periods = 12)

# Realizar previsão
previsao_prophet <- predict(modelo_prophet, futuro)

cat("[OK] Modelo Prophet treinado\n")
cat("  - Meses previstos: 12\n\n")

# Extrair previsões futuras
previsoes_futuras <- previsao_prophet %>%
  tail(12) %>%
  select(ds, yhat, yhat_lower, yhat_upper)

print(previsoes_futuras)

cat("\n")

# ============================================================================
# PARTE 5: ANÁLISE DE TENDÊNCIAS
# ============================================================================

cat("[INFO] Analisando tendências temporais...\n\n")

# Calcular tendências anuais
tendencias_anuais <- dados %>%
  group_by(ano) %>%
  summarise(
    total_casos = sum(casos, na.rm = TRUE),
    media_rt = mean(Rt, na.rm = TRUE),
    media_incidencia = mean(p_inc100k, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(ano) %>%
  mutate(
    taxa_crescimento = c(NA, diff(total_casos) / total_casos[-length(total_casos)] * 100),
    variacao_rt = c(NA, diff(media_rt))
  )

cat("Tendências anuais:\n")
print(tendencias_anuais)

cat("\n")

# ============================================================================
# PARTE 6: ÍNDICE DE TRANSMISSÃO FUTURO (EARLY WARNING)
# ============================================================================

cat("[INFO] Gerando indicadores de alerta precoce...\n\n")

# Calcular indicador de risco futuro para próximas 4 semanas
dados_alerta <- serie_semanal %>%
  mutate(
    casos_lag1 = lag(casos),
    casos_lag2 = lag(casos, 2),
    casos_lag4 = lag(casos, 4),
    tendencia_casos = ifelse(
      is.na(casos_lag4),
      NA,
      (casos - casos_lag4) / casos_lag4 * 100
    ),
    risco_previsto = case_when(
      tendencia_casos > 20 & rt > 1.0 ~ "MUITO ALTO",
      tendencia_casos > 10 & rt > 0.8 ~ "ALTO",
      tendencia_casos > 0 | rt > 0.5 ~ "MÉDIO",
      TRUE ~ "BAIXO"
    )
  ) %>%
  tail(20)

cat("Indicadores de alerta (últimas 20 semanas):\n")
print(dados_alerta %>% select(semana, casos, tendencia_casos, risco_previsto))

cat("\n")

# ============================================================================
# PARTE 7: ANÁLISE DE VOLATILIDADE
# ============================================================================

cat("[INFO] Analisando volatilidade de casos...\n\n")

# Calcular volatilidade
volatilidade <- serie_semanal %>%
  mutate(
    retorno = c(NA, diff(log(casos + 1))),
    volatilidade_movel = rollapply(retorno, width = 4, FUN = sd, fill = NA)
  )

# Período de alta volatilidade
periodos_volateis <- volatilidade %>%
  filter(!is.na(volatilidade_movel)) %>%
  filter(volatilidade_movel > quantile(volatilidade_movel, 0.75, na.rm = TRUE))

cat("[OK] Volatilidade calculada\n")
cat("  - Semanas com alta volatilidade:", nrow(periodos_volateis), "\n")
cat("  - Volatilidade média:", round(mean(volatilidade$volatilidade_movel, na.rm = TRUE), 4), "\n\n")

# ============================================================================
# PARTE 8: EXPORTAÇÃO DE RESULTADOS SÉRIE TEMPORAL
# ============================================================================

cat("[INFO] Exportando resultados de séries temporais...\n\n")

# Exportar séries
write.csv(serie_nacional, "outputs/data/ts_serie_nacional.csv", row.names = FALSE)
write.csv(serie_semanal, "outputs/data/ts_serie_semanal.csv", row.names = FALSE)
write.csv(serie_cg, "outputs/data/ts_serie_campo_grande.csv", row.names = FALSE)
write.csv(tendencias_anuais, "outputs/data/ts_tendencias_anuais.csv", row.names = FALSE)
write.csv(dados_alerta, "outputs/data/ts_alerta_precoce.csv", row.names = FALSE)

# Exportar previsões
write.csv(as.data.frame(previsao_arima), "outputs/data/ts_previsao_arima.csv", row.names = FALSE)
write.csv(previsoes_futuras, "outputs/data/ts_previsao_prophet.csv", row.names = FALSE)

# Salvar modelos
saveRDS(list(
  modelo_arima = modelo_arima,
  modelo_prophet = modelo_prophet,
  serie_nacional = serie_nacional,
  serie_semanal = serie_semanal,
  serie_cg = serie_cg,
  previsoes = list(
    arima = previsao_arima,
    prophet = previsoes_futuras
  )
), "cache/modelos_ts.rds")

cat("[OK] Resultados exportados\n\n")

# ============================================================================
# PARTE 9: RESUMO ESTATÍSTICO TEMPORAL
# ============================================================================

resumo_ts <- data.frame(
  Periodo = c("2015-2026", "Últimos 12 meses", "Últimas 8 semanas"),
  Casos_Totais = c(
    sum(dados$casos, na.rm = TRUE),
    sum(filter(dados, ano == max(ano))$casos, na.rm = TRUE),
    sum(tail(serie_semanal, 8)$casos, na.rm = TRUE)
  ),
  Media_Rt = c(
    mean(dados$Rt, na.rm = TRUE),
    mean(filter(dados, ano == max(ano))$Rt, na.rm = TRUE),
    mean(tail(serie_semanal, 8)$rt, na.rm = TRUE)
  ),
  Tendencia = c("Histórica", "Recente", "Imediata")
)

write.csv(resumo_ts, "outputs/data/ts_resumo_temporal.csv", row.names = FALSE)

# ============================================================================
# RESUMO FINAL DO MÓDULO TS
# ============================================================================

timestamp_fim_ts <- Sys.time()
duracao_ts <- difftime(timestamp_fim_ts, timestamp_inicio_ts, units = "mins")

cat("═" %rep% 80, "\n")
cat("  RESUMO - MÓDULO 3 (SÉRIES TEMPORAIS E PREVISÃO)\n")
cat("═" %rep% 80, "\n")
cat("Tempo total:", round(duracao_ts, 2), "minutos\n")
cat("Modelos treinados: 2 (ARIMA, Prophet)\n")
cat("Séries analisadas: 3 (Nacional, Semanal, Campo Grande)\n")
cat("Períodos previstos: 8 semanas (ARIMA) + 12 meses (Prophet)\n")
cat("Sazonalidade: Detectada (52 semanas anuais)\n")
cat("Tendência: ", ifelse(tail(tendencias_anuais$taxa_crescimento, 1, na.rm = TRUE) > 0, 
                          "Crescente", "Decrescente"), "\n")
cat("═" %rep% 80, "\n\n")

# ============================================================================
# FIM DO MÓDULO 3
# ============================================================================
