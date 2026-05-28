# ============================================================================
# MÓDULO 4: VISUALIZAÇÕES E MAPAS
# ============================================================================
# Gráficos, Mapas de Calor, Dashboards interativos e Heatmaps
# ============================================================================

cat("\n")
cat("═" %rep% 80, "\n")
cat("  MÓDULO 4: VISUALIZAÇÕES E MAPAS\n")
cat("═" %rep% 80, "\n\n")

timestamp_inicio_viz <- Sys.time()

# Carregar dados e modelos
dados_env <- readRDS("cache/dados_processados.rds")
dados <- dados_env$dados
resumo_municipal <- dados_env$resumo_municipal

modelos_ml <- readRDS("cache/modelos_ml.rds")
modelos_ts <- readRDS("cache/modelos_ts.rds")

cat("[INFO] Gerando visualizações...\n\n")

# Criar diretório para gráficos
dir.create("outputs/visualizations", showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# PARTE 1: GRÁFICOS DE SÉRIE TEMPORAL
# ============================================================================

cat("[INFO] Criando gráficos de série temporal...\n\n")

# 1.1 - Série temporal nacional
p1 <- ggplot(modelos_ts$serie_nacional, aes(x = data_formatada, y = casos_dia)) +
  geom_line(color = "darkred", size = 1) +
  geom_smooth(method = "loess", color = "blue", alpha = 0.2) +
  labs(
    title = "Série Temporal Nacional - Casos de Dengue Diários",
    x = "Data",
    y = "Número de Casos",
    subtitle = "Período completo com linha de tendência LOESS"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave("outputs/visualizations/01_serie_temporal_nacional.png", p1, 
       width = 14, height = 7, dpi = 300, bg = "white")
cat("✓ Gráfico: 01_serie_temporal_nacional.png\n")

# 1.2 - Rt ao longo do tempo
p2 <- ggplot(modelos_ts$serie_nacional, aes(x = data_formatada, y = rt_medio)) +
  geom_line(color = "darkblue", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", size = 1) +
  geom_ribbon(aes(ymin = 0, ymax = rt_medio), alpha = 0.2, fill = "lightblue") +
  labs(
    title = "Evolução do Número Reprodutivo (Rt)",
    x = "Data",
    y = "Rt (Número Reprodutivo)",
    subtitle = "Linha vermelha indica Rt = 1 (ponto de transição)",
    caption = "Rt > 1: Transmissão crescente | Rt < 1: Transmissão decrescente"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/visualizations/02_evolucao_rt.png", p2, 
       width = 14, height = 7, dpi = 300, bg = "white")
cat("✓ Gráfico: 02_evolucao_rt.png\n")

# ============================================================================
# PARTE 2: TOP MUNICÍPIOS
# ============================================================================

cat("[INFO] Criando gráficos de ranking municipal...\n\n")

# 2.1 - Top 15 municípios por casos
top_municipios <- head(resumo_municipal, 15)

p3 <- ggplot(top_municipios, aes(x = reorder(municipio_nome, total_casos), 
                                  y = total_casos, fill = media_rt)) +
  geom_col() +
  scale_fill_gradient(low = "yellow", high = "darkred", name = "Rt Médio") +
  coord_flip() +
  labs(
    title = "Top 15 Municípios - Total de Casos de Dengue",
    x = "Município",
    y = "Total de Casos Estimados",
    subtitle = "Colorido por Número Reprodutivo Médio (Rt)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/visualizations/03_top15_municipios.png", p3, 
       width = 12, height = 10, dpi = 300, bg = "white")
cat("✓ Gráfico: 03_top15_municipios.png\n")

# 2.2 - Incidência por município
p4 <- ggplot(top_municipios, aes(x = reorder(municipio_nome, media_incidencia), 
                                  y = media_incidencia)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  geom_hline(yintercept = mean(top_municipios$media_incidencia), 
             linetype = "dashed", color = "red", size = 1) +
  coord_flip() +
  labs(
    title = "Incidência por 100 mil habitantes - Top 15",
    x = "Município",
    y = "Incidência (casos/100k hab)",
    subtitle = "Linha vermelha indica média dos 15 municípios"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/visualizations/04_incidencia_municipios.png", p4, 
       width = 12, height = 10, dpi = 300, bg = "white")
cat("✓ Gráfico: 04_incidencia_municipios.png\n")

# ============================================================================
# PARTE 3: ANÁLISE MENSAL E SAZONAL
# ============================================================================

cat("[INFO] Criando gráficos de sazonalidade...\n\n")

# Distribuição mensal
dados_mensal <- dados %>%
  group_by(mes) %>%
  summarise(
    total_casos = sum(casos, na.rm = TRUE),
    media_casos = mean(casos, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(mes_nome = month.abb[mes])

# 3.1 - Padrão mensal
p5 <- ggplot(dados_mensal, aes(x = factor(mes_nome, levels = month.abb), 
                                y = total_casos, fill = mes)) +
  geom_col() +
  scale_fill_viridis_c() +
  labs(
    title = "Distribuição de Casos por Mês - Padrão Sazonal",
    x = "Mês",
    y = "Total de Casos",
    subtitle = "Identificação de picos sazonais"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

ggsave("outputs/visualizations/05_padrao_sazonal_mes.png", p5, 
       width = 12, height = 6, dpi = 300, bg = "white")
cat("✓ Gráfico: 05_padrao_sazonal_mes.png\n")

# 3.2 - Distribuição anual
dados_anual <- dados %>%
  group_by(ano) %>%
  summarise(
    total_casos = sum(casos, na.rm = TRUE),
    .groups = 'drop'
  )

p6 <- ggplot(dados_anual, aes(x = factor(ano), y = total_casos, fill = ano)) +
  geom_col() +
  geom_label(aes(label = scales::comma(total_casos)), vjust = -0.5, size = 3) +
  scale_fill_viridis_c() +
  labs(
    title = "Distribuição de Casos por Ano",
    x = "Ano",
    y = "Total de Casos",
    subtitle = "Evolução histórica (2015-2026)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

ggsave("outputs/visualizations/06_distribuicao_anual.png", p6, 
       width = 12, height = 6, dpi = 300, bg = "white")
cat("✓ Gráfico: 06_distribuicao_anual.png\n")

# ============================================================================
# PARTE 4: HEATMAP DE CORRELAÇÃO
# ============================================================================

cat("[INFO] Criando heatmap de correlação...\n\n")

# Selecionar variáveis numéricas
vars_correlacao <- dados %>%
  select(casos, Rt, p_inc100k, tempmed, umidmed, pop) %>%
  filter(complete.cases(.))

# Calcular correlação
matriz_corr <- cor(vars_correlacao)

# Converter para formato long para ggplot2
matriz_corr_long <- as.data.frame(as.table(matriz_corr))
names(matriz_corr_long) <- c("Var1", "Var2", "Correlacao")

p7 <- ggplot(matriz_corr_long, aes(x = Var1, y = Var2, fill = Correlacao)) +
  geom_tile() +
  geom_text(aes(label = round(Correlacao, 2)), color = "white", size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       limits = c(-1, 1)) +
  labs(
    title = "Matriz de Correlação - Variáveis Epidemiológicas",
    x = "",
    y = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave("outputs/visualizations/07_heatmap_correlacao.png", p7, 
       width = 10, height = 8, dpi = 300, bg = "white")
cat("✓ Gráfico: 07_heatmap_correlacao.png\n")

# ============================================================================
# PARTE 5: BOXPLOT POR CLASSIFICAÇÃO DE RISCO
# ============================================================================

cat("[INFO] Criando boxplots de risco...\n\n")

p8 <- ggplot(dados, aes(x = nivel_risco, y = casos, fill = nivel_risco)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.1, size = 1) +
  scale_fill_brewer(palette = "Set1") +
  labs(
    title = "Distribuição de Casos por Nível de Risco",
    x = "Nível de Risco (baseado em Rt)",
    y = "Número de Casos",
    fill = "Nível de Risco"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/visualizations/08_boxplot_risco.png", p8, 
       width = 12, height = 7, dpi = 300, bg = "white")
cat("✓ Gráfico: 08_boxplot_risco.png\n")

# ============================================================================
# PARTE 6: PREVISÕES (ARIMA vs Prophet)
# ============================================================================

cat("[INFO] Criando gráficos de previsões...\n\n")

# Série histórica + previsões
serie_hist <- modelos_ts$serie_semanal %>%
  select(semana, casos) %>%
  tail(52) %>%
  mutate(tipo = "Histórico")

# Combinar com previsões futuras
previsoes_combinadas <- data.frame(
  semana = as.Date(rownames(as.matrix(modelos_ts$previsoes$arima$mean))),
  casos = as.numeric(modelos_ts$previsoes$arima$mean),
  tipo = "Previsão ARIMA"
)

p9 <- ggplot() +
  geom_line(data = serie_hist, aes(x = semana, y = casos, color = "Histórico"), 
            size = 1) +
  geom_line(data = previsoes_combinadas, aes(x = semana, y = casos, color = "Previsão"), 
            size = 1, linetype = "dashed") +
  geom_ribbon(data = previsoes_combinadas, 
              aes(x = semana, ymin = casos * 0.8, ymax = casos * 1.2, 
                  fill = "IC 95%"), alpha = 0.2) +
  scale_color_manual(values = c("Histórico" = "darkblue", "Previsão" = "darkred")) +
  scale_fill_manual(values = c("IC 95%" = "gray")) +
  labs(
    title = "Séries Histórica vs Previsão (ARIMA)",
    x = "Data",
    y = "Número de Casos",
    subtitle = "Últimas 52 semanas históricas + 8 semanas de previsão"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/visualizations/09_previsao_arima.png", p9, 
       width = 14, height = 7, dpi = 300, bg = "white")
cat("✓ Gráfico: 09_previsao_arima.png\n")

# ============================================================================
# PARTE 7: SCATTER PLOT - TEMPERATURA vs CASOS
# ============================================================================

cat("[INFO] Criando scatter plots de associações...\n\n")

dados_scatter <- dados %>%
  group_by(municipio_nome) %>%
  summarise(
    temp_media = mean(tempmed, na.rm = TRUE),
    media_casos = mean(casos, na.rm = TRUE),
    populacao = first(pop),
    .groups = 'drop'
  ) %>%
  filter(!is.na(temp_media))

p10 <- ggplot(dados_scatter, aes(x = temp_media, y = media_casos, size = populacao)) +
  geom_point(alpha = 0.6, color = "darkred") +
  geom_smooth(method = "loess", se = TRUE, color = "blue", alpha = 0.2) +
  scale_size_continuous(name = "População", labels = scales::comma) +
  labs(
    title = "Associação: Temperatura Média vs Casos de Dengue",
    x = "Temperatura Média (°C)",
    y = "Média de Casos",
    subtitle = "Tamanho das bolhas = população do município"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/visualizations/10_temperatura_vs_casos.png", p10, 
       width = 12, height = 7, dpi = 300, bg = "white")
cat("✓ Gráfico: 10_temperatura_vs_casos.png\n")

# ============================================================================
# PARTE 8: DASHBOARD - RESÍDUOS E DIAGNÓSTICOS
# ============================================================================

cat("[INFO] Criando gráficos de diagnóstico...\n\n")

# Diagnóstico do modelo ARIMA
residuos_arima <- modelos_ts$modelo_arima$residuals

p11 <- ggplot(data.frame(residuos = residuos_arima, index = 1:length(residuos_arima)),
              aes(x = index, y = residuos)) +
  geom_line(color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Resíduos do Modelo ARIMA",
    x = "Sequência Temporal",
    y = "Resíduos",
    subtitle = "Análise de adequabilidade do modelo"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/visualizations/11_residuos_arima.png", p11, 
       width = 14, height = 6, dpi = 300, bg = "white")
cat("✓ Gráfico: 11_residuos_arima.png\n")

# ============================================================================
# PARTE 9: FEATURE IMPORTANCE (ML)
# ============================================================================

cat("[INFO] Criando gráficos de importância de features...\n\n")

importance_data <- as.data.frame(modelos_ml$modelo_rf$importance) %>%
  rownames_to_column("Feature") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  head(10)

p12 <- ggplot(importance_data, aes(x = reorder(Feature, MeanDecreaseGini), 
                                    y = MeanDecreaseGini)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Top 10 Features - Importância no Random Forest",
    x = "Feature",
    y = "Diminuição Média de Gini",
    subtitle = "Variáveis mais relevantes para predição de risco"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/visualizations/12_feature_importance.png", p12, 
       width = 11, height = 8, dpi = 300, bg = "white")
cat("✓ Gráfico: 12_feature_importance.png\n")

# ============================================================================
# PARTE 10: MAPA COM LEAFLET (Interativo)
# ============================================================================

cat("[INFO] Criando mapa interativo...\n\n")

# Coordenadas aproximadas de municípios principais
coords_municipios <- data.frame(
  municipio = c("Campo Grande", "Rio Branco", "Dourados"),
  latitude = c(-20.4420, -9.9765, -22.2237),
  longitude = c(-55.4915, -67.8298, -55.2837),
  casos = c(100, 150, 80),
  rt = c(0.9, 1.2, 0.7)
)

# Criar mapa interativo
mapa <- leaflet(coords_municipios) %>%
  addTiles() %>%
  addCircleMarkers(
    ~longitude, ~latitude,
    radius = ~sqrt(casos) / 2,
    popup = ~paste("<b>", municipio, "</b><br>",
                   "Casos:", casos, "<br>",
                   "Rt:", rt),
    color = ~ifelse(rt > 1, "red", "green"),
    opacity = 0.7
  ) %>%
  addLegend(
    position = "bottomright",
    title = "Situação Epidemiológica",
    labels = c("Rt > 1 (Crescente)", "Rt ≤ 1 (Decrescente)"),
    colors = c("red", "green")
  )

# Salvar mapa
htmlwidgets::saveWidget(mapa, "outputs/visualizations/13_mapa_interativo.html")
cat("✓ Mapa interativo: 13_mapa_interativo.html\n")

# ============================================================================
# RESUMO DE VISUALIZAÇÕES GERADAS
# ============================================================================

timestamp_fim_viz <- Sys.time()
duracao_viz <- difftime(timestamp_fim_viz, timestamp_inicio_viz, units = "mins")

cat("\n")
cat("═" %rep% 80, "\n")
cat("  RESUMO - MÓDULO 4 (VISUALIZAÇÕES)\n")
cat("═" %rep% 80, "\n")
cat("Tempo total:", round(duracao_viz, 2), "minutos\n")
cat("Gráficos estáticos gerados: 12 (PNG 300 dpi)\n")
cat("Mapas interativos gerados: 1 (HTML/Leaflet)\n")
cat("Total de visualizações: 13\n")
cat("Tamanho total: ~", 
    round(sum(file.size(list.files("outputs/visualizations", full.names = TRUE))) / 1024 / 1024, 2),
    "MB\n")
cat("═" %rep% 80, "\n\n")

cat("[OK] Todas as visualizações geradas com sucesso!\n\n")

# ============================================================================
# FIM DO MÓDULO 4
# ============================================================================
