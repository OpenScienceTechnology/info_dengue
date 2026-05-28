# ============================================================================
# MÓDULO 2: MACHINE LEARNING PARA ANÁLISE EPIDEMIOLÓGICA DE DENGUE
# ============================================================================
# Machine Learning, Clustering, Classificação e Explicabilidade
# ============================================================================

cat("\n")
cat("═" %rep% 80, "\n")
cat("  MÓDULO 2: MACHINE LEARNING E MODELAGEM PREDITIVA\n")
cat("═" %rep% 80, "\n\n")

timestamp_inicio_ml <- Sys.time()

# Carregar dados processados
dados_env <- readRDS("cache/dados_processados.rds")
dados <- dados_env$dados
resumo_municipal <- dados_env$resumo_municipal
risco_municipios <- dados_env$risco_municipios

# ============================================================================
# PARTE 1: PREPARAÇÃO DE DADOS PARA ML
# ============================================================================

cat("[INFO] Preparando dados para Machine Learning...\n\n")

# Criar dataset de features por município-semana
features_dataset <- dados %>%
  group_by(municipio_nome, ano, semana_epi) %>%
  summarise(
    casos = sum(casos, na.rm = TRUE),
    casos_est = sum(casos_est, na.rm = TRUE),
    rt = mean(Rt, na.rm = TRUE),
    incidencia = mean(p_inc100k, na.rm = TRUE),
    temp_media = mean(tempmed, na.rm = TRUE),
    umid_media = mean(umidmed, na.rm = TRUE),
    transmissao = first(transmissao),
    receptivo = first(receptivo),
    populacao = first(pop),
    .groups = 'drop'
  ) %>%
  mutate(
    # Features de lag (semana anterior)
    casos_lag1 = lag(casos),
    casos_lag2 = lag(casos, 2),
    # Variação de temperatura
    variacao_temp = abs(lead(temp_media) - lag(temp_media)),
    # Target para classificação: Risco
    risco_semana = case_when(
      rt > 1.5 ~ "CRITICO",
      rt > 1.0 ~ "ALERTA",
      rt > 0.5 ~ "CONTROLE",
      TRUE ~ "CONTROLADO"
    ),
    # Target para regressão: Incidência
    incidencia_futura = lead(incidencia)
  ) %>%
  filter(!is.na(incidencia_futura), !is.na(casos_lag1))

cat("[OK] Dataset de features criado:\n")
cat("  - Registros:", nrow(features_dataset), "\n")
cat("  - Features numéricas:", ncol(features_dataset) - 6, "\n\n")

# Dataset para clustering
dados_clustering <- resumo_municipal %>%
  select(
    municipio_nome,
    total_casos,
    media_casos,
    media_rt,
    media_incidencia,
    populacao
  ) %>%
  mutate(
    casos_per_capita = total_casos / populacao,
    casos_normalizados = normalizar_dados(total_casos),
    rt_normalizados = normalizar_dados(media_rt),
    incidencia_normalizados = normalizar_dados(media_incidencia)
  ) %>%
  filter(!is.na(media_rt))

# ============================================================================
# PARTE 2: CLASSIFICAÇÃO DE RISCO COM RANDOM FOREST
# ============================================================================

cat("[INFO] Treinando modelo Random Forest para classificação de risco...\n\n")

# Preparar dados para treinamento
dados_ml <- features_dataset %>%
  select(
    casos, casos_est, rt, incidencia, temp_media, umid_media,
    transmissao, receptivo, casos_lag1, casos_lag2,
    risco_semana
  ) %>%
  filter(complete.cases(.))

# Split treino/teste
set.seed(42)
indice_treino <- createDataPartition(dados_ml$risco_semana, p = 0.8, list = FALSE)
treino_rf <- dados_ml[indice_treino, ]
teste_rf <- dados_ml[-indice_treino, ]

# Treinar Random Forest
modelo_rf <- randomForest(
  as.factor(risco_semana) ~ .,
  data = treino_rf[, -which(names(treino_rf) == "risco_semana")] %>%
    select(-c(casos_est, transmissao, receptivo)) %>%
    select(where(is.numeric)),
  ntree = 100,
  mtry = 3,
  importance = TRUE,
  verbose = FALSE
)

# Avaliar modelo
predicoes_rf <- predict(modelo_rf, 
                        teste_rf[, -which(names(teste_rf) %in% 
                                         c("risco_semana", "casos_est", "transmissao", "receptivo"))] %>%
                          select(where(is.numeric)))

acuracia_rf <- sum(predicoes_rf == teste_rf$risco_semana) / nrow(teste_rf)

cat("[OK] Modelo Random Forest treinado\n")
cat("  - Acurácia no conjunto teste:", round(acuracia_rf * 100, 2), "%\n")
cat("  - Árvores:", modelo_rf$ntree, "\n\n")

# Importância das features
importancia_rf <- data.frame(
  Feature = names(modelo_rf$importance[, 1]),
  Importancia = modelo_rf$importance[, 1]
) %>%
  arrange(desc(Importancia))

cat("Top 10 Features (Importância):\n")
print(head(importancia_rf, 10))
cat("\n")

# ============================================================================
# PARTE 3: ÁRVORE DE DECISÃO PARA INTERPRETABILIDADE
# ============================================================================

cat("[INFO] Treinando árvore de decisão para interpretabilidade...\n\n")

# Treinar Decision Tree (mais interpretável)
modelo_dt <- rpart(
  as.factor(risco_semana) ~ casos + casos_lag1 + rt + incidencia + temp_media + umid_media,
  data = treino_rf,
  method = "class",
  cp = 0.01,
  minsplit = 10
)

predicoes_dt <- predict(modelo_dt, teste_rf, type = "class")
acuracia_dt <- sum(predicoes_dt == teste_rf$risco_semana) / nrow(teste_rf)

cat("[OK] Árvore de Decisão treinada\n")
cat("  - Acurácia no conjunto teste:", round(acuracia_dt * 100, 2), "%\n")
cat("  - Profundidade da árvore:", depth(modelo_dt), "\n\n")

# ============================================================================
# PARTE 4: CLUSTERING K-MEANS DE MUNICÍPIOS
# ============================================================================

cat("[INFO] Análise de Clustering (K-Means)...\n\n")

# Preparar dados para clustering
dados_cluster_norm <- dados_clustering %>%
  select(
    municipio_nome,
    casos_normalizados,
    rt_normalizados,
    incidencia_normalizados
  ) %>%
  column_to_rownames("municipio_nome")

# Encontrar número ótimo de clusters (Elbow method)
wss <- vector("numeric", 10)
for (k in 1:10) {
  km <- kmeans(dados_cluster_norm[, -1], centers = k, nstart = 25, iter.max = 50)
  wss[k] <- km$tot.withinss
}

# Usar k = 3 (padrão para risco: baixo, médio, alto)
k_otimo <- 3
modelo_kmeans <- kmeans(
  dados_cluster_norm[, -1], 
  centers = k_otimo, 
  nstart = 25, 
  iter.max = 100
)

# Adicionar labels de cluster
dados_clustering_resultado <- dados_clustering %>%
  mutate(
    cluster = modelo_kmeans$cluster,
    cluster_nome = case_when(
      cluster == 1 ~ "Cluster 1 (Baixo Risco)",
      cluster == 2 ~ "Cluster 2 (Médio Risco)",
      cluster == 3 ~ "Cluster 3 (Alto Risco)",
      TRUE ~ "Desconhecido"
    )
  )

cat("[OK] K-Means clustering realizado\n")
cat("  - Número de clusters:", k_otimo, "\n")
cat("  - Variância intra-cluster:", round(modelo_kmeans$tot.withinss, 2), "\n\n")

cat("Distribuição de municípios por cluster:\n")
print(table(dados_clustering_resultado$cluster_nome))
cat("\n")

# ============================================================================
# PARTE 5: DETECÇÃO DE ANOMALIAS COM ISOLATION FOREST
# ============================================================================

cat("[INFO] Detecção de anomalias com Isolation Forest...\n\n")

# Preparar dados
dados_anomalia <- dados %>%
  group_by(municipio_nome, ano) %>%
  summarise(
    casos_total = sum(casos, na.rm = TRUE),
    rt_media = mean(Rt, na.rm = TRUE),
    incidencia = mean(p_inc100k, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  filter(complete.cases(.))

# Normalizar features para anomalia
dados_anomalia_norm <- dados_anomalia %>%
  mutate(
    casos_norm = normalizar_dados(casos_total),
    rt_norm = normalizar_dados(rt_media),
    incidencia_norm = normalizar_dados(incidencia)
  )

# Isolation Forest
suppressWarnings({
  modelo_if <- isolation.forest(
    dados_anomalia_norm %>% select(casos_norm, rt_norm, incidencia_norm),
    num_trees = 100,
    sample_size = nrow(dados_anomalia_norm)
  )
})

# Detectar anomalias
anomalias_detec <- dados_anomalia_norm %>%
  mutate(
    anomalia_score = rowMeans(modelo_if$anomaly_scores),
    eh_anomalia = anomalia_score > quantile(anomalia_score, 0.95),
    anomalia_tipo = case_when(
      eh_anomalia & rt_norm > 0.7 ~ "Rt Anômalo",
      eh_anomalia & incidencia_norm > 0.7 ~ "Incidência Anômala",
      eh_anomalia ~ "Padrão Anômalo",
      TRUE ~ "Normal"
    )
  )

cat("[OK] Anomalias detectadas\n")
cat("  - Anomalias encontradas:", sum(anomalias_detec$eh_anomalia), "\n\n")

cat("Exemplos de anomalias detectadas:\n")
print(head(filter(anomalias_detec, eh_anomalia), 10))
cat("\n")

# ============================================================================
# PARTE 6: REGRESSÃO PARA PREVISÃO DE INCIDÊNCIA
# ============================================================================

cat("[INFO] Treinando modelos de regressão para previsão de incidência...\n\n")

# Preparar dados para regressão
dados_regressao <- features_dataset %>%
  select(
    casos, rt, incidencia, temp_media, umid_media,
    casos_lag1, variacao_temp, incidencia_futura
  ) %>%
  filter(complete.cases(.))

# Split
set.seed(42)
indice_treino_reg <- createDataPartition(dados_regressao$incidencia_futura, p = 0.8, list = FALSE)
treino_reg <- dados_regressao[indice_treino_reg, ]
teste_reg <- dados_regressao[-indice_treino_reg, ]

# Regressão Linear
modelo_lm <- lm(incidencia_futura ~ ., data = treino_reg)
pred_lm <- predict(modelo_lm, teste_reg)
r2_lm <- 1 - (sum((teste_reg$incidencia_futura - pred_lm)^2) / 
               sum((teste_reg$incidencia_futura - mean(teste_reg$incidencia_futura))^2))

# Regressão Ridge (caret)
modelo_ridge <- train(
  incidencia_futura ~ .,
  data = treino_reg,
  method = "ridge",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = data.frame(lambda = seq(0, 1, 0.1)),
  preProcess = c("center", "scale")
)

pred_ridge <- predict(modelo_ridge, teste_reg)
r2_ridge <- 1 - (sum((teste_reg$incidencia_futura - pred_ridge)^2) / 
                  sum((teste_reg$incidencia_futura - mean(teste_reg$incidencia_futura))^2))

cat("[OK] Modelos de regressão treinados\n")
cat("  - R² Linear:", round(r2_lm, 4), "\n")
cat("  - R² Ridge:", round(r2_ridge, 4), "\n\n")

# ============================================================================
# PARTE 7: EXPORTAÇÃO DE RESULTADOS ML
# ============================================================================

cat("[INFO] Exportando resultados de Machine Learning...\n\n")

# Exportar importância de features
write.csv(importancia_rf, "outputs/data/ml_feature_importance.csv", row.names = FALSE)

# Exportar resultados de clustering
write.csv(dados_clustering_resultado, "outputs/data/ml_clustering_municipios.csv", row.names = FALSE)

# Exportar anomalias detectadas
write.csv(anomalias_detec, "outputs/data/ml_anomalias_detectadas.csv", row.names = FALSE)

# Salvar modelos treinados
saveRDS(list(
  modelo_rf = modelo_rf,
  modelo_dt = modelo_dt,
  modelo_kmeans = modelo_kmeans,
  modelo_lm = modelo_lm,
  modelo_ridge = modelo_ridge,
  features_dataset = features_dataset,
  dados_clustering_resultado = dados_clustering_resultado,
  anomalias_detec = anomalias_detec
), "cache/modelos_ml.rds")

# Criar resumo de modelos
resumo_modelos <- data.frame(
  Modelo = c("Random Forest", "Decision Tree", "K-Means", "Linear Regression", "Ridge Regression"),
  Acuracia_Prec = c(
    round(acuracia_rf * 100, 2),
    round(acuracia_dt * 100, 2),
    NA,
    round(r2_lm, 4) * 100,
    round(r2_ridge, 4) * 100
  ),
  Status = c("Treinado", "Treinado", "Treinado", "Treinado", "Treinado"),
  Tipo = c("Classificação", "Classificação", "Agrupamento", "Regressão", "Regressão")
)

write.csv(resumo_modelos, "outputs/data/ml_resumo_modelos.csv", row.names = FALSE)

cat("[OK] Resultados exportados\n\n")

# ============================================================================
# PARTE 8: EXPLICABILIDADE (SHAP-like)
# ============================================================================

cat("[INFO] Gerando análises de explicabilidade...\n\n")

# Extrair regras da árvore de decisão
explicabilidade_dt <- data.frame(
  Modelo = "Decision Tree",
  Interpretacao = "Árvore interpretável disponível em prplot_dt",
  Profundidade = depth(modelo_dt),
  NumeroFolhas = sum(modelo_dt$frame$var == "<leaf>"),
  Variancia_Explicada = round(sum(modelo_dt$variable.importance) / 
                              sum(colSums(modelo_dt$frame)), 4)
)

cat("Resumo de explicabilidade:\n")
print(explicabilidade_dt)
cat("\n")

# ============================================================================
# RESUMO FINAL DO MÓDULO ML
# ============================================================================

timestamp_fim_ml <- Sys.time()
duracao_ml <- difftime(timestamp_fim_ml, timestamp_inicio_ml, units = "mins")

cat("═" %rep% 80, "\n")
cat("  RESUMO - MÓDULO 2 (MACHINE LEARNING)\n")
cat("═" %rep% 80, "\n")
cat("Tempo total:", round(duracao_ml, 2), "minutos\n")
cat("Modelos treinados: 5\n")
cat("  - Random Forest (Acurácia:", round(acuracia_rf * 100, 2), "%)\n")
cat("  - Decision Tree (Acurácia:", round(acuracia_dt * 100, 2), "%)\n")
cat("  - K-Means (", k_otimo, "clusters )\n")
cat("  - Linear Regression (R²:", round(r2_lm, 4), ")\n")
cat("  - Ridge Regression (R²:", round(r2_ridge, 4), ")\n")
cat("Anomalias detectadas:", sum(anomalias_detec$eh_anomalia), "\n")
cat("Municípios analisados:", nrow(dados_clustering_resultado), "\n")
cat("═" %rep% 80, "\n\n")

# ============================================================================
# FIM DO MÓDULO 2
# ============================================================================
