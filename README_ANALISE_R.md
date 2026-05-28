# Análise de Dados de Dengue em R

Este diretório contém dois programas em **Linguagem R** para análise dos dados epidemiológicos de dengue do repositório Info Dengue.

## 📋 Arquivos

### 1. `dengue_analysis.R` - Análise Completa
Script completo com análise exploratória avançada, incluindo:
- Carregamento e combinação de múltiplos arquivos CSV
- Estatísticas descritivas detalhadas
- Análise por município
- Análise temporal com série histórica
- Geração de 4 visualizações em gráficos
- Correlação entre temperatura e casos
- Relatório executivo
- Exportação de dados processados

**Requisitos:**
- R 3.6+ 
- Pacotes: `dplyr`, `ggplot2`, `tidyr`, `readr`, `lubridate`, `gridExtra`

**Como executar:**
```bash
Rscript dengue_analysis.R
```

**Saída gerada:**
- `outputs/01_casos_por_ano_mes.png` - Gráfico de distribuição de casos
- `outputs/02_evolucao_rt.png` - Série temporal de Rt
- `outputs/03_top10_municipios.png` - Top 10 municípios
- `outputs/04_temperatura_vs_casos.png` - Correlação temperatura-casos
- `outputs/resumo_municipal.csv` - Dados agregados por município
- `outputs/resumo_temporal.csv` - Dados agregados por período

### 2. `dengue_analysis_simple.R` - Análise Simplificada
Script enxuto com análise básica, ideal para execução rápida:
- Carregamento dos dados
- Estatísticas gerais (total casos, municípios, período)
- Top 10 municípios
- Análise temporal por mês
- Estatísticas descritivas rápidas

**Requisitos:**
- R 3.6+
- Pacotes: `dplyr`, `readr`

**Como executar:**
```bash
Rscript dengue_analysis_simple.R
```

## 🔧 Instalação de Dependências

No R ou RStudio, execute:

```r
# Para o script completo
install.packages(c("dplyr", "ggplot2", "tidyr", "readr", "lubridate", "gridExtra"))

# Para o script simplificado
install.packages(c("dplyr", "readr"))
```

Ou instale tudo de uma vez:
```r
install.packages(c("dplyr", "ggplot2", "tidyr", "readr", "lubridate", "gridExtra"))
```

## 📊 O que os Scripts Analisam

### Variáveis Principais:
- **casos / casos_est**: Número de casos estimados/confirmados de dengue
- **Rt**: Número reprodutivo (mede a transmissibilidade)
- **Temperatura**: Variáveis climáticas (tempmin, tempmax, tempmed)
- **Umidade**: Fatores ambientais (umidmax, umidmin, umidmed)
- **p_inc100k**: Incidência por 100 mil habitantes
- **municipio_nome**: Localização geográfica

### Análises Realizadas:

1. **Exploração Inicial**: Estrutura, dimensões e tipos de dados
2. **Descritiva**: Média, mediana, desvio padrão, mínimo, máximo
3. **Geográfica**: Ranking de municípios mais afetados
4. **Temporal**: Evolução ao longo do tempo
5. **Correlação**: Relação entre fatores climáticos e transmissão
6. **Epidemiológica**: Classificação de situação (crítica, alerta, controle)

## 📈 Interpretação dos Resultados

### Número Reprodutivo (Rt):
- **Rt > 1.5**: Epidemia acelerada (crítica)
- **1.0 < Rt ≤ 1.5**: Epidemia em expansão (alerta)
- **0.5 < Rt ≤ 1.0**: Epidemia em declínio (controle)
- **Rt ≤ 0.5**: Sem transmissão (controlada)

### Incidência (casos/100k):
- **Baixa**: < 100 casos/100k
- **Média**: 100-300 casos/100k
- **Alta**: > 300 casos/100k

## 🚀 Como Usar no RStudio

1. Abra RStudio
2. Defina o diretório de trabalho para a raiz do repositório
3. Abra o arquivo desejado (dengue_analysis.R ou dengue_analysis_simple.R)
4. Clique em "Source" ou pressione Ctrl+Shift+S
5. Verifique a saída no console
6. Os gráficos serão salvos em `outputs/`

## 📝 Dados de Origem

Os dados utilizados são do repositório **ArchiZikDen** (Info Dengue):
- Fonte: API pública do InfoDengue (FIOCRUZ)
- Cobertura: Municípios brasileiros
- Período: Incluso no arquivo
- Formato: CSV

### Estrutura dos Dados:
- **data_iniSE**: Timestamp do início da semana epidemiológica
- **SE**: Semana epidemiológica
- **casos_est**: Casos estimados
- **Rt**: Número reprodutivo
- **municipio_nome**: Nome do município
- **pop**: População do município
- Variáveis climáticas: temp (min/máx/média), umidade (min/máx/média)

## 🔍 Exemplos de Uso Avançado

### Análise específica de um município:
```r
dengue_filtered <- dengue %>% 
  filter(municipio_nome == "Rio Branco")

summary(dengue_filtered$casos)
```

### Criar seu próprio gráfico:
```r
ggplot(dengue, aes(x = data, y = casos)) +
  geom_line() +
  facet_wrap(~municipio_nome)
```

### Exportar dados de um período:
```r
period_data <- dengue %>%
  filter(ano >= 2025)

write_csv(period_data, "dengue_2025.csv")
```

## ⚙️ Requisitos do Sistema

- **OS**: Linux, macOS ou Windows
- **R**: versão 3.6 ou superior
- **Memória**: Mínimo 2GB (recomendado 4GB+)
- **Espaço**: ~50MB para dados + saídas

## 🐛 Troubleshooting

**Erro: "could not find function"**
- Instale os pacotes faltando: `install.packages("nome_do_pacote")`

**Erro: "cannot open file 'Dataset/Dengue/...'**
- Verifique se está no diretório correto: `getwd()`
- Use o caminho absoluto se necessário

**Lentidão na execução**
- Reduza o período de análise
- Use o script simplificado em vez do completo

## 📚 Referências

- [InfoDengue - FIOCRUZ](https://www.infodengue.ufmg.br/)
- [Documentação dplyr](https://dplyr.tidyverse.org/)
- [ggplot2 - Visualização em R](https://ggplot2.tidyverse.org/)
- [IBGE - Dados Municipais](https://www.ibge.gov.br/)

---

**Versão**: 1.0  
**Última atualização**: 2026-05-28  
**Autor**: Data Analysis Scripts
