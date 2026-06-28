# SIPREV v1.2 

**Sistema Inteligente de Previsão Epidemiológica de Dengue**

Plataforma integrada para análise epidemiológica e previsão de dengue em
**Campo Grande/MS**, com expansão massiva orientada à pesquisa em tecnologia
emergente. Esta versão (v1.2) incorpora **130+ seções analíticas**,
**~20.000 linhas de código**, **300+ bibliotecas catalogadas** (100 ML +
100 DL + 100 NN) e dezenas de modelos treinados.

---

## 📦 Arquivos de entrega

| Arquivo | Descrição |
|---|---|
| `SIPREV_Data_Epidemiological_InfoDeng_v1.2.py`   | Script Python autossuficiente (~20k linhas) |
| `SIPREV_Data_Epidemiological_InfoDeng_v1.2.ipynb` | Notebook Jupyter equivalente (149 células) |
| `BAK_SIPREV_Data_Epidemiological_InfoDeng_v1.1.ipynb` | Backup da versão anterior (v1.1) |
| `README.md`                                       | Este documento |

Ambos os artefatos principais (`.py` e `.ipynb`) são **independentes** e
**autossuficientes**: funcionam **localmente** (Python 3.9 a 3.14), no
**Google Colab** e no **Google Cloud Console** via terminal.

| Recurso | `.py` | `.ipynb` |
|---|:---:|:---:|
| Autossuficiente | ✅ | ✅ |
| Google Colab (terminal) | ✅ | ✅ |
| Google Colab (upload direto) | ✅ | ✅ |
| Google Cloud Console | ✅ | ✅ |
| Local (terminal) | ✅ | ✅ |

---

## 🚀 Novidades da v1.2

### 1. 🧠 Inventários massivos de bibliotecas
- **100 bibliotecas Machine Learning** (Seção 100): sklearn, XGBoost, LightGBM,
  CatBoost, statsmodels, Optuna, Hyperopt, SHAP, LIME, MLflow, DVC, AutoGluon,
  PyCaret, H2O, FLAML, e muitas outras
- **100 bibliotecas Deep Learning** (Seção 101): TensorFlow, PyTorch, JAX,
  Keras, PyTorch Lightning, Hugging Face Transformers, Diffusers, DeepSpeed,
  ONNX, Stable Baselines3, OpenCV, Albumentations, Whisper, vLLM, etc.
- **100 bibliotecas Neural Networks** (Seção 102): torch.nn, keras.layers,
  PyTorch Geometric, DGL, Vision Transformer, BERT, GPT-2, T5, RoBERTa,
  FAISS, Captum, Bitsandbytes, FlashAttention, e mais

### 2. 🧬 Modelos RNN/ANN/NLP
- **Seção 103 — RNNs**: Elman, LSTM, GRU, BiLSTM, BiGRU em PyTorch
- **Seção 104 — ANNs**: 6 combinações de ativação (ReLU/GELU/Tanh/SELU/LeakyReLU)
  × otimizador (Adam/AdamW/SGD/RMSprop)
- **Seção 105 — NLP básico**: TF-IDF, frequências, coocorrência via NetworkX
- **Seção 110 — NLP avançado**: Topic Modeling com Latent Dirichlet Allocation (LDA)

### 3. 📈 Modelagem preditiva, prevenção, comparação
- **Seção 106 — Multi-horizonte**: previsões 1, 4, 8 e 12 semanas + ensemble
- **Seção 107 — Prevenção**: ranking de municípios com score composto e classes
  (Crítica/Alta/Média/Baixa/Vigilância)
- **Seção 108 — Comparação final**: benchmark cross-paradigma de todos os modelos
- **Seção 121 — Auto-ML**: RandomizedSearchCV em HistGBM/RF/ExtraTrees/LightGBM

### 4. 📥 Downloader robusto com barra de progresso (Seção 99)
Se os CSVs não estiverem localmente, baixa automaticamente do repositório
oficial **com barra de progresso inline**, registrando início, fim, tamanho,
URL e caminho local de cada arquivo (compatível com Local, Colab, Cloud Console).

### 5. 🔬 Análises avançadas
- **Bayesiana** com bootstrap de 10.000 reamostras (Seção 115)
- **Suite de testes estatísticos**: Mann-Whitney, Wilcoxon, Kruskal-Wallis,
  Kolmogorov-Smirnov, Anderson-Darling, Levene, Spearman, Kendall (Seção 116)
- **Benchmark de inferência**: latência por modelo, throughput (Seção 117)
- **Análise espectral de Fourier** (Seção 128)
- **Análise multivariada**: PCA + K-Means das capitais (Seção 127)
- **Equidade regional**: Gini, curva de Lorenz (Seção 124)

### 6. 📝 Documentação publication-ready
- **Manuscrito auto-gerado** (Seção 113)
- **Recomendações para gestores** em 4 horizontes temporais (Seção 125)
- **Relatório narrativo integrado** publication-ready (Seção 131)

---

## ▶️ Como executar

### Localmente (terminal)
```bash
cd C:\Users\Workstation\Desktop\wokspace_módulo_4\Project_4\update
python SIPREV_Data_Epidemiological_InfoDeng_v1.2.py
```

O programa:
1. Localiza automaticamente os CSVs em `../dataset/csv_archive/` ou `../input/csv_archive/`
2. Se não encontrar, **baixa do repositório oficial** com barra de progresso
3. Executa todas as 130+ seções
4. Gera o `.zip` final com timestamp

### Localmente (notebook)
Abra o `.ipynb` no Jupyter Lab, VS Code ou Jupyter Notebook e use **Run All**.

### Google Colab
1. Faça upload do `.py` **ou** do `.ipynb`
2. **Run All** — a Seção 0 instala dependências em modo seguro (sem `--upgrade`)
3. A Seção 99 baixa os CSVs automaticamente

### Google Cloud Console
Mesmo procedimento do local; o sistema detecta automaticamente o ambiente.

---

## 🐍 Compatibilidade Python

Testado em:
- **Python 3.12** (Anaconda) — recomendado, mais bibliotecas
- **Python 3.14** — também recomendado, mais bibliotecas
- **Python 3.13** — funciona (algumas libs opcionais podem faltar)

Todas as seções têm **fallback gracioso** quando uma biblioteca está ausente:
o pipeline continua e a seção é simplesmente pulada com aviso no log.

---

## 📜 Saídas e exportações

Tudo gravado **inline durante a execução** (na ordem do pipeline), organizado em:

| Pasta | Conteúdo |
|---|---|
| `graficos/` | Gráficos `.png` (séries, barras, boxplots, heatmaps, periodogramas) |
| `mapas/` | Mapas interativos `.html` (Folium: calor, marcadores, hotspots) |
| `redes/` | Redes de coocorrência: `.png`, `.html`, `.graphml`, `.csv`, `.xlsx` |
| `dashboards/` | Dashboards interativos `.html` (Plotly) |
| `relatorios/` | Relatórios `.txt` e `.md` via **Texttable** |
| `logs/` | Tabelas `.log` e log de execução |
| `dados/` | `.csv`, `.xlsx`, `.json`, `.parquet`, inventários, checklist |
| `modelos/` | Métricas, rankings, previsões, modelos persistidos |
| `pdf/` | Relatórios `.pdf` (FPDF), bundle unificado |

Ao final, a Seção 29 compacta **tudo** em
`EpiAnalysis_DENG_<timestamp>.zip` (no Colab, com download automático).

---

## 📊 Modelos treinados

A execução típica registra **30+ modelos** entre todos os paradigmas:

| Paradigma | Modelos |
|---|---|
| Machine Learning | HistGBM, RF, ExtraTrees, GradientBoosting, Ridge, Huber, XGBoost, LightGBM, CatBoost, Voting, Stacking |
| Modelos de Contagem | GLM Poisson, Binomial Negativa, Linear Ref |
| Deep Learning (PyTorch) | LSTM, GRU, TCN |
| Neural Networks (PyTorch) | MLP, CNN-1D, Autoencoder |
| RNNs (PyTorch) | Elman, LSTM, GRU, BiLSTM, BiGRU |
| ANNs (PyTorch) | 6 combinações ativação × otimizador |
| Multi-horizonte | HistGBM, RF, ExtraTrees, LightGBM, XGBoost × 4 horizontes |
| Auto-ML | RandomizedSearchCV(HistGBM/RF/ExtraTrees/LightGBM) |
| Séries Temporais | ARIMA, SARIMA, Prophet, ETS |

---

## 🧠 Camadas de Inteligência Computacional

### 1. Machine Learning (100 bibliotecas)
RandomForest, ExtraTrees, **HistGradientBoosting**, **XGBoost**, **LightGBM**,
**CatBoost**, Ridge/Huber, ensembles **Voting** e **Stacking**, e modelos de
contagem **GLM Poisson** e **Binomial Negativa**.

### 2. Deep Learning (100 bibliotecas)
**PyTorch**: LSTM empilhada, **GRU** bidirecional e **TCN** (Temporal
Convolutional Network). Quando o **TensorFlow** está disponível, as seções-base
de LSTM/GRU/Transformer e Autoencoder Keras também são executadas.

### 3. Neural Networks (100 bibliotecas)
**PyTorch**: **MLP profundo** (BatchNorm + Dropout) sobre features tabulares,
**CNN-1D** sobre janelas da série e **Autoencoder** para detecção de anomalias.

### 4. RNNs / ANNs / NLP
- **RNNs**: 5 arquiteturas recorrentes comparadas (Elman, LSTM, GRU, BiLSTM, BiGRU)
- **ANNs**: MLPs profundos com 6 combinações ativação × otimizador
- **NLP**: TF-IDF, coocorrência via NetworkX, LDA topic modeling, opcional
  HuggingFace tokenizer

### 5. Redes de Coocorrência (NetworkX)
- Municípios de MS em alerta
- Capitais brasileiras em sincronia
- Variáveis clima × epidemiologia
- Evolução temporal anual
- Concordância entre modelos

---

## 🗂️ Estrutura esperada do projeto

```
Project_4/
├── dataset/
│   └── csv_archive/
│       ├── DENGCAPBR_16_25.csv     ← Capitais brasileiras
│       ├── DENGCG-MS_16_25.csv     ← Campo Grande/MS
│       └── DENGMS-BR_16_25.csv     ← Municípios de MS
└── update/
    ├── SIPREV_Data_Epidemiological_InfoDeng_v1.2.py
    ├── SIPREV_Data_Epidemiological_InfoDeng_v1.2.ipynb
    ├── BAK_SIPREV_Data_Epidemiological_InfoDeng_v1.1.ipynb
    └── README.md
```

Se a pasta `dataset/csv_archive/` não existir ou estiver vazia, o sistema
baixa os CSVs do repositório oficial automaticamente.

---

## 📋 Mapa das seções novas (99–131)

| Nº | Seção | Categoria |
|---:|---|---|
| 99  | Downloader robusto de CSVs | Dados |
| 100 | Inventário 100 bibliotecas ML | Documentação |
| 101 | Inventário 100 bibliotecas DL | Documentação |
| 102 | Inventário 100 bibliotecas NN | Documentação |
| 103 | RNNs (Elman, LSTM, GRU, BiLSTM, BiGRU) | Deep Learning |
| 104 | ANNs (6 combinações ativação × otimizador) | Neural Networks |
| 105 | NLP básico (TF-IDF, coocorrência) | NLP |
| 106 | Modelagem preditiva multi-horizonte | Predição |
| 107 | Modelos de prevenção (ranking) | Prevenção |
| 108 | Comparação final cross-paradigma | Benchmark |
| 109 | Manipulação e processamento avançado | Wrangling |
| 110 | NLP avançado: Topic Modeling LDA | NLP |
| 111 | Análise de sensibilidade | Robustez |
| 112 | Score composto de risco operacional | Vigilância |
| 113 | Manuscrito auto-gerado | Documentação |
| 114 | Sumário executivo v1.2 | Síntese |
| 115 | Análise bayesiana via bootstrap | Estatística |
| 116 | Suite de testes estatísticos | Estatística |
| 117 | Benchmark de tempo de inferência | Performance |
| 118 | Conclusões da pesquisa | Documentação |
| 119 | Dinâmica ambiental | Clima |
| 120 | Análise interanual por estação | Temporal |
| 121 | Auto-ML simplificado | AutoML |
| 122 | Bundle final de relatórios PDF | Documentação |
| 123 | Comparação estadual cruzada (capitais) | Comparativo |
| 124 | Avaliação de equidade regional (Gini) | Equidade |
| 125 | Recomendações para gestores | Vigilância |
| 126 | Checklist de entrega + auditoria | Auditoria |
| 127 | Análise multivariada (PCA + K-Means) | Multivariada |
| 128 | Análise espectral de Fourier | Temporal |
| 129 | Análise de cohort temporal | Temporal |
| 130 | Tabela mestra de execução | Síntese |
| 131 | Relatório narrativo integrado | Publication |

---

## 🔬 Para o artigo de pesquisa em tecnologia emergente

A v1.2 foi desenhada como **caso de uso prático** para um artigo científico
sobre tecnologia emergente em vigilância em saúde pública. Os materiais
diretamente relevantes para o artigo:

1. **Inventário de 300+ bibliotecas** — referência para análises comparativas
2. **Manuscrito auto-gerado** (Seção 113) — rascunho de data brief
3. **Conclusões da pesquisa** (Seção 118) — takeaways estruturados
4. **Relatório narrativo integrado** (Seção 131) — texto publication-ready
   com resumo, métodos, resultados, discussão, limitações e direções futuras
5. **Benchmark cross-paradigma** (Seção 108) — comparação justa entre ML/DL/NN/RNN
6. **Tabela mestra** (Seção 130) — síntese absoluta para anexo do artigo

---

## ⚙️ Notas técnicas

- **Section 0 (Colab-safe)**: instala apenas dependências leves ausentes, sem
  `--upgrade`, evitando o restart de runtime que travava versões anteriores
- **Tolerância a falhas**: cada seção em `try/except`; pipeline continua se uma falhar
- **Reprodutibilidade**: sementes fixas (42) em sklearn/XGBoost/LightGBM/CatBoost/PyTorch
- **Compatibilidade**: Python 3.9 a 3.14, com fallback gracioso para libs opcionais

---

## 🔗 Fontes de dados

- **InfoDengue** — <https://info.dengue.mat.br> · API: `/api/alertcity/`
- **Repositório oficial dos CSVs** (auto-download pela Seção 99):
  <https://github.com/OpenScienceTechnology/info_dengue/tree/main/Dataset/Dengue/csv_archive>
- **Campo Grande/MS** — código IBGE **5002704**

---

_Disciplina: Análise Organizacional e Soluções Tecnológicas — Ciência dos Dados_
_Módulo 4 — Previsão Epidemiológica de Dengue (versão v1.2)_
_Pesquisa em Tecnologia Emergente_
