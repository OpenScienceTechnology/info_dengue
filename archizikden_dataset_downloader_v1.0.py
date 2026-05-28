#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ArchiZikDen Dataset Downloader —  Cobertura Nacional Completa              ║
║   Version                       :  v1.0                                      ║
║   Author                        :  VIANA                                     ║
║   ArchiZikDen Dataset Downloader:  info.dengue.mat.br                        ║
║   Arboviroses                   : Dengue · Chikungunya · Zika                ║
║   Cobertura Nacional Completa   : via API IBGE  (~5.570 municípios)          ║
║   Ciência de Dados              : Epidemiologia Computacional                ║
╚══════════════════════════════════════════════════════════════════════════════╝


📁 Arbovirose
├── [1]  🦠  Chikungunya
├── [2]  🦟  Dengue
└── [3]  🧬  Zika

Dependências:
    pip install requests tqdm texttable pandas

Uso:
    archizikden_dataset_downloader_v1.0.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import csv
import json
import time
import logging
import requests
import datetime
import unicodedata
import platform
import socket
from pathlib import Path
from typing import Optional
 
try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False
 
try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False
 
try:
    from texttable import Texttable
    TEXTTABLE_OK = True
except ImportError:
    TEXTTABLE_OK = False
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBAIS
# ─────────────────────────────────────────────────────────────────────────────
INFODENGUE_URL  = "https://info.dengue.mat.br/api/alertcity"
IBGE_MUNIC_URL  = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
SCRIPT_DIR      = Path(__file__).resolve().parent
CACHE_FILE      = SCRIPT_DIR / "ibge_municipios_cache.json"
CACHE_MAX_DAYS  = 30
API_DELAY       = 0.08
API_RETRIES     = 3
API_TIMEOUT     = 30
 
# ── Arboviroses disponíveis ──────────────────────────────────────────────────
# ATENÇÃO: a ordem aqui define os números exibidos no menu interativo.
# [1] Chikungunya · [2] Dengue · [3] Zika  (ordem alfabética em português)
DISEASES = {
    "1": {"id": "chikungunya",  "label": "🦠 Chikungunya",  "emoji": "🦠"},
    "2": {"id": "dengue",       "label": "🦟 Dengue",       "emoji": "🦟"},
    "3": {"id": "zika",         "label": "🧬 Zika",         "emoji": "🧬"},
}
 
# ── Capitais brasileiras (geocode IBGE oficial) ───────────────────────────────
CAPITAIS: dict[str, dict] = {
    "AC": {"nome": "Rio Branco",       "geocode": 1200401},
    "AL": {"nome": "Maceió",           "geocode": 2704302},
    "AM": {"nome": "Manaus",           "geocode": 1302603},
    "AP": {"nome": "Macapá",           "geocode": 1600303},
    "BA": {"nome": "Salvador",         "geocode": 2927408},
    "CE": {"nome": "Fortaleza",        "geocode": 2304400},
    "DF": {"nome": "Brasília",         "geocode": 5300108},
    "ES": {"nome": "Vitória",          "geocode": 3205309},
    "GO": {"nome": "Goiânia",          "geocode": 5208707},
    "MA": {"nome": "São Luís",         "geocode": 2111300},
    "MG": {"nome": "Belo Horizonte",   "geocode": 3106200},
    "MS": {"nome": "Campo Grande",     "geocode": 5002704},
    "MT": {"nome": "Cuiabá",           "geocode": 5103403},
    "PA": {"nome": "Belém",            "geocode": 1501402},
    "PB": {"nome": "João Pessoa",      "geocode": 2507507},
    "PE": {"nome": "Recife",           "geocode": 2611606},
    "PI": {"nome": "Teresina",         "geocode": 2211001},
    "PR": {"nome": "Curitiba",         "geocode": 4106902},
    "RJ": {"nome": "Rio de Janeiro",   "geocode": 3304557},
    "RN": {"nome": "Natal",            "geocode": 2408102},
    "RO": {"nome": "Porto Velho",      "geocode": 1100205},
    "RR": {"nome": "Boa Vista",        "geocode": 1400100},
    "RS": {"nome": "Porto Alegre",     "geocode": 4314902},
    "SC": {"nome": "Florianópolis",    "geocode": 4205407},
    "SE": {"nome": "Aracaju",          "geocode": 2800308},
    "SP": {"nome": "São Paulo",        "geocode": 3550308},
    "TO": {"nome": "Palmas",           "geocode": 1721000},
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CORES ANSI
# ─────────────────────────────────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"
 
def ok(msg):    print(f"{GREEN}  ✔  {msg}{RESET}")
def err(msg):   print(f"{RED}  ✘  {msg}{RESET}")
def info(msg):  print(f"{CYAN}  ℹ  {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}  ⚠  {msg}{RESET}")
def step(msg):  print(f"{MAGENTA}  ►  {msg}{RESET}")
 
def menu_title(title: str, color: str = YELLOW) -> None:
    print(f"\n{color}{BOLD}{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}{RESET}")
 
def section(title: str) -> None:
    print(f"\n{BLUE}{BOLD}  ── {title}{RESET}")
 
def banner() -> None:
    print(f"""{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║   🦟  InfoDengue Downloader  ·  info.dengue.mat.br                         ║
║   Arboviroses: Dengue · Chikungunya · Zika                                 ║
║   Cobertura Nacional Completa · ~5.570 municípios brasileiros              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{RESET}""")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TEXTTABLE  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _tt(rows: list, headers: list,
        col_align: Optional[list] = None,
        col_dtype: Optional[list] = None,
        width: int = 110) -> str:
    if not TEXTTABLE_OK:
        lines = ["\t".join(map(str, headers))]
        lines += ["\t".join(map(str, r)) for r in rows]
        return "\n".join(lines)
    t = Texttable(max_width=width)
    t.set_deco(Texttable.HEADER | Texttable.BORDER | Texttable.VLINES)
    if col_align:
        t.set_cols_align(col_align)
    if col_dtype:
        t.set_cols_dtype(col_dtype)
    t.header(headers)
    for row in rows:
        t.add_row(row)
    return t.draw()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LOGGING DINÂMICO POR FORMATO E TIMESTAMP
# ─────────────────────────────────────────────────────────────────────────────
def _make_log_path(fmt: str, ts: datetime.datetime) -> Path:
    """
    Gera o caminho do log conforme o formato e timestamp:
      infodengue_csv_20240315_143022.log
      infodengue_json_20240315_143022.log
    """
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / f"infodengue_{fmt.lower()}_{ts_str}.log"
 
 
def write_log_entry(
    log_path: Path,
    disease: str,
    scope: str,
    date_filter: str,
    output_fmt: str,
    output_path: str,
    start_ts: datetime.datetime,
    end_ts: datetime.datetime,
    total_records: int,
    total_requests: int,
    errors: int,
    ibge_source: str,
    geocodes_sample: list[int],
    stats: Optional[dict] = None,
) -> None:
    elapsed  = (end_ts - start_ts).total_seconds()
    success  = total_requests - errors
    taxa_ok  = 100 * success / max(total_requests, 1)
 
    # ── Seção 1: Identificação da Operação ───────────────────────────────────
    id_rows = [
        ["Script",              "infodengue_downloader.py"],
        ["Versão",              "3.0.0 — Cobertura Nacional Completa"],
        ["Host",                socket.gethostname()],
        ["SO / Python",         f"{platform.system()} {platform.release()} / Python {platform.python_version()}"],
        ["Arbovirose",          disease.upper()],
        ["Arquivo de log",      log_path.name],
    ]
 
    # ── Seção 2: Parâmetros da Consulta ──────────────────────────────────────
    param_rows = [
        ["Escopo geográfico",   scope],
        ["Filtro temporal",     date_filter],
        ["Formato de saída",    output_fmt.upper()],
        ["Arquivo gerado",      output_path],
        ["Fonte geocodes IBGE", ibge_source],
        ["Total geocodes",      f"{total_requests:,}"],
        ["Amostra geocodes",    str(geocodes_sample[:5]) + (" …" if len(geocodes_sample) > 5 else "")],
    ]
 
    # ── Seção 3: Temporalidade ────────────────────────────────────────────────
    time_rows = [
        ["Data início",         start_ts.strftime("%d/%m/%Y")],
        ["Hora início",         start_ts.strftime("%H:%M:%S")],
        ["Data fim",            end_ts.strftime("%d/%m/%Y")],
        ["Hora fim",            end_ts.strftime("%H:%M:%S")],
        ["Tempo total (s)",     f"{elapsed:.2f}"],
        ["Tempo médio/req (s)", f"{elapsed/max(total_requests,1):.4f}"],
    ]
 
    # ── Seção 4: Resultado do Download ───────────────────────────────────────
    result_rows = [
        ["Requisições enviadas",  f"{total_requests:,}"],
        ["Respostas com dados",   f"{success:,}"],
        ["Sem dados / erros",     f"{errors:,}"],
        ["Taxa de sucesso (%)",   f"{taxa_ok:.2f}"],
        ["Registros baixados",    f"{total_records:,}"],
    ]
 
    # ── Seção 5: Estatísticas dos dados (se disponíveis) ─────────────────────
    stats_table = ""
    if stats:
        stat_rows = [[k, v] for k, v in stats.items()]
        stats_table = (
            "\n  ── Estatísticas dos Dados ──\n\n"
            + _tt(stat_rows, ["Indicador", "Valor"],
                  col_align=["l", "r"], width=80)
        )
 
    sep = "═" * 110
    entry = (
        f"\n{sep}\n"
        f"  📅  OPERAÇÃO InfoDengue  ·  {end_ts.strftime('%d/%m/%Y  %H:%M:%S')}\n"
        f"{sep}\n\n"
        f"  ── Identificação ──\n\n{_tt(id_rows,     ['Campo','Valor'],col_align=['l','l'],width=100)}\n\n"
        f"  ── Parâmetros da Consulta ──\n\n{_tt(param_rows,  ['Parâmetro','Valor'],col_align=['l','l'],width=100)}\n\n"
        f"  ── Temporalidade ──\n\n{_tt(time_rows,   ['Campo','Valor'],col_align=['l','l'],width=80)}\n\n"
        f"  ── Resultado do Download ──\n\n{_tt(result_rows, ['Métrica','Valor'],col_align=['l','r'],width=80)}\n"
        f"{stats_table}\n"
    )
 
    # Grava no arquivo de log específico desta execução
    logger = logging.getLogger(f"infodengue.{log_path.stem}")
    if not logger.handlers:
        handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
 
    logger.info(entry)
    ok(f"Log gravado → {log_path.name}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# IBGE — CARREGAMENTO COMPLETO DE MUNICÍPIOS  (~5.570)
# ─────────────────────────────────────────────────────────────────────────────
def _cache_valid() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = (datetime.datetime.now() -
           datetime.datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)).days
    return age < CACHE_MAX_DAYS
 
 
def _fetch_ibge_api() -> list[dict]:
    step("Conectando à API IBGE Localidades…")
    try:
        resp = requests.get(IBGE_MUNIC_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        ok(f"{len(data):,} municípios recebidos da API IBGE.")
        return data
    except Exception as e:
        warn(f"API IBGE indisponível: {e}")
        return []
 
 
def _safe_get(obj, *keys, default=None):
    """Navegação segura em dicionários aninhados, tolerando None em qualquer nível."""
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur if cur is not None else default
 
 
def _parse_uf_map(raw: list[dict]) -> dict[str, list[dict]]:
    """
    Constrói o mapa UF → lista de municípios a partir da resposta bruta da
    API IBGE. Tolera campos None / ausentes em qualquer nível da hierarquia:
      município → microrregiao → mesorregiao → UF → sigla
    Municípios sem UF identificável são agrupados em '??'.
    """
    uf_map: dict[str, list[dict]] = {}
    for m in raw:
        try:
            # Caminhos alternativos que a API IBGE pode devolver
            sigla = (
                _safe_get(m, "microrregiao", "mesorregiao", "UF", "sigla")
                or _safe_get(m, "mesorregiao", "UF", "sigla")
                or _safe_get(m, "regiao-imediata", "regiao-intermediaria", "UF", "sigla")
                or "??"
            )
            geocode = int(m["id"])
            nome    = str(m.get("nome", f"id:{geocode}"))
            uf_map.setdefault(sigla, []).append(
                {"id": geocode, "nome": nome, "uf": sigla}
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            # Registro malformado — ignora silenciosamente
            continue
    return uf_map
 
 
def load_ibge_municipios() -> tuple[dict[str, list[dict]], str]:
    raw: list[dict] = []
    fonte = "desconhecida"
 
    if _cache_valid():
        step(f"Cache IBGE válido encontrado → {CACHE_FILE.name}")
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                raw = json.load(f)
            ok(f"{len(raw):,} municípios carregados do cache.")
            fonte = f"cache local · {CACHE_FILE.name}"
        except Exception:
            raw = []
 
    if not raw:
        raw = _fetch_ibge_api()
        if raw:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False)
            ok(f"Cache salvo → {CACHE_FILE.name}")
            fonte = "API IBGE (baixado agora)"
 
    if not raw:
        warn("Usando lista de fallback embutida (capitais + principais cidades).")
        raw = _fallback_raw()
        fonte = "fallback embutido (sem internet)"
 
    return _parse_uf_map(raw), fonte
 
 
def _fallback_raw() -> list[dict]:
    """Fallback mínimo: todas as capitais + cidades principais por UF."""
    entries = [
        # Capital de cada estado (todas garantidas)
        (1200401,"Rio Branco","AC",12),    (1302603,"Manaus","AM",13),
        (1600303,"Macapá","AP",16),        (1501402,"Belém","PA",15),
        (2111300,"São Luís","MA",21),      (2211001,"Teresina","PI",22),
        (2304400,"Fortaleza","CE",23),     (2408102,"Natal","RN",24),
        (2507507,"João Pessoa","PB",25),   (2611606,"Recife","PE",26),
        (2800308,"Aracaju","SE",28),       (2927408,"Salvador","BA",29),
        (3106200,"Belo Horizonte","MG",31),(3304557,"Rio de Janeiro","RJ",33),
        (3550308,"São Paulo","SP",35),     (4106902,"Curitiba","PR",41),
        (4205407,"Florianópolis","SC",42), (4314902,"Porto Alegre","RS",43),
        (5002704,"Campo Grande","MS",50),  (5103403,"Cuiabá","MT",51),
        (5208707,"Goiânia","GO",52),       (5300108,"Brasília","DF",53),
        (1100205,"Porto Velho","RO",11),   (1400100,"Boa Vista","RR",14),
        (1721000,"Palmas","TO",17),        (2704302,"Maceió","AL",27),
        (3205309,"Vitória","ES",32),       (2704302,"Maceió","AL",27),
        # Cidades adicionais relevantes
        (3518800,"Guarulhos","SP",35),     (3509502,"Campinas","SP",35),
        (3548708,"Ribeirão Preto","SP",35),(3170206,"Uberlândia","MG",31),
        (2910800,"Feira de Santana","BA",29),(2604106,"Caruaru","PE",26),
        (4115200,"Londrina","PR",41),      (4209102,"Joinville","SC",42),
        (4304606,"Caxias do Sul","RS",43), (5201405,"Aparecida de Goiânia","GO",52),
        (1301902,"Parintins","AM",13),     (1500602,"Ananindeua","PA",15),
        (2112001,"Timon","MA",21),         (2408003,"Mossoró","RN",24),
        (2905701,"Camaçari","BA",29),      (3156700,"Uberaba","MG",31),
        (3301702,"Duque de Caxias","RJ",33),(3303500,"Nova Iguaçu","RJ",33),
    ]
    result = []
    seen = set()
    for geocode, nome, uf_sigla, uf_id in entries:
        if geocode in seen:
            continue
        seen.add(geocode)
        result.append({
            "id": geocode, "nome": nome,
            "microrregiao": {"mesorregiao": {"UF": {"sigla": uf_sigla, "id": uf_id}}}
        })
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# BUSCA POR NOME
# ─────────────────────────────────────────────────────────────────────────────
def _norm(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )
 
def search_municipio(uf_map: dict, query: str) -> list[dict]:
    q = _norm(query)
    return sorted(
        [m for mlist in uf_map.values() for m in mlist if q in _norm(m["nome"])],
        key=lambda x: _norm(x["nome"])
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MENU — ARBOVIROSE
# ─────────────────────────────────────────────────────────────────────────────
def ask_disease() -> tuple[str, str]:
    """
    Exibe o menu de arboviroses gerado dinamicamente a partir de DISEASES.
    Retorna (disease_id, disease_label).
    O menu é sempre consistente com o dicionário — sem risco de dessincronização.
    """
    menu_title("PASSO 1 / 5  —  🦟 ARBOVIROSE")
 
    # ── Gera as linhas do menu diretamente do dicionário ─────────────────────
    keys   = sorted(DISEASES.keys())          # ["1", "2", "3"]
    last   = keys[-1]
    print(f"\n  {YELLOW}📁 Arbovirose{RESET}")
    for k in keys:
        d      = DISEASES[k]
        prefix = "└──" if k == last else "├──"
        print(f"  {prefix} [{k}]  {d['emoji']}  {d['label'].split(' ', 1)[1]}")
    print()
 
    choice = input("  Selecione a arbovirose (1-3): ").strip()
    if choice not in DISEASES:
        err(f"Opção '{choice}' inválida. Escolha entre: {', '.join(keys)}")
        sys.exit(1)
 
    d = DISEASES[choice]
    ok(f"Arbovirose selecionada: [{choice}] {d['label']}")
    return d["id"], d["label"]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MENU — ESCOPO GEOGRÁFICO
# ─────────────────────────────────────────────────────────────────────────────
def ask_scope(uf_map: dict[str, list[dict]]) -> tuple[str, list[int]]:
    all_geocodes = [m["id"] for mlist in uf_map.values() for m in mlist]
    total_mun    = len(all_geocodes)
 
    menu_title("PASSO 2 / 5  —  🗺  ESCOPO GEOGRÁFICO")
    print(f"""
  {YELLOW}📁 Base de dados{RESET}
  ├── [1]  🌎  Nacional      ({total_mun:,} municípios)
  ├── [2]  🗺   Estadual      (selecione a UF)
  ├── [3]  🏛️   Capitais      (27 capitais estaduais)
  └── [4]  📍  Municipal     (código IBGE ou nome)
""")
    choice = input("  Escolha (1-4): ").strip()
 
    # ── Nacional ─────────────────────────────────────────────────────────────
    if choice == "1":
        _show_uf_summary(uf_map)
        confirm = input(
            f"\n  ⚠️  Serão processados {total_mun:,} municípios. "
            "Isso pode levar muito tempo.\n"
            "  Confirma? (s/N): "
        ).strip().lower()
        if confirm != "s":
            err("Operação cancelada.")
            sys.exit(0)
        return "Nacional", all_geocodes
 
    # ── Estadual ─────────────────────────────────────────────────────────────
    elif choice == "2":
        _show_uf_summary(uf_map)
        uf_in = input("\n  Digite a sigla da UF (ex: SP): ").strip().upper()
        if uf_in not in uf_map:
            err(f"UF '{uf_in}' não encontrada.")
            sys.exit(1)
        municipios = uf_map[uf_in]
        ok(f"{len(municipios):,} municípios encontrados para {uf_in}.")
        return f"Estadual — {uf_in}", [m["id"] for m in municipios]
 
    # ── Capitais ─────────────────────────────────────────────────────────────
    elif choice == "3":
        rows = [[uf, cap["nome"], cap["geocode"]]
                for uf, cap in sorted(CAPITAIS.items())]
        print()
        print(_tt(rows, ["UF", "Capital", "Geocode IBGE"],
                  col_align=["c", "l", "r"], width=60))
        confirm = input(
            f"\n  Baixar dados das 27 capitais? (S/n): "
        ).strip().lower()
        if confirm == "n":
            err("Operação cancelada.")
            sys.exit(0)
        geocodes = [v["geocode"] for v in CAPITAIS.values()]
        ok(f"27 capitais selecionadas.")
        return "Capitais", geocodes
 
    # ── Municipal ────────────────────────────────────────────────────────────
    elif choice == "4":
        print("\n  Como deseja pesquisar?")
        print("    [A]  Código IBGE (7 dígitos)   ex: 3550308")
        print("    [B]  Nome do município          ex: São Paulo")
        sub = input("  Escolha (A/B): ").strip().upper()
 
        if sub == "A":
            raw = input("  Código IBGE: ").strip()
            if not raw.isdigit() or len(raw) not in (6, 7):
                err("Código IBGE inválido (6 ou 7 dígitos).")
                sys.exit(1)
            geocode = int(raw)
            nome = next(
                (m["nome"] for mlist in uf_map.values()
                 for m in mlist if m["id"] == geocode),
                f"geocode:{geocode}"
            )
            ok(f"Município: {nome} ({geocode})")
            return f"Municipal — {nome} ({geocode})", [geocode]
 
        elif sub == "B":
            query = input("  Nome (mínimo 3 letras): ").strip()
            if len(query) < 3:
                err("Mínimo 3 caracteres.")
                sys.exit(1)
            results = search_municipio(uf_map, query)
            if not results:
                err(f"Nenhum município encontrado para '{query}'.")
                sys.exit(1)
            display = results[:30]
            rows = [[i+1, m["nome"], m["uf"], m["id"]]
                    for i, m in enumerate(display)]
            print()
            print(_tt(rows, ["#", "Município", "UF", "Geocode"],
                      col_align=["r","l","c","r"], width=70))
            if len(results) > 30:
                warn(f"Exibindo primeiros 30 de {len(results)} resultados.")
            idx = input("\n  Número do município desejado: ").strip()
            if not idx.isdigit() or not (1 <= int(idx) <= len(display)):
                err("Seleção inválida.")
                sys.exit(1)
            chosen = display[int(idx) - 1]
            ok(f"Selecionado: {chosen['nome']} — {chosen['uf']} ({chosen['id']})")
            return (
                f"Municipal — {chosen['nome']}/{chosen['uf']} ({chosen['id']})",
                [chosen["id"]]
            )
        else:
            err("Opção inválida.")
            sys.exit(1)
    else:
        err("Opção inválida.")
        sys.exit(1)
 
 
def _show_uf_summary(uf_map: dict) -> None:
    rows = sorted([[uf, len(uf_map[uf])] for uf in uf_map])
    chunk_size = (len(rows) + 2) // 3
    chunks = [rows[i:i+chunk_size] for i in range(0, len(rows), chunk_size)]
    max_len = max(len(c) for c in chunks)
    col_rows = []
    for i in range(max_len):
        row = []
        for c in chunks:
            row.extend(c[i] if i < len(c) else ["", ""])
        col_rows.append(row)
    print()
    if TEXTTABLE_OK:
        t = Texttable(max_width=90)
        t.set_deco(Texttable.HEADER | Texttable.BORDER | Texttable.VLINES)
        t.header(["UF", "Mun.", "UF", "Mun.", "UF", "Mun."])
        t.set_cols_align(["c","r","c","r","c","r"])
        for r in col_rows:
            padded = (r + ["", "", "", "", "", ""])[:6]
            t.add_row(padded)
        print(t.draw())
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MENU — RECORTE TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────
def ask_date_range() -> tuple[str, dict]:
    menu_title("PASSO 3 / 5  —  📅  RECORTE TEMPORAL")
    print(f"""
  {YELLOW}📁 Data Epidemiológica{RESET}
  ├── [1]  📅  ANO completo    (SE 01 a 53)
  ├── [2]  📆  MÊS             (converte para semanas epidemiológicas)
  └── [3]  🗓   SEMANA          (intervalo customizado de SE)
""")
    choice = input("  Escolha (1-3): ").strip()
 
    if choice == "1":
        y_ini = _ask_int("  Ano início (ex: 2022): ", 2000, 2030)
        y_end = _ask_int("  Ano fim    (ex: 2024): ", y_ini, 2030)
        params = dict(ew_start=1, ew_end=53, ey_start=y_ini, ey_end=y_end)
        label  = f"Ano {y_ini}–{y_end}  |  SE 01–53"
 
    elif choice == "2":
        y_ini = _ask_int("  Ano início: ", 2000, 2030)
        m_ini = _ask_int("  Mês início (1-12): ", 1, 12)
        y_end = _ask_int("  Ano fim:   ", y_ini, 2030)
        m_end = _ask_int("  Mês fim   (1-12): ", 1, 12)
        se_ini = max(1,  int((m_ini - 1) * 52 / 12) + 1)
        se_end = min(53, int(m_end * 52 / 12))
        params = dict(ew_start=se_ini, ew_end=se_end, ey_start=y_ini, ey_end=y_end)
        label  = f"{m_ini:02d}/{y_ini}–{m_end:02d}/{y_end}  |  SE {se_ini:02d}–{se_end:02d}"
 
    elif choice == "3":
        y_ini  = _ask_int("  Ano início: ", 2000, 2030)
        se_ini = _ask_int("  SE início (1-53): ", 1, 53)
        y_end  = _ask_int("  Ano fim:   ", y_ini, 2030)
        se_end = _ask_int("  SE fim    (1-53): ", 1, 53)
        params = dict(ew_start=se_ini, ew_end=se_end, ey_start=y_ini, ey_end=y_end)
        label  = f"SE {se_ini:02d}/{y_ini}–SE {se_end:02d}/{y_end}"
 
    else:
        err("Opção inválida.")
        sys.exit(1)
 
    return label, params
 
 
def _ask_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        warn(f"Informe um número entre {lo} e {hi}.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MENU — FORMATO DE SAÍDA
# ─────────────────────────────────────────────────────────────────────────────
def ask_output_format() -> str:
    menu_title("PASSO 4 / 5  —  📁  FORMATO DE SAÍDA")
    print(f"""
  {YELLOW}📁 Selecione o formato do arquivo de saída{RESET}
  ├── [1]  📄  CSV    (compatível com Excel · R · pandas)
  └── [2]  🗂   JSON   (estruturado · APIs · web)
""")
    choice = input("  Escolha (1-2): ").strip()
    return "json" if choice == "2" else "csv"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ESTRUTURA DE PASTAS
# ─────────────────────────────────────────────────────────────────────────────
def build_output_path(disease_id: str, scope_label: str,
                      date_label: str, fmt: str) -> Path:
    """
    Base_de_dados/
      └─ Arbovirose/
           └─ DENGUE | CHIKUNGUNYA | ZIKA
                └─ ESCOPO (Nacional / Estadual / Capitais / Municipal)
                     └─ ANO | MÊS | SEMANA
                          └─ CSV | JSON
                               └─ arquivo.csv / arquivo.json
    """
    disease_dir = disease_id.upper()
    scope_clean = scope_label.split("—")[0].strip()
 
    if "Ano" in date_label:
        time_dir = "ANO"
    elif "MÊS" in date_label or ("/" in date_label and "SE" in date_label
                                  and "–SE" not in date_label):
        time_dir = "MÊS"
    else:
        time_dir = "SEMANA"
 
    out_dir = (SCRIPT_DIR / "Base_de_dados" / "Arbovirose"
               / disease_dir / scope_clean / time_dir / fmt.upper())
    out_dir.mkdir(parents=True, exist_ok=True)
 
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{disease_id}_{scope_clean.lower()}_{ts}.{fmt}"
    return out_dir / filename
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD — CORE
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_one(geocode: int, params: dict, disease: str) -> list[dict]:
    query = {
        "geocode":  geocode,
        "disease":  disease,
        "format":   "json",
        "ew_start": params["ew_start"],
        "ew_end":   params["ew_end"],
        "ey_start": params["ey_start"],
        "ey_end":   params["ey_end"],
    }
    for attempt in range(1, API_RETRIES + 1):
        try:
            resp = requests.get(INFODENGUE_URL, params=query, timeout=API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except requests.exceptions.HTTPError:
            if resp.status_code == 404:
                return []
            if attempt < API_RETRIES:
                time.sleep(1.5 * attempt)
        except requests.exceptions.Timeout:
            if attempt < API_RETRIES:
                time.sleep(2.0 * attempt)
        except Exception:
            return []
    return []
 
 
def download_all(
    geocodes: list[int], params: dict, disease: str,
    disease_label: str, fmt: str, out_path: Path
) -> tuple[int, int, int]:
    all_records: list[dict] = []
    errors = 0
 
    menu_title(f"PASSO 5 / 5  —  {disease_label}  ·  DOWNLOAD EM PROGRESSO")
    info(f"Municípios na fila  : {len(geocodes):,}")
    info(f"Arbovirose          : {disease_label}")
    info(f"Parâmetros da API   : {params}")
    info(f"Destino             : {out_path}\n")
 
    iterator = (
        tqdm(geocodes,
             desc=f"  {DISEASES.get(next((k for k,v in DISEASES.items() if v['id']==disease),'1'),DISEASES['1'])['emoji']} Baixando",
             unit="mun", colour="cyan", dynamic_ncols=True)
        if TQDM_OK else geocodes
    )
 
    for gc in iterator:
        records = _fetch_one(gc, params, disease)
        if not records:
            errors += 1
        else:
            all_records.extend(records)
        time.sleep(API_DELAY)
 
    total_records  = len(all_records)
    total_requests = len(geocodes)
 
    print()
    if total_records == 0:
        err("Nenhum registro retornado. Verifique parâmetros e tente novamente.")
    else:
        if fmt == "csv":
            _save_csv(all_records, out_path)
        else:
            _save_json(all_records, out_path)
        ok(f"Arquivo salvo       → {out_path}")
        ok(f"Total de registros  : {total_records:,}")
        ok(f"Erros / sem dados   : {errors} / {total_requests}")
 
    return total_records, total_requests, errors
 
 
def _save_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    fields = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
 
 
def _save_json(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISE EXPLORATÓRIA
# ─────────────────────────────────────────────────────────────────────────────
NUMERIC_COLS = ["casos","casos_est","cases_est_min","cases_est_max",
                "p_rt1","p_inc100k","Rt","pop",
                "tempmin","tempmed","tempmax",
                "umidmin","umidmed","umidmax"]
 
ALERT_LEVELS = {1:"🟢 Verde",2:"🟡 Amarelo",3:"🟠 Laranja",4:"🔴 Vermelho"}
 
def explore(path: Path) -> Optional[dict]:
    """Análise exploratória; retorna dict de stats para o log."""
    if not PANDAS_OK or not path.exists():
        return None
    try:
        df = (pd.read_csv(path, low_memory=False)
              if str(path).endswith(".csv")
              else pd.read_json(path))
    except Exception as e:
        warn(f"Análise não disponível: {e}")
        return None
 
    if df.empty:
        return None
 
    menu_title("📊  ANÁLISE EXPLORATÓRIA DOS DADOS BAIXADOS")
    info(f"Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
 
    existing = [c for c in NUMERIC_COLS if c in df.columns]
    stats_dict = {}
 
    if existing and TEXTTABLE_OK:
        rows = []
        for col in existing:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            rows.append([col, f"{s.min():.2f}", f"{s.mean():.2f}",
                         f"{s.median():.2f}", f"{s.max():.2f}",
                         str(df[col].isna().sum())])
            stats_dict[f"{col}_mean"] = f"{s.mean():.2f}"
            stats_dict[f"{col}_max"]  = f"{s.max():.2f}"
        if rows:
            print()
            print(_tt(rows, ["Coluna","Mín","Média","Mediana","Máx","Nulos"],
                      col_align=["l","r","r","r","r","r"]))
 
    if "nivel" in df.columns and TEXTTABLE_OK:
        counts = df["nivel"].value_counts().sort_index()
        alert_rows = []
        for nivel, cnt in counts.items():
            label = ALERT_LEVELS.get(int(nivel), str(nivel))
            pct   = 100 * cnt / len(df)
            alert_rows.append([label, f"{cnt:,}", f"{pct:.1f}%"])
            stats_dict[f"alerta_{label.split()[1].lower()}"] = f"{cnt:,}"
        print()
        print(_tt(alert_rows, ["Nível de Alerta","Registros","% Total"],
                  col_align=["l","r","r"], width=60))
 
    if "data_ini_SE" in df.columns:
        dates = pd.to_datetime(df["data_ini_SE"], errors="coerce").dropna()
        if not dates.empty:
            periodo = f"{dates.min().date()} → {dates.max().date()}"
            info(f"Período dos dados : {periodo}")
            stats_dict["periodo_dados"] = periodo
 
    return stats_dict
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DICIONÁRIO DE DADOS
# ─────────────────────────────────────────────────────────────────────────────
DICT_DADOS = [
    ["data_ini_SE",      "Primeiro dia da semana epidemiológica (Domingo)"],
    ["SE",               "Número da semana epidemiológica"],
    ["casos_est",        "Casos estimados por nowcasting (atualização retroativa semanal)"],
    ["cases_est_min",    "Limite inferior do IC 95% dos casos estimados"],
    ["cases_est_max",    "Limite superior do IC 95% dos casos estimados"],
    ["casos",            "Casos notificados por semana (atualizado retroativamente)"],
    ["p_rt1",            "P(Rt > 1) — alerta laranja quando > 0,95 por ≥ 3 SE"],
    ["p_inc100k",        "Taxa de incidência estimada por 100.000 habitantes"],
    ["nivel",            "Nível de alerta: 1=Verde  2=Amarelo  3=Laranja  4=Vermelho"],
    ["Rt",               "Número reprodutivo efetivo estimado"],
    ["pop",              "População municipal estimada pelo IBGE"],
    ["tempmin",          "Média das temperaturas mínimas diárias da semana (°C)"],
    ["tempmed",          "Média das temperaturas diárias da semana (°C)"],
    ["tempmax",          "Média das temperaturas máximas diárias da semana (°C)"],
    ["umidmin",          "Média da umidade relativa mínima diária (%)"],
    ["umidmed",          "Média da umidade relativa diária (%)"],
    ["umidmax",          "Média da umidade relativa máxima diária (%)"],
    ["receptivo",        "Receptividade climática: 0=desfav. 1=fav. 2=×2sem 3=≥3sem"],
    ["transmissao",      "Transmissão: 0=nenhuma 1=possível 2=provável 3=altamente provável"],
    ["nivel_inc",        "Incidência: 0=sub-limiar  1=pré-epidemia  2=epidêmico"],
    ["notif_accum_year", "Acumulado de casos no ano corrente"],
    ["Localidade_id",    "Divisão submunicipal (apenas Rio de Janeiro)"],
]
 
def show_data_dict() -> None:
    if not TEXTTABLE_OK:
        return
    menu_title("📚  DICIONÁRIO DE DADOS — InfoDengue")
    print(_tt(DICT_DADOS, ["Campo","Descrição"],
              col_align=["l","l"], width=105))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# VERIFICAÇÃO DE DEPENDÊNCIAS
# ─────────────────────────────────────────────────────────────────────────────
def _check_deps() -> None:
    missing = []
    if not TQDM_OK:      missing.append("tqdm")
    if not TEXTTABLE_OK: missing.append("texttable")
    if missing:
        err(f"Dependências ausentes — instale com:")
        err(f"    pip install {' '.join(missing)}")
        sys.exit(1)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    banner()
    _check_deps()
 
    # ── 0. Catálogo IBGE ─────────────────────────────────────────────────────
    menu_title("🗺   CARREGANDO CATÁLOGO IBGE DE MUNICÍPIOS")
    uf_map, ibge_source = load_ibge_municipios()
    total_mun = sum(len(v) for v in uf_map.values())
    ok(f"Municípios carregados : {total_mun:,}  |  UFs: {len(uf_map)}")
    info(f"Fonte: {ibge_source}")
 
    show_data_dict()
 
    # ── 1–5. Coleta de parâmetros ─────────────────────────────────────────────
    disease_id, disease_label = ask_disease()
    scope_label, geocodes     = ask_scope(uf_map)
    date_label,  params       = ask_date_range()
    fmt                       = ask_output_format()
 
    # Paths
    start_ts = datetime.datetime.now()
    out_path  = build_output_path(disease_id, scope_label, date_label, fmt)
    log_path  = _make_log_path(fmt, start_ts)
 
    # ── Confirmação ──────────────────────────────────────────────────────────
    menu_title("⚙️   RESUMO DA OPERAÇÃO")
    resume_rows = [
        ["Arbovirose",            disease_label],
        ["Escopo geográfico",     scope_label],
        ["Municípios na fila",    f"{len(geocodes):,}"],
        ["Filtro temporal",       date_label],
        ["Formato de saída",      fmt.upper()],
        ["Arquivo de dados",      str(out_path)],
        ["Arquivo de log",        log_path.name],
        ["Fonte geocodes (IBGE)", ibge_source],
        ["Delay / Retry",         f"{API_DELAY}s  ·  {API_RETRIES} tentativas"],
    ]
    print(_tt(resume_rows, ["Parâmetro","Valor"],
              col_align=["l","l"], width=105))
 
    confirm = input("\n  ▶  Iniciar download? (S/n): ").strip().lower()
    if confirm == "n":
        err("Operação cancelada pelo usuário.")
        sys.exit(0)
 
    # ── Execução ─────────────────────────────────────────────────────────────
    total_records, total_requests, errors = download_all(
        geocodes, params, disease_id, disease_label, fmt, out_path
    )
    end_ts = datetime.datetime.now()
 
    # ── Análise exploratória ─────────────────────────────────────────────────
    stats = explore(out_path) if total_records > 0 else None
 
    # ── Registro de log ──────────────────────────────────────────────────────
    write_log_entry(
        log_path        = log_path,
        disease         = disease_id,
        scope           = scope_label,
        date_filter     = date_label,
        output_fmt      = fmt,
        output_path     = str(out_path),
        start_ts        = start_ts,
        end_ts          = end_ts,
        total_records   = total_records,
        total_requests  = total_requests,
        errors          = errors,
        ibge_source     = ibge_source,
        geocodes_sample = geocodes,
        stats           = stats,
    )
 
    elapsed = (end_ts - start_ts).total_seconds()
    print(f"\n{GREEN}{BOLD}"
          f"  ✅  {disease_label}  ·  Concluído em {elapsed:.2f}s  |  "
          f"{total_records:,} registros  |  {errors} erros"
          f"{RESET}\n")
 
 
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()