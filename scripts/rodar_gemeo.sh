#!/usr/bin/env bash
# Roda o gêmeo digital do MOVER: o caminhão autônomo dá a volta no campus do IMT sozinho no
# CARLA. A cada 10 s ele manda aos agentes o plano dos próximos segundos (a velocidade e a
# aceleração que vai aplicar), montado a partir da corrida atual. Um modelo de ML propõe quanto
# suavizar as frenagens e arrancadas bruscas desse plano, o modelo do caminhão confere, o LLM local
# aprova e o caminhão executa o plano alterado.
#
# Uso, da raiz do repositório (no Windows, pelo Git Bash):
#   ./scripts/rodar_gemeo.sh coleta      # coleta os dados de uma volta com os agentes no CARLA (sem painel)
#   ./scripts/rodar_gemeo.sh visual      # a volta com os agentes no CARLA, vendo o caminhão, e o painel
#   ./scripts/rodar_gemeo.sh gravar      # grava a volta com os agentes no CARLA e gera o vídeo MP4
#   ./scripts/rodar_gemeo.sh video       # gera de novo o MP4 da última corrida gravada
#   ./scripts/rodar_gemeo.sh painel      # o painel em http://127.0.0.1:8000/ sem o CARLA
#   ./scripts/rodar_gemeo.sh servidor    # só o servidor do painel (para várias corridas)
#   ./scripts/rodar_gemeo.sh benchmark   # N voltas sem e N com os agentes (CSV e gráfico)
#   ./scripts/rodar_gemeo.sh treinar     # treina de novo o modelo de ML que propõe a suavização
#   ./scripts/rodar_gemeo.sh analise     # a camada de logs e o ML sobre a corrida mais recente
#   ./scripts/rodar_gemeo.sh ollama      # instala e liga o Ollama e baixa o gemma2:2b
#   ./scripts/rodar_gemeo.sh carla       # só abre o CARLA com janela (precisa de CARLA_ROOT)
#   ./scripts/rodar_gemeo.sh testes      # roda os testes
# Sem modo, roda o visual. Na primeira corrida com agentes, o modelo de ML é treinado sozinho
# (~1-2 min). Opções a mais vão para o comando Python, por exemplo:
#   ./scripts/rodar_gemeo.sh coleta --sem-agentes    # a volta do caminhão sozinho, para comparar
#   ./scripts/rodar_gemeo.sh visual --voltas 2 --sem-llm
#   ./scripts/rodar_gemeo.sh video --camera frontal
#
# Variáveis de ambiente (todas opcionais):
#   PYTHON           Python a usar (padrão: env_carla/ ou .venv/ da raiz; senão python3 ou python)
#   CARLA_ROOT       pasta do CARLA. Se o simulador não estiver aberto, o script o abre com janela
#                    (no Windows, o padrão é C:/CARLA_0.9.16, se existir)
#   CARLA_HOST       padrão localhost; CARLA_PORTA, padrão 2000 (os do config/simulacao.yaml)
#   CARLA_QUALIDADE  Low ou Epic (padrão: Epic em gravar e video, Low no resto)
#   OLLAMA_HOST      padrão http://localhost:11434
#   MOVER_SECO=1     só mostra os comandos, sem rodar nada
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODO="${1:-visual}"
if [ $# -gt 0 ]; then shift; fi

CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORTA="${CARLA_PORTA:-2000}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
SECO="${MOVER_SECO:-0}"
TEMP="${TMPDIR:-/tmp}"
# acentos e símbolos (m/s², ≥) sem erro de codificação no terminal do Windows
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

escrever() { printf '[MOVER] %s\n' "$*"; }
falhar() { printf '[MOVER] ERRO: %s\n' "$*" >&2; exit 1; }

ajuda() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

no_windows() {
  case "$(uname -s 2>/dev/null)" in
    MINGW* | MSYS* | CYGWIN*) return 0 ;;
    *) return 1 ;;
  esac
}

achar_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return 0; fi
  local candidato
  for candidato in "$RAIZ/env_carla/Scripts/python.exe" "$RAIZ/env_carla/bin/python" \
                   "$RAIZ/.venv/Scripts/python.exe" "$RAIZ/.venv/bin/python"; do
    if [ -x "$candidato" ]; then echo "$candidato"; return 0; fi
  done
  for candidato in python3 python py; do
    if command -v "$candidato" >/dev/null 2>&1; then echo "$candidato"; return 0; fi
  done
  falhar "Python não encontrado. Crie o ambiente (veja o README) ou defina PYTHON."
}

# Roda um comando Python a partir de src/, como no README. Com MOVER_SECO=1, só mostra.
py_src() {
  if [ "$SECO" = "1" ]; then escrever "(cd src) $PY $*"; return 0; fi
  (cd "$RAIZ/src" && "$PY" "$@")
}

# em_segundo_plano ARQUIVO_DE_LOG COMANDO...: o processo continua depois que o script termina
em_segundo_plano() {
  local registro="$1"; shift
  if command -v nohup >/dev/null 2>&1; then
    nohup "$@" >"$registro" 2>&1 &
  else
    "$@" >"$registro" 2>&1 &
  fi
}

tem_opcao() {  # tem_opcao --sem-carla "$@"
  local alvo="$1"; shift
  local arg
  for arg in "$@"; do
    if [ "$arg" = "$alvo" ]; then return 0; fi
  done
  return 1
}

valor_opcao() {  # valor_opcao --provedor "$@"  ->  o valor, ou nada
  local alvo="$1"; shift
  while [ $# -gt 0 ]; do
    case "$1" in
      "$alvo") echo "${2:-}"; return 0 ;;
      "$alvo"=*) echo "${1#*=}"; return 0 ;;
    esac
    shift
  done
  return 0
}

porta_aberta() {  # porta_aberta HOST PORTA
  "$PY" - "$1" "$2" >/dev/null 2>&1 <<'PY'
import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2).close()
except OSError:
    sys.exit(1)
PY
}

ollama_no_ar() {
  "$PY" - "$OLLAMA_HOST" >/dev/null 2>&1 <<'PY'
import sys, urllib.request
hospedeiro = sys.argv[1] if "://" in sys.argv[1] else "http://" + sys.argv[1]
try:
    urllib.request.urlopen(hospedeiro.rstrip("/") + "/api/version", timeout=3).close()
except Exception:
    sys.exit(1)
PY
}

executavel_carla() {
  local raiz_carla="${CARLA_ROOT:-}"
  if [ -z "$raiz_carla" ] && no_windows && [ -d "/c/CARLA_0.9.16" ]; then raiz_carla="/c/CARLA_0.9.16"; fi
  if [ -z "$raiz_carla" ]; then return 1; fi
  local exe
  for exe in "$raiz_carla/CarlaUE4.exe" "$raiz_carla/CarlaUE4.sh"; do
    if [ -f "$exe" ]; then echo "$exe"; return 0; fi
  done
  return 1
}

# garantir_carla QUALIDADE: confere se o CARLA está aberto; se não estiver e CARLA_ROOT apontar
# para ele, abre com janela (sem -RenderOffScreen) e espera a porta responder.
garantir_carla() {
  local qualidade="${CARLA_QUALIDADE:-$1}"
  if [ "$SECO" = "1" ]; then
    escrever "(confere o CARLA em $CARLA_HOST:$CARLA_PORTA; se precisar abrir, com -quality-level=$qualidade)"
    return 0
  fi
  if porta_aberta "$CARLA_HOST" "$CARLA_PORTA"; then
    escrever "CARLA no ar em $CARLA_HOST:$CARLA_PORTA."
    if [ "$qualidade" = "Epic" ]; then
      escrever "Para o vídeo ficar bom, o CARLA precisa estar aberto com -quality-level=Epic."
    fi
    return 0
  fi
  local exe
  if ! exe="$(executavel_carla)"; then
    falhar "O CARLA não está aberto em $CARLA_HOST:$CARLA_PORTA. Abra com janela (sem -RenderOffScreen):
    C:\\CARLA_0.9.16\\CarlaUE4.exe -quality-level=$qualidade -world-port=$CARLA_PORTA
ou defina CARLA_ROOT (pasta do CARLA) para o script abrir sozinho."
  fi
  escrever "Abrindo o CARLA com janela: $exe -quality-level=$qualidade -world-port=$CARLA_PORTA"
  em_segundo_plano "$TEMP/carla_mover.log" "$exe" -quality-level="$qualidade" -world-port="$CARLA_PORTA"
  local i
  for i in $(seq 1 180); do
    if porta_aberta "$CARLA_HOST" "$CARLA_PORTA"; then
      escrever "CARLA no ar depois de ${i} s (ele continua aberto quando o script terminar)."
      sleep 5  # o mundo inicial ainda termina de carregar
      return 0
    fi
    if [ $((i % 15)) -eq 0 ]; then escrever "Esperando o CARLA... (${i} s)"; fi
    sleep 1
  done
  falhar "O CARLA não respondeu em 3 min (veja $TEMP/carla_mover.log)."
}

# garantir_ollama "$@": se as decisões vão usar o Ollama, confere o serviço e tenta ligá-lo.
# Sem ele a corrida roda igual, com as decisões só pela regra.
garantir_ollama() {
  if tem_opcao --sem-llm "$@" || tem_opcao --sem-agentes "$@"; then return 0; fi
  local provedor
  provedor="$(valor_opcao --provedor "$@")"
  if [ -n "$provedor" ] && [ "$provedor" != "ollama" ]; then return 0; fi
  if [ "$SECO" = "1" ]; then escrever "(confere o Ollama em $OLLAMA_HOST)"; return 0; fi
  if ollama_no_ar; then return 0; fi
  if command -v ollama >/dev/null 2>&1; then
    escrever "Ligando o Ollama (ollama serve)..."
    em_segundo_plano "$TEMP/ollama_mover.log" ollama serve
    local i
    for i in $(seq 1 30); do
      if ollama_no_ar; then return 0; fi
      sleep 1
    done
  fi
  escrever "Aviso: o Ollama não está no ar em $OLLAMA_HOST, então as decisões ficam com a regra."
  escrever "Para instalar e ligar: ./scripts/rodar_gemeo.sh ollama"
}

case "$MODO" in
  ajuda | -h | --help)
    ajuda
    exit 0
    ;;
esac

PY="$(achar_python)"

case "$MODO" in
  visual)
    tem_opcao --sem-carla "$@" || garantir_carla Low
    garantir_ollama "$@"
    escrever "A janela do CARLA segue o caminhão; o painel abre em http://127.0.0.1:8000/. No fim, Ctrl+C fecha o painel."
    py_src -m mover.autonomo.dirigir "$@"
    ;;
  coleta)
    tem_opcao --sem-carla "$@" || garantir_carla Low
    garantir_ollama "$@"
    py_src -m mover.autonomo.dirigir --sem-painel "$@"
    escrever "Dados em data/autonomo/corridas/<data>_<modo>/: telemetria.csv, decisoes.jsonl (com o plano de cada bloco) e resumo.json."
    ;;
  gravar)
    garantir_carla Epic
    garantir_ollama "$@"
    py_src -m mover.autonomo.dirigir --gravar --sair-no-fim "$@"
    py_src -m mover.video.gravar_video
    escrever "Vídeo em data/autonomo/corridas/<data>_<modo>/video.mp4."
    ;;
  video)
    tem_opcao --teste-hud "$@" || garantir_carla Epic
    py_src -m mover.video.gravar_video "$@"
    ;;
  painel)
    garantir_ollama "$@"
    escrever "Painel em http://127.0.0.1:8000/ com o simulador substituto (sem o CARLA). No fim, Ctrl+C fecha o painel."
    py_src -m mover.autonomo.dirigir --sem-carla "$@"
    ;;
  servidor)
    escrever "Painel em http://127.0.0.1:8000/. Rode as corridas em outro terminal (visual, coleta ou painel)."
    py_src -m mover.servidor.rodar_servidor "$@"
    ;;
  benchmark)
    tem_opcao --sem-carla "$@" || garantir_carla Low
    garantir_ollama "$@"
    py_src -m mover.autonomo.benchmark "$@"
    ;;
  treinar)
    py_src -m mover.agentes.suavizador_ml "$@"
    escrever "Modelo em data/modelos/suavizador_ml.joblib (as métricas do treino ficam no .json ao lado)."
    ;;
  analise)
    garantir_ollama "$@"
    py_src -m mover.agentes.rodar_agentes "$@"
    escrever "Log em data/agentes/log_agentes.jsonl (e o resumo ao lado)."
    ;;
  ollama)
    if no_windows; then
      ps1="$RAIZ/scripts/preparar_ollama.ps1"
      if command -v cygpath >/dev/null 2>&1; then ps1="$(cygpath -w "$ps1")"; fi
      if [ "$SECO" = "1" ]; then escrever "powershell.exe -ExecutionPolicy Bypass -File $ps1 $*"; exit 0; fi
      powershell.exe -ExecutionPolicy Bypass -File "$ps1" "$@"
    else
      if [ "$SECO" = "1" ]; then escrever "bash scripts/preparar_ollama.sh $*"; exit 0; fi
      bash "$RAIZ/scripts/preparar_ollama.sh" "$@"
    fi
    ;;
  carla)
    garantir_carla Low
    ;;
  testes)
    if [ "$SECO" = "1" ]; then escrever "$PY -m pytest tests -q $*"; exit 0; fi
    (cd "$RAIZ" && "$PY" -m pytest tests -q "$@")
    ;;
  *)
    falhar "Modo desconhecido: $MODO. Veja os modos com: ./scripts/rodar_gemeo.sh ajuda"
    ;;
esac
