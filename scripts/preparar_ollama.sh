#!/usr/bin/env bash
# Prepara o LLM local do MOVER no Linux/macOS: instala o Ollama (se faltar), liga o serviço, baixa o
# modelo gemma2:2b (o do benchmark do artigo), mede a latência e grava o .env da raiz do repositório.
#
#   bash scripts/preparar_ollama.sh                 # gemma2:2b
#   bash scripts/preparar_ollama.sh qwen2.5:1.5b    # outro modelo
set -euo pipefail

MODELO="${1:-gemma2:2b}"
HOSPEDEIRO="${OLLAMA_HOST:-http://localhost:11434}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"

escrever() { printf '[MOVER] %s\n' "$*"; }
no_ar() { curl -fsS --max-time 3 "$HOSPEDEIRO/api/version" >/dev/null 2>&1; }

if ! command -v ollama >/dev/null 2>&1; then
  escrever "Instalando o Ollama (script oficial de https://ollama.com)..."
  curl -fsSL https://ollama.com/install.sh | sh
fi
escrever "Ollama: $(command -v ollama)"

if ! no_ar; then
  escrever "Ligando o serviço do Ollama (ollama serve)..."
  nohup ollama serve >/tmp/ollama_mover.log 2>&1 &
  for _ in $(seq 1 60); do no_ar && break; sleep 1; done
  no_ar || { escrever "O serviço não respondeu em $HOSPEDEIRO (veja /tmp/ollama_mover.log)."; exit 1; }
fi
escrever "Serviço no ar: $(curl -fsS "$HOSPEDEIRO/api/version")"

escrever "Baixando o modelo $MODELO (na primeira vez, ~1,6 GB)..."
ollama pull "$MODELO"

corpo=$(printf '{"model": "%s", "prompt": "Em uma frase curta: o que é um gêmeo digital?", "stream": false, "options": {"num_predict": 60, "temperature": 0.1}}' "$MODELO")
for rodada in aquecimento medida; do
  inicio=$(date +%s.%N)
  resposta=$(curl -fsS --max-time 300 "$HOSPEDEIRO/api/generate" -H 'Content-Type: application/json' -d "$corpo")
  fim=$(date +%s.%N)
  python3 - "$rodada" "$inicio" "$fim" "$resposta" <<'PY'
import json, sys
rodada, inicio, fim, resposta = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), json.loads(sys.argv[4])
tokens_s = resposta.get("eval_count", 0) / (resposta.get("eval_duration", 1) / 1e9) if resposta.get("eval_duration") else 0
print(f"[MOVER] Chamada de {rodada}: {fim - inicio:.1f} s, {tokens_s:.1f} tokens/s.")
if rodada == "medida":
    print("[MOVER] Resposta:", resposta.get("response", "").strip())
    if fim - inicio > 6:
        print("[MOVER] Acima do prazo de 6 s das decisões: elas vão cair na regra (considere GPU ou um modelo menor).")
PY
done

ENV="$RAIZ/.env"
touch "$ENV"
grep -vE '^(MOVER_LLM_PROVEDOR|MOVER_LLM_MODELO|OLLAMA_HOST)=' "$ENV" > "$ENV.tmp" || true
printf 'MOVER_LLM_PROVEDOR=ollama\nMOVER_LLM_MODELO=%s\nOLLAMA_HOST=%s\n' "$MODELO" "$HOSPEDEIRO" >> "$ENV.tmp"
mv "$ENV.tmp" "$ENV"
escrever "Gravado em $ENV: MOVER_LLM_PROVEDOR=ollama, MOVER_LLM_MODELO=$MODELO."
escrever "Pronto. Com o CARLA aberto: cd src && python -m mover.autonomo.dirigir"
