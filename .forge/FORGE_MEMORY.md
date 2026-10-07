# FORGE Project Memory - TCC MOVER (Agentes IA)

## Projeto
- TCC Instituto Maua / Projeto MOVER: infraestrutura cognitiva multiagente para caminhoes autonomos.
- Stack: Python 3.12, CARLA 0.9.16 (modo sincrono, dt=0.05s), LangGraph + OpenAI (gpt-4o-mini), IsolationForest, Streamlit/Plotly, Ollama (LLMs locais).
- `backup/src/` = versoes antigas; fonte ativa em `src/`.

## Mapa do codigo (src/)
- `coleta_carla.py` (v7): coletor CARLA. Ego firetruck + NPCs/pedestres; sensores camera RGB, segmentacao semantica (video lado a lado), LiDAR (.npy), GNSS, IMU, collision, lane_invasion. HUD + minimapa no video.
  - `GeradorDadosCaminhao`: pacotes mock (manobras) a cada N s; injeta controle 1 tick e devolve ao autopilot. Modos `control`/`kinematic`.
  - Saidas por run: telemetria.csv (inclui cmd_*), eventos.csv/.log, video.mp4, dados_mock.csv, metadata.json, condicoes_iniciais.json, lidar/.
  - CLI: --duration --output --town --vehicles --pedestrians --no-hud --no-lidar --no-seg --autopilot --data-interval --data-mode --seed --replay.
- `dashboard.py`: PNG 2x2 (trajetoria, velocidade, comandos, IMU) a partir de telemetria.csv + eventos.csv.
- `gerar_videos.py`: legado (monta MP4 de camera/*.png; v7 ja grava video direto).
- `event_reader.py`: script trivial, path fixo `dataset/run_XYZ-colisao`.
- `visualizar_mapa.py`: plota OSM (`data/openstreetviewmap.osm`) x GPS (`data/csv_maua/Location.csv`); salva `data/mapa_check.png`. Validar antes de converter p/ .xodr.
- `poc-agents/app.py`: Streamlit + LangGraph. Especialista (IsolationForest, features speed/throttle/brake/steer/acc_x/acc_y/imu_gyro_z, buffer 20, cooldown 20 frames) -> Supervisor LLM (diagnostico + comando estruturado). Logs em data/logs_essenciais.json.
- `poc-agents/treinar_ml.py`: treina IsolationForest (contamination=0.05) -> modelo_especialista.joblib.
- `benchmarks/throughput.py`: compara LLMs Ollama (qwen2.5 0.5b/1.5b, gemma2 2b, llama3.2 1b) resumindo 20 logs; mede TPS; chama `hallucination.py` (auditoria GPT-4o). Saida em data/resultados_pesquisa/.

## Pontos de atencao identificados (2026-10-06)
- `app.py` e `treinar_ml.py`: DATA_DIR hardcoded para path de outro membro (`C:\Users\danie\...\src\data`); dados reais estao em `<root>/data`.
- `treinar_ml.py` le `telemetria.csv` relativo ao cwd.
- `app.py`: prompt diz "5 frames" mas envia buffer de 20.
- `throughput.py`: campos `*_ms` recebem nanossegundos do Ollama (calculo de TPS correto, nome enganoso).
- `requirements.txt` incompleto (faltam pandas, numpy, opencv-python, matplotlib, ollama, openai, pydantic; carla via wheel do simulador).
- README referencia nomes antigos (`coleta_carla_v4.py`, `gerar_video.py`).

## Status
- Contexto carregado; nenhuma alteracao de codigo feita ainda.

## Proximos passos sugeridos
1. Padronizar paths (relativos a raiz via `Path(__file__)`) em poc-agents.
2. Completar requirements.txt e atualizar README.
3. Definir proxima entrega do TCC com o usuario.
