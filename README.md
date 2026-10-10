# MOVER: gêmeo digital do caminhão com camada agêntica

Código das etapas 1 a 4 do TCC. A volta gravada pelo celular no campus do IMT vira telemetria no
formato do CARLA. Uma camada agêntica escreve o log da condução em linguagem natural e aponta os
problemas de jerk. O caminhão do CARLA refaz a volta em blocos de 10 s, e os agentes analisam cada
bloco antes de o caminhão chegar nele. Uma página com 4 painéis (simulação, dashboard, log e chat)
acompanha tudo ao vivo.

```
data/csv_maua ──(1) tratamento──▶ data/tratado/telemetria_tratada.csv (20 Hz, colunas do coleta_carla)
                                            │
                      (2) camada agêntica: Especialistas (regras + IsolationForest) e Supervisor (LLM)
                                            │   log em janelas de 5 s e problemas de jerk
                                            ▼
     (3) replay no CARLA, blocos de 10 s ◀──▶ servidor FastAPI: sessão, análise do próximo bloco, /ws
                                            │
                      (4) página de 4 painéis: simulação · dashboard · log · chat sobre o log
```

Este README tem outro nome para não sobrescrever o README do repositório. Pelo mesmo motivo, as
dependências e as variáveis de ambiente ficam em `requirements-mover.txt` e `.env.mover.example`.

## Estrutura

```
config/
  tratamento.yaml      filtros, calibração e fusão GPS + IMU (etapa 1)
  agentes.yaml         blocos, limiares de jerk, LLM, chat e eventos sintéticos (etapa 2)
  simulacao.yaml       mapa, alinhamento, CARLA, câmeras, replay e servidor (etapas 3 e 4)
src/mover/
  config.py            raiz do projeto e leitura dos YAML
  tratamento/          sensor_logger, geo, calibracao, fusao, sinais, tratar_dados, validar
  agentes/             fatos, especialistas, textos, llm, supervisor, grafo, executor,
                       eventos_sinteticos, treinar_especialista_ml, rodar_agentes
  simulacao/           opendrive, projecao, alinhamento, verificar_mapa, camera_painel,
                       fila_envio, replay_carla
  servidor/            app (FastAPI + WebSocket), sessao, rodar_servidor
  interface/           rotas, cena, chat e estatico/ (index.html, painel.css e 6 módulos JS)
tests/                 auxiliares, conftest, test_agentes, test_servidor, test_simulacao, test_interface
requirements-mover.txt
.env.mover.example
```

Para usar junto com o repositório, copie o conteúdo da pasta `mover_gemeo/` do .zip para a raiz dele.
Nada é sobrescrito: o repositório não tem `src/mover/`, `config/` nem `tests/`, e os arquivos da raiz
têm nomes próprios.

Dados usados (não vão no .zip):

| Arquivo | Origem |
|---|---|
| `data/csv_maua/*.csv` | gravação do Sensor Logger, já no repositório |
| `data/openstreetviewmap.osm` | já no repositório (referência geográfica do tratamento) |
| `data/mapas/mapa_final.xodr` | mapa do campus feito pelo Eduardo (pasta do Drive `mapa_imt_plano_carla`), só lido |
| `data/tratado/`, `data/modelos/`, `data/agentes/`, `data/simulacao/` | saídas geradas pelos comandos abaixo |

Os mapas de `data/mapa_imt/` no repositório (`mapa_final_plano_2vias.xodr` e `mapa_final_3d.xodr`)
são a mesma malha do campus, com duas faixas por via, mas não têm `geoReference`. Sem ela, o
alinhamento supõe que o plano do mapa é o plano local da Etapa 1 e a volta fica a ~113 m das vias.
Por isso o padrão continua sendo o `mapa_final.xodr`.

## Instalação

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-mover.txt      # Windows: .venv\Scripts\pip install -r requirements-mover.txt
```

- Testado em Python 3.13; o código também compila em 3.12.
- Para o replay no CARLA 0.9.16, use um ambiente com Python 3.12, como o `venv_carla` do
  `executar.ps1` do Eduardo, e instale as dependências acima mais `pip install carla==0.9.16`.
- LLM: o padrão é o Ollama com `gemma2:2b` (`ollama pull gemma2:2b`). Para trocar, copie as linhas
  do `.env.mover.example` para o `.env` da raiz e escolha `MOVER_LLM_PROVEDOR` (`ollama`, `openai`,
  `gemini` ou `falso`). Também dá para usar `--provedor` e `--modelo` na linha de comando. O
  provedor `falso` não usa rede e serve para testar o fluxo inteiro.

Todos os comandos abaixo rodam a partir da pasta `src/`.

## Etapa 1: tratamento dos dados

```bash
python -m mover.tratamento.tratar_dados    # data/csv_maua -> data/tratado/telemetria_tratada.csv
python -m mover.tratamento.validar         # figuras de validação a partir dos arquivos tratados
```

A saída tem 3.032 linhas a 20 Hz: as colunas do `telemetria.csv` do `coleta_carla.py` (v7), na
mesma ordem, mais colunas extras. Throttle, brake e steer são estimativas (colunas `*_est`). O
relatório fica em `data/tratado/relatorio_tratamento.json`.

## Etapa 2: camada agêntica

```bash
python -m mover.agentes.treinar_especialista_ml           # IsolationForest -> data/modelos/especialista_ml.joblib
python -m mover.agentes.rodar_agentes --provedor falso    # log completo em data/agentes/log_agentes.jsonl
python -m mover.agentes.rodar_agentes --injetar-eventos   # soma a frenagem (75 s) e a arrancada (118 s) sintéticas
```

- Problema = jerk longitudinal: atenção a partir de 2,5 m/s³ e crítico a partir de 5,0 m/s³
  (`config/agentes.yaml`). Com pico vertical na janela, a causa é "provável lombada"; sem pico,
  "condução brusca".
- O log usa janelas de 5 s. Janelas seguidas no mesmo estado viram uma entrada só, até 20 s. Cada
  problema ganha a sua entrada.
- O Supervisor (LLM) redige cada entrada a partir dos números da janela. Uma guarda confere se cada
  número citado, com a sua unidade, existe nos fatos. Se a guarda recusar o texto, vale o texto-modelo.
- A volta real não tem condução brusca (os 7 eventos são lombadas). Por isso existe
  `--injetar-eventos`, que marca as linhas sintéticas com `fonte` diferente de `real` e não altera o
  CSV tratado.

## Etapas 3 e 4: replay no CARLA e página de 4 painéis

1. Copie o `mapa_final.xodr` do Eduardo para `data/mapas/` e confira o alinhamento sem abrir o CARLA:

   ```bash
   python -m mover.simulacao.verificar_mapa     # relatório e data/simulacao/alinhamento.png
   ```

2. Com o CARLA: abra o CarlaUE4 (0.9.16) e rode, num terminal só,

   ```bash
   python -m mover.simulacao.replay_carla --iniciar-servidor --manter-servidor
   ```

   ou, em dois terminais,

   ```bash
   python -m mover.servidor.rodar_servidor
   python -m mover.simulacao.replay_carla
   ```

3. Sem o CARLA, o painel 1 mostra uma visão 2D de perseguição no lugar da câmera:

   ```bash
   python -m mover.simulacao.replay_carla --sem-carla --iniciar-servidor --manter-servidor --provedor falso --injetar-eventos
   ```

4. Abra http://127.0.0.1:8000/. A documentação das rotas fica em http://127.0.0.1:8000/docs.

Opções úteis do replay:

- `--fator-tempo 3`: volta 3 vezes mais rápida.
- `--sem-espera`: ignora o relógio.
- `--camera perseguicao|cima|livre`: câmera do simulador.
- `--sem-camera-painel`: sem a câmera RGB do painel 1.
- `--mapa` e `--manter-mundo`: escolhem o mundo usado.
- `--offset-x`, `--offset-y` e `--rotacao`: ajuste manual do alinhamento.

O que a página mostra:

- **Simulação:** câmera de perseguição do CARLA (~10 quadros/s) ou a visão 2D, com velocidade,
  manobra e o próximo problema.
- **Dashboard:** velocidade e aceleração no tempo, com o instante atual, e o mapa da volta com os
  problemas. A previsão do próximo bloco aparece tracejada.
- **Log:** as entradas da camada agêntica, a previsão do próximo bloco e o resumo no fim da volta.
- **Chat:** perguntas sobre o log, respondidas com citações (#id, tempo) que levam à entrada. Usa o
  mesmo provedor de LLM, e uma guarda confere números e citações. Se o LLM falhar ou a resposta for
  recusada, vale uma resposta por busca no log.

## Ingestão ao vivo do Sensor Logger (HTTP Push)

O servidor também recebe o HTTP Push do Sensor Logger em `POST /ingestao/sensorlogger` e grava cada
gravação em `data/ingestao/<data>_<sessão>/` (`raw.jsonl` + um CSV por sensor). O andamento fica em
`GET /ingestao/status`.

1. Suba o servidor aceitando conexões da rede local:

   ```bash
   python -m mover.servidor.rodar_servidor --host 0.0.0.0
   ```

   Na máquina de campo, que só recebe os dados, basta o servidor de ingestão (não carrega numpy,
   scipy, agentes nem CARLA; escuta em 0.0.0.0 por padrão):

   ```bash
   python -m mover.ingestao.rodar_ingestao
   ```

   No Windows, libere a porta uma vez (PowerShell como administrador):
   `New-NetFirewallRule -DisplayName "MOVER 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any`

2. No celular, conectado à mesma rede (por exemplo, o notebook no hotspot do iPhone): Settings >
   Data Streaming > HTTP Push, Push URL `http://<IP do notebook>:8000/ingestao/sensorlogger`, batch
   period de 200 ms. "Tap to Test Pushing" deve responder 200. O IP sai do `ipconfig`.
3. Inicie a gravação e acompanhe o log do servidor ou `http://<IP>:8000/ingestao/status`.

Sem o celular, reenvie uma gravação já exportada no mesmo formato e no mesmo ritmo:

```bash
python -m mover.ingestao.replay_csv --duracao 30            # data/csv_maua, tempo real
python -m mover.ingestao.replay_csv --fator-tempo 5 --sensores location accelerometer gyroscope
```

- Token opcional: `MOVER_INGESTAO_TOKEN` no `.env`. A Push URL passa a terminar com `?token=<valor>`.
- A latência do status só faz sentido com os relógios do celular e do notebook sincronizados (NTP).
- Use apenas em rede local; não exponha a porta para a internet.

## Gêmeo ao vivo: estado do veículo em tempo real

Com a seção `gemeo` do `config/simulacao.yaml` ativa, cada mensagem recebida alimenta um EKF causal
(o mesmo modelo da Etapa 1, sem o suavizador RTS). Ele publica posição, rumo e velocidade a 20 Hz no
plano local e no plano do mapa/CARLA, grava `data/gemeo/<data>_<sessão>/estado_ao_vivo.csv` e
responde em `GET /gemeo/estado`. O fix do GPS chega atrasado; o filtro volta ao instante do fix e
repropaga até o presente. Na volta gravada, o resultado ao vivo fica a ~1,3 m (mediana) do
tratamento offline, e uma volta de 150 s é processada em ~0,3 s.

Calibração: a orientação do celular vem de uma **volta de calibração** já tratada pela Etapa 1
(`gemeo.calibracao`, padrão `data/tratado/relatorio_tratamento.json`). Ao trocar de celular ou de
suporte, grave uma volta (com o sensor Gravity ligado), exporte, rode o tratamento e aponte o YAML
para o novo relatório.

Topologia de campo (celular -> notebook -> VM):

```bash
# VM (rede da Mauá): servidor completo, com ingestão + gêmeo + painel
python -m mover.servidor.rodar_servidor --host 0.0.0.0

# Notebook no hotspot do celular, com a VPN da Mauá: recebe e repassa para a VM
python -m mover.ingestao.rodar_ingestao --repassar http://<IP da VM>:8000/ingestao/sensorlogger
```

O repasse roda em segundo plano (fila de 120 mensagens, descarta as mais antigas se a VM cair) e o
seu andamento aparece em `GET /saude` do notebook. Sem VM, `rodar_ingestao --gemeo` liga o
estimador no próprio notebook (precisa de numpy).

Fator de tempo real do CARLA (na VM, com o CarlaUE4 aberto): diz se o horizonte do gêmeo cabe no
ciclo de 1 s.

```bash
python -m mover.simulacao.benchmark_tempo_real --horizonte 10 --repeticoes 5
```

## Testes

```bash
python -m pytest tests -q     # a partir da raiz do projeto: 51 testes, sem CARLA, sem LLM e sem navegador
```

## Limitações conhecidas

- O replay não foi rodado num CARLA de verdade. As chamadas foram conferidas contra o pacote
  `carla` 0.9.16, e o lado CARLA (câmera incluída) foi testado com um módulo falso.
- Nenhum LLM real foi testado; só o provedor `falso` e, nos testes, um LLM de roteiro.
- A guarda confere números, unidades e citações. Ela não confere o sentido do texto.
- O IsolationForest é treinado e avaliado na mesma volta.
- Os eventos sintéticos mudam só a telemetria analisada, não a pose do caminhão.
- O mapa é plano. A margem de 0,5 m da faixa vale para o centro do caminhão.
- As vias do mapa são de mão única e foram desenhadas no sentido horário. A volta gravada é
  anti-horária, então o autopilot do Traffic Manager andaria na contramão; o replay não é afetado.
- A trajetória fundida (EKF) precisa de cerca de 1,9° de rotação para casar com o mapa.

## O que este código não altera

- `requirements.txt` e `.env.example` do repositório.
- O `coleta_carla.py`, o dashboard e a POC.
- O `modelo_especialista.joblib` da POC: o modelo retreinado fica em `data/modelos/especialista_ml.joblib`.
- O `.xodr` e os scripts do Eduardo, que só são lidos.
