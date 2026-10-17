# MOVER: gêmeo digital do caminhão autônomo com camada agêntica

Um caminhão autônomo dá a volta no campus do IMT, no CARLA, guiado pelo piloto automático do
simulador (Traffic Manager, TM). A cada 10 s, ele monta o plano do trecho seguinte a partir do
estado medido naquele instante e manda esse plano por HTTP a uma camada de agentes. Quando o plano
tem uma frenagem ou arrancada brusca, um modelo de ML propõe quanto suavizar, o modelo do caminhão
confere a proposta e um LLM local (gemma2:2b no Ollama) aprova e escreve o log. O caminhão então
executa o plano alterado.

O projeto também tem um painel ao vivo (simulação, gráficos, log e chat), um benchmark com N voltas
sem e N voltas com os agentes, e a gravação da corrida em MP4.

```
mapa do CARLA (campus_maua.xodr) -> plano de missão (config/missao.yaml)
          |
          v
caminhão no CARLA (TM) --- a cada 10 s: plano dos 10 s seguintes ---> camada agêntica (HTTP)
          ^                                                             jerk: acha as decisões bruscas
          |                                                             ML: propõe a suavização
          |                                                             modelo do caminhão: simula as opções
          +------------- velocidade desejada alterada <---------------- regra + LLM: aprovam e escrevem o log
```

Nenhum dado gravado no mundo real entra no laço. A rota sai do mapa do CARLA, e o plano que os
agentes alteram é sempre o da corrida atual.

## Resultado no CARLA

O benchmark rodou no CARLA 0.9.16, no mapa `campus_maua.xodr` (volta de 783 m, sentido
anti-horário), com o gemma2:2b e o modelo de ML. Foram 3 voltas sem agentes e 3 voltas com agentes,
na mesma execução. Os arquivos estão em
[`data/benchmark/20261015_004001`](data/benchmark/20261015_004001).

As métricas foram recalculadas em 17/10/2026 a partir das telemetrias salvas, com o medidor de
consumo unificado. Veja a seção "Como as métricas são calculadas".

| Por volta | Sem agentes | Com agentes | Diferença |
|---|---|---|---|
| Diesel | 336,2 mL | 267,4 mL | **−20,5 %** |
| Eventos bruscos (jerk ≥ 2,5 m/s³) | 24,3 | 7,3 | **−69,9 %** |
| Eventos críticos (jerk ≥ 5 m/s³) | 22,3 | 4,7 | **−79,1 %** |
| Pior desaceleração | −15,1 m/s² | −3,0 m/s² | |
| Energia dissipada nos freios | 586 Wh | 352 Wh | −40,0 % |
| Tempo | 166,2 s | 189,4 s | +14,0 % |

Extrapolando para 100 voltas, são 33,3 L contra 26,5 L de diesel: 6,8 L, R$ 40,59 e 18,1 kg de CO2
a menos. Os eventos bruscos caem de 2.304 para 604, ao custo de 38,5 min a mais.

![Velocidade na 1ª volta e diesel por volta, sem e com agentes](data/benchmark/20261015_004001/comparativo.png)

Como ler esses números:

- **De onde vem o ganho.** Com agentes, o caminhão freia antes e retoma com calma. Entre as zonas,
  ele fica num teto de 18 a 20 km/h: na volta com agentes, nunca passou de 20 km/h, e sem agentes
  chegou a 28,6 km/h. O custo é tempo, ~23 s a mais por volta.
- **Quem decide.** A proposta do ML, conferida pelo modelo do caminhão e pela regra, foi aplicada
  nas 31 decisões. O LLM aprovou as 31, em 3,6 a 5,1 s cada (o prazo é de 6 s), e escreveu os logs.
  Em 2 delas, a guarda recusou o texto do LLM, que citou um número sem respaldo, e valeu o texto da
  regra. O LLM não mudou nenhuma alteração.
- **Eventos que sobram.** Os eventos críticos que restam com agentes vêm do controle do TM:
  - acelerador cheio nas arrancadas, na largada e na saída do portão (que fica 3 m antes da
    lombada 2);
  - freio cheio nos últimos km/h das paradas;
  - um degrau de freio no começo de algumas frenagens, a ~16 km/h.

  O `python -m mover.autonomo.diagnostico` lista cada um.
- **Repetibilidade.** A bateria sem agentes sai idêntica entre execuções. A bateria com agentes
  varia pouco, porque o plano usa as ações do TM: em duas execuções no mapa anterior, o diesel
  diferiu 0,15 ponto percentual.
- **Robustez.** Com outras suavizações razoáveis da velocidade, a economia de diesel fica entre
  −16 % e −23 %. A contagem de eventos depende da suavização de 1,5 Hz da aceleração: com 2 Hz, a
  queda é de −36 %. Os eventos críticos, a pior desaceleração e o tempo de freio cheio caem em
  qualquer caso.

## Instalação (uma vez)

No Windows, como no setup do time (Python 3.12):

```bash
cd C:\CARLA_0.9.16\PythonAPI\TCC_MOVER_AGENTES_IA
py -3.12 -m venv env_carla
env_carla\Scripts\activate
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts\preparar_ollama.ps1
```

- **CARLA.** O `requirements.txt` já traz o cliente `carla==0.9.16`, que existe no PyPI para o
  Python 3.10 a 3.12 e tem que ser da mesma versão do simulador.
- **LLM.** O script do Ollama instala e liga o Ollama, baixa o `gemma2:2b`, mede a latência e grava
  `MOVER_LLM_PROVEDOR` e `MOVER_LLM_MODELO` no `.env`. O modelo do `.env` está no `.env.example`.
- **Linux.** Crie o ambiente com `python3 -m venv .venv` e
  `.venv/bin/pip install -r requirements.txt`, e rode `bash scripts/preparar_ollama.sh`.

## Como rodar

Primeiro, abra o CARLA. Para ver a corrida, abra com janela:

```bash
C:\CARLA_0.9.16\CarlaUE4.exe -quality-level=Low -world-port=2000
```

Para gravar vídeo, use `-quality-level=Epic`. No benchmark, pode abrir sem janela:
`-RenderOffScreen -quality-level=Low -nosound`.

Os comandos `python -m ...` rodam da pasta `src/`. O `scripts/rodar_gemeo.sh` roda da raiz (no
Windows, pelo Git Bash): ele confere o CARLA e o Ollama antes e repassa as opções extras ao Python.

| Tarefa | Python (pasta `src/`) | Script (raiz) |
|---|---|---|
| Volta com agentes no CARLA, com o painel em http://127.0.0.1:8000/ | `python -m mover.autonomo.dirigir` | `./scripts/rodar_gemeo.sh visual` |
| Só coletar os dados de uma volta (sem painel) | `python -m mover.autonomo.dirigir --sem-painel` | `./scripts/rodar_gemeo.sh coleta` |
| Painel sem o CARLA (simulador substituto) | `python -m mover.autonomo.dirigir --sem-carla --provedor falso` | `./scripts/rodar_gemeo.sh painel` |
| Benchmark: N voltas sem e N com agentes | `python -m mover.autonomo.benchmark --voltas 3` | `./scripts/rodar_gemeo.sh benchmark --voltas 3` |
| Refazer as métricas de um benchmark salvo | `python -m mover.autonomo.benchmark --recalcular ../data/benchmark/<data>` | — |
| Diagnóstico de uma corrida (trechos lentos e eventos bruscos) | `python -m mover.autonomo.diagnostico [telemetria.csv]` | — |
| Gravar a volta em MP4 | `python -m mover.autonomo.dirigir --gravar --sair-no-fim` e `python -m mover.video.gravar_video` | `./scripts/rodar_gemeo.sh gravar` |
| Servidor do painel e dos agentes à parte | `python -m mover.servidor.rodar_servidor` | `./scripts/rodar_gemeo.sh servidor` |
| Treinar de novo o modelo de ML | `python -m mover.agentes.suavizador_ml` | `./scripts/rodar_gemeo.sh treinar` |
| Análise de uma corrida pela camada de logs | `python -m mover.agentes.rodar_agentes` | `./scripts/rodar_gemeo.sh analise` |
| Testes (da raiz) | `python -m pytest tests -q` | `./scripts/rodar_gemeo.sh testes` |

Opções do `dirigir`:

| Opção | O que faz |
|---|---|
| `--voltas N` | N voltas seguidas |
| `--sem-agentes` | o caminhão sozinho, executando o próprio plano |
| `--sem-ml` | os agentes só com as opções fixas |
| `--sem-llm` | os agentes decidem só pela regra |
| `--provedor falso` | sem LLM instalado (usa um texto-modelo) |
| `--sem-render` | o CARLA não desenha a cena |
| `--sem-espera` | não segue o relógio |
| `--servidor http://<ip>:8000` | usa um servidor de agentes já no ar |
| `--camera cima` ou `--camera livre` | muda a câmera da janela do CARLA |
| `--fator-tempo 2` | acelera o relógio |

No `gravar_video`, as opções são `--camera frontal`, `--corrida <pasta>`, `--modo poses` (se a
reprodução do gravador falhar), `--inicio 30 --duracao 60` e `--sem-hud`.

Na primeira corrida com agentes, se faltar `data/modelos/suavizador_ml.joblib`, o modelo de ML é
treinado sozinho (~1 a 2 min).

### Sequência usada na VM para o resultado acima

São dois terminais, os dois na mesma pasta do repositório. O servidor recusa, com HTTP 409, um
caminhão que tenha outro `config/missao.yaml`.

```bash
# Terminal A: servidor do painel e dos agentes
cd src
python -m mover.servidor.rodar_servidor --depurar

# Terminal B: o caminhão
cd src
python -m mover.autonomo.dirigir --sem-painel --sem-render --sem-espera --provedor falso   # volta de conferência
python -m mover.autonomo.benchmark --voltas 3 --servidor http://127.0.0.1:8000 --provedor ollama
python -m mover.autonomo.diagnostico
```

O passo a passo completo da VM está em [`EXECUCAO_VM.md`](EXECUCAO_VM.md).

## Saídas

- **Corrida:** fica em `data/autonomo/corridas/<data>_<agentes|bruto>/`, com:
  - `telemetria.csv`: um tick por linha, a 20 Hz, com posição, velocidade, aceleração, pedais,
    marcha, velocidade desejada, zona que limita e consumo acumulado;
  - `decisoes.jsonl`: para cada decisão, o plano enviado, as propostas do ML, as opções simuladas,
    a escolhida e o log;
  - `resumo.json` e `linha_de_base.json`: o plano sem agentes previsto, usado pelo painel e pelo
    vídeo;
  - `video.mp4`, quando a corrida for gravada.
- **Benchmark:** fica em `data/benchmark/<data>/`, com:
  - `resultados.csv`: uma linha por volta, as médias, as diferenças e o total em 100 voltas;
  - `resumo.json` e `comparativo.png`;
  - as duas telemetrias e o `decisoes.jsonl`.

## Como a camada agêntica decide

1. **O caminhão planeja** (`autonomo/planejador.py`). No começo de cada bloco de 10 s, a partir do
   estado medido, ele monta o plano do bloco seguinte: a velocidade desejada em cada zona, a
   velocidade e a aceleração que isso gera (previstas por um modelo do controle do TM), as manobras
   à frente (`get_all_actions` do TM) e as decisões bruscas, com |jerk| ≥ 2,5 m/s³.
2. **O ML propõe** (`agentes/suavizador_ml.py`). Uma floresta aleatória (200 árvores, 3 saídas),
   treinada com 2.000 situações simuladas, propõe três valores para cada zona com decisão brusca:
   - a desaceleração antes da zona (0,4 a 1,8 m/s²);
   - a aceleração na retomada (0,3 a 1,2 m/s²);
   - o teto de velocidade depois da zona.
3. **O modelo do caminhão simula** (`agentes/decisao.py`). Ele simula o plano sem alteração, a
   proposta do ML e as opções fixas: econômica (0,6 / 0,5 m/s², teto de 22 km/h), suave
   (0,8 / 0,6), moderada (1,2 / 0,9) e firme (1,8 / 1,2). Para cada uma, mede diesel, tempo e
   eventos bruscos.
4. **A regra escolhe.** Sem decisão brusca no plano, nada muda. Com decisão brusca, vale a proposta
   do ML, desde que ela não tenha mais eventos que a melhor opção fixa nem passe de +60 % de tempo
   no trecho. Senão, vale a opção fixa de menor custo (diesel em R$ + tempo × R$ 30/h).
5. **O LLM supervisiona.** Com prazo de 6 s, o LLM aprova ou troca entre as opções já simuladas e
   escreve o log. Uma guarda recusa opções com mais eventos bruscos e números sem respaldo. O LLM
   nunca mexe no pedal.
6. **O governador executa** (`autonomo/governador.py`). Ele transforma a escolha num envelope de
   velocidade com jerk limitado a 1,6 m/s³ e manda a velocidade desejada ao TM a cada tick
   (`set_desired_speed`). Sem agentes, o caminhão segue o plano em degrau e só começa a frear a
   4,5 m/s² da zona (`governador.desacel_bruta_mps2`).

O buffer de 10 s é o `decisao.horizonte_s` do `config/autonomo.yaml`. Para diminuí-lo, baixe junto o
`decisao.prazo_llm_s` e o `blocos.duracao_s` do `config/agentes.yaml`.

### Comunicação por HTTP

O caminhão (`dirigir` ou `benchmark`) e os agentes conversam como um caminhão real ligado a um
servidor. O código fica em `autonomo/canal_agentes.py` e em `agentes/servico.py`, e as rotas estão
documentadas em `/docs`.

| Rota | Quem chama | O que vai e volta |
|---|---|---|
| `POST /agentes/sessao` | o caminhão, antes da largada | voltas, LLM e ML; o servidor confere o plano de missão, aquece o LLM e treina o ML se faltar |
| `POST /agentes/plano` | o caminhão, a cada bloco | o plano dos próximos 10 s; responde 202, ou as decisões na hora com `"esperar": true` (largada e benchmark) |
| `GET /agentes/decisoes?desde=N` | o caminhão | as mensagens `regra` (aplicada na hora), `final` (do LLM, com o log), `vazia` ou `erro` |
| `DELETE /agentes/sessao` | o caminhão, no fim | encerra a sessão |
| `POST /sessao/telemetria` e `POST /sessao/decisoes` | o caminhão, a cada 0,5 s | dados para o painel |

Ao vivo, o caminhão não espera os agentes. No benchmark, a simulação espera cada decisão, e o
resultado não depende da velocidade do computador. Para separar as máquinas, rode
`rodar_servidor --host 0.0.0.0` numa e `--servidor http://<ip>:8000` na outra. As rotas não têm
senha: use só em rede confiável.

## Como as métricas são calculadas

- **Diesel, custo e CO2.** O CARLA não simula combustível. O consumo vem do modelo VT-CPFM (Rakha
  et al.):

  ```
  consumo (L/s) = marcha lenta + a1 · P · (1 + P / P2),   com P = max(0, P_rodas) / 0,88
  ```

  - **Potência nas rodas.** O `P_rodas` sai da variação da energia cinética da velocidade medida,
    mais o arrasto, o rolamento e a rampa (pelo *pitch* do caminhão). Antes, a velocidade é
    suavizada: uma média de dois ticks zera o tremor de 10 Hz do CARLA, e uma média móvel
    exponencial de 0,16 s dá um corte de ~1 Hz.
  - **Um só medidor.** O painel ao vivo, a coluna `combustivel_ml` da telemetria, o `resumo.json`
    da corrida e o benchmark usam esse mesmo cálculo (`MedidorConsumo` em `autonomo/consumo.py`).
  - **Parâmetros** (`config/autonomo.yaml`, seções `veiculo` e `consumo`):
    - caminhão de 9 t (o `european_hgv` do CARLA tem 7 t na física);
    - BSFC de 225 g/kWh e diesel a 0,835 kg/L;
    - marcha lenta de 2 L/h;
    - P2 de 1.000 kW, uma suposição;
    - R$ 6,00/L e 2,68 kg de CO2 por litro.
- **Por que o diesel mudou de −27 % para −20,5 %.** Até 15/10/2026, o benchmark suavizava a
  aceleração com um Butterworth de fase zero. Esse filtro oscila antes das freadas bruscas e somava
  consumo que não existiu à bateria sem agentes. Além disso, o painel usava outro filtro e mostrava
  outro número. O recálculo corrigiu as duas coisas; tempo e eventos não mudaram.
- **Energia nos freios.** É a parte negativa da mesma potência.
- **Eventos bruscos e pior desaceleração.** Usam a aceleração longitudinal suavizada em 1,5 Hz
  (Butterworth de 4ª ordem, fase zero), a mesma convenção da camada de logs. Um evento é de
  atenção a partir de |jerk| ≥ 2,5 m/s³ e crítico a partir de ≥ 5 m/s³. Picos a menos de 1 s um do
  outro contam como um evento só.
- **Total em 100 voltas.** Soma a 1ª volta (saída parada), a última (parada final) e 98 vezes a
  média das voltas do meio. Por isso o benchmark pede pelo menos 3 voltas.
- **`--recalcular`.** Refaz o `resultados.csv`, o `resumo.json` e o `comparativo.png` a partir das
  telemetrias salvas, sem CARLA. As telemetrias não mudam. Em `20261015_004001`, as colunas
  `combustivel_ml`, `custo_r` e `co2_g` ainda são as do medidor usado na corrida.

## Plano de missão e configuração

O `config/missao.yaml` define a volta de 783 m no anel de mão única do `campus_maua.xodr` (R01,
R10+R07, R02 e R11, sentido anti-horário), com cruzeiro de 30 km/h e estas zonas:

- 4 lombadas a 10 km/h;
- 8 faixas de pedestre a 15 km/h;
- 3 curvas, limitadas a 1,6 m/s² de aceleração lateral;
- a parada de 4 s no portão, a 422 m da largada.

As lombadas e as faixas são objetos do mapa, e o portão vem do OSM do campus. Para gerar o plano de
novo, rode `python -m mover.autonomo.gerar_missao`. Para ver os ids das vias, rode
`python -m mover.autonomo.rota_mapa --figura vias.png`.

| Arquivo | O que tem |
|---|---|
| `config/autonomo.yaml` | missão, caminhão, piloto, governador, opções fixas, decisão (buffer, prazo do LLM, valor do tempo), modelo de ML, consumo, limiares, benchmark e CARLA (porta 8100 do TM, clima, velocidade mínima de 5 km/h no TM) |
| `config/missao.yaml` | rota, zonas e paradas (gerado pelo `gerar_missao`) |
| `config/simulacao.yaml` | mapa, conexão com o CARLA (`mapa_carla` carrega um mapa instalado pelo nome), caminhão, câmeras e servidor |
| `config/agentes.yaml` | LLM, camada de logs, chat |
| `.env` | `MOVER_LLM_PROVEDOR`, `MOVER_LLM_MODELO` e `OLLAMA_HOST` |

## Estrutura

```
config/        autonomo.yaml, missao.yaml (+ missao.png), simulacao.yaml, agentes.yaml
scripts/       rodar_gemeo.sh, preparar_ollama.ps1, preparar_ollama.sh
src/mover/
  autonomo/    planejador, governador, consumo, conducao (laço), carla_tm (CARLA),
               simulador (substituto), canal_agentes, dirigir, benchmark, diagnostico,
               rota, rota_mapa, missao, gerar_missao, veiculo, previsao, montagem
  agentes/     decisao, servico (rotas dos agentes), suavizador_ml, supervisor, grafo,
               rodar_agentes, llm
  servidor/    app (FastAPI + WebSocket), sessao, rodar_servidor
  interface/   rotas, chat e estatico/ (página, CSS e JS)
  video/       hud, gravar_video
  simulacao/   opendrive, projecao, camera_painel, fila_envio
  tratamento/  legado (só o sinais.py é usado)
tests/         115 testes
data/          mapas/ (campus_maua.xodr; mapa_final.xodr, o anterior), modelos/,
               autonomo/corridas/, benchmark/20261015_004001 (o resultado do TCC)
```

## Testes

Rode `python -m pytest tests -q` da raiz do repositório. São 115 testes, sem CARLA e sem LLM de
verdade: o lado CARLA usa um módulo `carla` falso, e o LLM usa o provedor falso e um LLM de
roteiro.

Algumas situações fazem testes serem pulados:

- no Windows sem bash, os 16 do `rodar_gemeo.sh`;
- sem o `pyproj` e sem a gravação antiga, mais 2 (legado).

Os testes cobrem:

- o plano a partir do estado atual;
- o ML e a conferência das propostas;
- a regra e o LLM dentro das opções seguras;
- a corrida e o benchmark no substituto;
- o HTTP entre caminhão e agentes;
- a rota e os objetos do mapa;
- o fim da corrida com motivo;
- o servidor, o chat e o vídeo;
- o medidor de consumo (`tests/test_consumo.py`): nada somado antes de uma freada, a energia
  inteira de cada frenagem, o tremor ignorado, o mesmo número no painel e no benchmark, e o
  `--recalcular`.

## Limitações

- **O TM não é aprendizado de máquina.** São regras e PID, e ele não expõe a velocidade nem a
  aceleração que vai aplicar. O plano enviado aos agentes é uma previsão do modelo do caminhão; o
  que vale é o medido no CARLA.
- **O ML aprendeu no modelo do caminhão, não no CARLA.**
  - Se mudar `veiculo`, `piloto` ou o valor do tempo, treine de novo.
  - O modelo do repositório foi treinado em 14/10/2026 (R² de 0,45 / 0,88 / 0,98 e 100 % das
    propostas sem eventos no teste) e salvo com o scikit-learn 1.7.2. Com outra versão, o
    carregamento avisa; instale a 1.7.2 ou treine de novo.
- **O consumo é estimado.** A economia de diesel fica entre −16 % e −23 % conforme a suavização, e
  o modelo usa 9 t, enquanto o caminhão do CARLA tem 7 t.
- **A contagem de eventos depende do corte de 1,5 Hz.** Com acelerador parcial, o caminhão do CARLA
  oscila de leve. Com cortes acima de ~2,5 Hz, essas oscilações viram eventos de atenção, e a
  bateria com agentes empata ou passa a sem agentes nesse critério.
- **A física do CARLA não é a de um caminhão real.**
  - Sem agentes, as freadas chegam a −16,7 m/s² (1,7 g), e as arrancadas a ~3 m/s².
  - Com a velocidade desejada em 0, o freio do TM vem NaN (o PID divide pelo alvo), e o caminhão
    para seco.
- **O ganho depende da linha de base e do teto de velocidade.** Sem agentes, o caminhão freia tarde
  por projeto (4,5 m/s²). Com agentes, parte da economia vem de andar mais devagar entre as zonas.
- **Ainda não há ablações no CARLA.** Faltam rodar `--sem-ml` e `--sem-llm`. No benchmark, o LLM
  não mudou nenhuma decisão.
- **A latência do LLM depende da máquina.** Na VM, foi de 3,6 a 5,1 s. Sem GPU, pode passar dos
  6 s, e então vale a regra (o log mostra o motivo).
- **O mapa usado não fica registrado.** O `resumo.json` não diz se o mundo veio do `.xodr` ou do
  pacote instalado (`mapa_carla`).

### Problemas resolvidos para rodar no CARLA

- **Altura das vias.** As vias ficam a ~750 m de altitude, então os pontos da largada e do caminho
  do TM levam o z da via mais próxima.
- **Caminho do TM.** O caminho (`set_path`) vai de uma vez, da largada até 30 m depois da chegada,
  e sem pontos dentro das junções. Renová-lo no meio da corrida, ou deixar pontos nas junções,
  fazia o TM trocar de faixa.
- **Velocidade mínima no TM.** Quando o governador quer andar, o TM recebe pelo menos 5 km/h; com
  pedidos de 1 a 3 km/h, o caminhão rastejava. Na arrancada, o pedido é a velocidade atual
  + 5 km/h.
- **Fim da corrida com motivo.** A corrida termina como `preso` (30 s parado), `fora_da_rota`
  (mais de 12 m por 3 s), `removido` ou `sem_resposta` (um tick que não volta em 90 s). Em todos os
  casos, ela grava o que andou.
- **Portas.** O TM usa a porta 8100, porque a 8000 é do painel.

## Legado

O código das etapas antigas, que usavam a volta gravada com o celular (`src/mover/tratamento/`,
`simulacao/alinhamento.py`, `simulacao/verificar_mapa.py` e `config/tratamento.yaml`), continua no
repositório, mas nada do fluxo acima o usa. A exceção é o `tratamento/sinais.py`, com os filtros.
