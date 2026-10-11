# MOVER: gêmeo digital do caminhão autônomo com camada agêntica

O caminhão já é autônomo: ele dá a volta no campus do IMT sozinho, no CARLA, guiado pelo piloto
automático do simulador (o Traffic Manager). Como todo carro autônomo, ele decide antes de agir. A
camada agêntica entra entre a decisão e a execução, sempre com os dados da corrida atual:

**1. O caminhão planeja (X).** A cada 10 s, a partir do estado medido naquele instante (posição, velocidade e aceleração no CARLA), o caminhão monta o plano do trecho seguinte: a velocidade desejada que vai pedir ao piloto em cada zona, a velocidade e a aceleração que isso vai gerar, as manobras à frente (no CARLA, as do próprio Traffic Manager, por `get_all_actions`) e as frenagens e arrancadas bruscas desse plano. Ele manda esse plano aos agentes antes de executá-lo.

**2. Os agentes alteram o plano.** Quando o plano tem uma decisão brusca, um modelo de ML (floresta aleatória treinada com milhares de situações simuladas) propõe quanto suavizar: com que desaceleração frear antes, com que aceleração retomar e até que velocidade. O modelo do caminhão simula a proposta e as opções fixas, a regra confere e o LLM local (gemma2:2b no Ollama) aprova ou troca a alteração e escreve o log em linguagem natural.

**3. O caminhão executa o plano alterado (Y).** A alteração muda de verdade a velocidade desejada mandada ao piloto automático, a cada tick. Enquanto anda, o caminhão já planeja o trecho seguinte e o manda aos agentes, e assim por diante: há sempre um buffer de 10 s entre a decisão e a execução.

A página tem quatro painéis: a simulação, o dashboard com gráficos que se montam ao vivo e o
consumo comparado com o plano sem agentes, o log e o chat. Um benchmark mede o efeito prático em
N voltas sem e N voltas com os agentes (diesel, R$, CO2, eventos bruscos e tempo, extrapolados
para 100 voltas). Um script grava a corrida em MP4.

```
mapa do CARLA (mapa_final.xodr) + OSM do campus + config
   |  rota, lombadas, faixa de pedestre, curvas e portaria
   v
plano de missão (config/missao.yaml)
   |
   v
caminhão autônomo no CARLA (Traffic Manager) <----------------+
   |  a cada 10 s, a partir do estado medido agora: o plano    |
   |  dos 10 s seguintes (velocidade, aceleração, manobras     |
   |  e decisões bruscas), ainda não executado                 |
   v                                                           |
camada agêntica                                                |
   especialista de jerk: aponta as decisões bruscas do plano   |
   especialista de ML: propõe quanto suavizar cada uma         |
   modelo do caminhão: simula a proposta e as opções fixas     |
   regra + LLM (gemma2:2b): aprova e escreve o log             |
   |                                                           |
   +--- plano alterado: velocidade desejada a cada tick -------+
   |
   v
painel ao vivo (gráficos, consumo x plano sem agentes, log e chat),
benchmark (N voltas sem x N com) e vídeo MP4
```

> Nenhum dado gravado no mundo real entra no projeto, e nenhuma corrida anterior entra no laço
> de decisão: o plano que os agentes alteram é o da corrida atual. A rota sai do mapa do CARLA. O
> código antigo, da gravação feita com o celular, ficou no repositório só como legado (seção
> Legado, no fim).

## Comandos de cada tarefa

Os comandos `python -m ...` rodam a partir da pasta `src/`. O script `rodar_gemeo.sh` roda da raiz
do repositório (no Windows, pelo Git Bash) e faz o mesmo, conferindo antes o CARLA e o Ollama.

| Tarefa | Com o script | Direto no Python (pasta `src/`) |
|---|---|---|
| Coletar os dados de uma volta no CARLA (sem o painel) | `./scripts/rodar_gemeo.sh coleta` | `python -m mover.autonomo.dirigir --sem-painel` |
| Ver a volta com os agentes no CARLA, com o painel | `./scripts/rodar_gemeo.sh visual` | `python -m mover.autonomo.dirigir` |
| Gravar a volta em vídeo MP4 | `./scripts/rodar_gemeo.sh gravar` | `python -m mover.autonomo.dirigir --gravar --sair-no-fim` e depois `python -m mover.video.gravar_video` |
| Interface no localhost, sem o CARLA | `./scripts/rodar_gemeo.sh painel` | `python -m mover.autonomo.dirigir --sem-carla` |
| Benchmark sem e com os agentes | `./scripts/rodar_gemeo.sh benchmark --voltas 3` | `python -m mover.autonomo.benchmark --voltas 3` |
| Treinar de novo o modelo de ML | `./scripts/rodar_gemeo.sh treinar` | `python -m mover.agentes.suavizador_ml` |
| Análise de uma corrida pela camada de logs, sem o CARLA | `./scripts/rodar_gemeo.sh analise` | `python -m mover.agentes.rodar_agentes` |

A ordem é: abrir o CARLA (seção 1) e rodar a tarefa. Não há coleta prévia: cada corrida grava os
próprios dados. Na primeira corrida com agentes, o modelo de ML é treinado sozinho (uma vez, ~1 a 2
min; seção 7). As seções 1 a 8 explicam cada tarefa. Antes, faça a instalação abaixo uma vez.

## Instalação (uma vez, no Windows como no setup do time)

### Ambiente Python e CARLA

```bash
cd C:\CARLA_0.9.16\PythonAPI\TCC_MOVER_AGENTES_IA
py -3.12 -m venv env_carla
env_carla\Scripts\activate
pip install -r requirements.txt
```

O `requirements.txt` já traz o cliente do CARLA (`carla==0.9.16`, do PyPI), que precisa ser da mesma versão do simulador. O PyPI tem esse pacote para o Python 3.10, 3.11 e 3.12. Outra forma é instalar o arquivo que vem com o simulador: `pip install C:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl`.

### LLM local (Ollama + gemma2:2b)

```bash
powershell -ExecutionPolicy Bypass -File scripts\preparar_ollama.ps1
```

O script instala o Ollama pelo winget (se faltar), liga o serviço e baixa o `gemma2:2b`. Esse é o
modelo escolhido no benchmark do artigo: 25 % de alucinação e ~59 tokens/s. Depois ele mede a
latência e grava `MOVER_LLM_PROVEDOR=ollama` e `MOVER_LLM_MODELO=gemma2:2b` no `.env` da raiz (o
modelo do arquivo está em `.env.example`). No Linux ou macOS, use `bash scripts/preparar_ollama.sh`;
no Git Bash, `./scripts/rodar_gemeo.sh ollama`. Outro modelo:
`scripts\preparar_ollama.ps1 -Modelo qwen2.5:1.5b`.

O piloto aquece o modelo antes da largada, porque a primeira chamada carrega o modelo na memória.
Cada decisão tem prazo de 6 s (`decisao.prazo_llm_s`). Se o LLM atrasar, falhar, escolher uma
opção que não existe ou citar um número sem respaldo, vale a decisão da regra, e o log registra o
motivo.

### O script rodar_gemeo.sh

No Windows ele roda no Git Bash, que vem com o Git for Windows. Ele acha sozinho o Python do `env_carla` (ou do `.venv`), confere se o CARLA e o Ollama estão no ar e entra na pasta `src/`. Se o CARLA estiver fechado e `CARLA_ROOT` apontar para a pasta dele (no Windows, `C:\CARLA_0.9.16` é achado sozinho), o script abre o CARLA com janela e espera ele responder.

| Modo | O que faz |
|---|---|
| `coleta` | uma volta com os agentes no CARLA, sem o painel: grava a telemetria, as decisões e o plano de cada bloco |
| `visual` (padrão) | a volta com os agentes no CARLA, vendo o caminhão, com o painel no navegador |
| `gravar` | grava a volta com os agentes no CARLA e gera o vídeo MP4 |
| `video` | refaz o MP4 da última corrida gravada |
| `painel` | o painel em http://127.0.0.1:8000/ com o simulador substituto, sem o CARLA |
| `servidor` | só o servidor do painel, para várias corridas |
| `benchmark` | N voltas sem e N com os agentes |
| `treinar` | treina de novo o modelo de ML que propõe a suavização |
| `analise` | a camada de logs e o especialista de ML dos logs sobre a corrida mais recente, sem o CARLA |
| `ollama` | instala e liga o Ollama e baixa o gemma2:2b |
| `carla` | só abre o CARLA com janela |
| `testes` | roda os testes |
| `ajuda` | mostra os modos e as variáveis |

Opções a mais vão direto para o comando Python: `./scripts/rodar_gemeo.sh visual --voltas 2 --sem-llm` ou `./scripts/rodar_gemeo.sh coleta --sem-agentes`. Variáveis de ambiente, todas opcionais:

- `CARLA_ROOT`: pasta do CARLA, para o script abri-lo (ex.: `CARLA_ROOT=/c/CARLA_0.9.16`);
- `CARLA_QUALIDADE`: `Low` ou `Epic` (padrão: `Epic` em `gravar` e `video`, `Low` no resto);
- `CARLA_HOST` e `CARLA_PORTA`: padrão `localhost` e `2000`, os mesmos do `config/simulacao.yaml`;
- `PYTHON`: outro Python;
- `OLLAMA_HOST`: padrão `http://localhost:11434`;
- `MOVER_SECO=1`: só mostra os comandos, sem rodar.

Se o Git não marcar o script como executável, rode `bash scripts/rodar_gemeo.sh visual` ou, uma vez, `git update-index --chmod=+x scripts/rodar_gemeo.sh`. O `.gitattributes` mantém o fim de linha LF nos `.sh`, sem o qual o bash não roda o script.

## 1. Abrir o CARLA com janela

O setup do time abria o CARLA com `-RenderOffScreen`. Com essa opção não há janela, então não dá
para ver o mapa nem o caminhão. Para ver a corrida, abra assim:

```bash
C:\CARLA_0.9.16\CarlaUE4.exe -quality-level=Low -world-port=2000
```

Para gravar o vídeo, troque por `-quality-level=Epic`: é essa opção que define a qualidade da imagem das câmeras. Com o script, `./scripts/rodar_gemeo.sh carla` abre o CARLA do mesmo jeito.

Os comandos geram o mundo a partir de `data/mapas/mapa_final.xodr` e criam o caminhão `vehicle.carlamotors.european_hgv` na largada. A câmera da janela (spectator) passa a seguir o caminhão. Use `--camera cima` para ver de cima ou `--camera livre` para mexer a câmera você mesmo.

## 2. Coletar os dados de uma volta no CARLA

Com o CARLA aberto:

```bash
cd src
python -m mover.autonomo.dirigir --sem-painel
```

O caminhão sai parado da largada e dá a volta com os agentes, sem abrir a página. Os dados saem desta corrida: a cada 10 s, o caminhão manda aos agentes o plano do trecho seguinte, e os agentes o alteram antes de ele ser executado. Para ter também os dados do caminhão sozinho, rode de novo com `--sem-agentes`. Com o script: `./scripts/rodar_gemeo.sh coleta` (ou `coleta --sem-agentes`).

| Opção | O que faz |
|---|---|
| `--voltas 3` | três voltas seguidas (o padrão vem de `missao.voltas` no `config/autonomo.yaml`) |
| `--sem-agentes` | o caminhão sozinho, executando o próprio plano sem alteração |
| `--sem-ml` | os agentes só com as opções fixas de suavização, sem o modelo de ML |
| `--sem-llm` | os agentes decidem só pela regra, sem o Ollama |
| `--sem-render` | o CARLA não desenha a cena (mais rápido) |
| `--sem-espera` | não segue o relógio: o mais rápido possível, e a simulação espera cada decisão |
| `--sem-carla` | simulador substituto, só para testar sem o CARLA |

Cada corrida fica em `data/autonomo/corridas/<data>_<modo>/`, com o modo `agentes` ou `bruto` (o caminhão sozinho):

- `telemetria.csv`: uma linha por tick do simulador (20 Hz);
- `decisoes.jsonl`: uma linha por decisão dos agentes, com o plano que o caminhão mandou (velocidade, aceleração e velocidade desejada a 2 Hz, decisões bruscas e manobras), as propostas do ML, as alterações simuladas, a escolhida e o texto do log;
- `resumo.json`: tempo, distância, diesel, custo, CO2, eventos bruscos e jerk, da corrida e de cada volta;
- `linha_de_base.json`: o plano do caminhão sem agentes, previsto pelo modelo do caminhão (consumo e eventos pela distância), para o painel e o vídeo compararem.

Colunas do `telemetria.csv`:

- tempo: `frame`, `sim_time`, `t` e `hora_local`;
- posição no plano local do mapa: `x`, `y`, `z` e `rumo_graus`; no CARLA, também `x_carla`, `y_carla`, `yaw_carla`, `pitch_carla` e `roll_carla`;
- movimento: `speed_mps`, `speed_kmh`, `acc_long`, `acc_lat`, `yaw_rate_dps` e `rampa`;
- comandos do piloto automático: `throttle`, `brake`, `steer`, `gear` e `rpm` (o `rpm` só com `telemetria_motor: true` no `config/autonomo.yaml`);
- rota: `d` e `odom_m` (distância percorrida), `s` (posição na volta), `volta` e, no CARLA, `desvio_m` (distância lateral até a rota);
- velocidades: `v_desejada_kmh` (a mandada ao Traffic Manager), `v_limite_kmh` (a do plano), `v_alvo_kmh` e `limitado_por` (a zona ou parada que está limitando);
- modo: `modo`, `bloco` e `modo_tick` (`bruto` quando o piloto segue o degrau do plano, `suave` quando vale uma alteração, `parado`);
- consumo acumulado: `combustivel_ml`, `custo_r` e `co2_g`;
- `fonte`: `carla` ou `substituto`.

O terminal mostra cada decisão. No simulador substituto, que serve só para testes, uma delas foi:

```
[  10.0 s] decisão do bloco 02: ajuste do ML (LLM, 0.2 s) - O caminhão
planeja frenagem brusca para a curva 1 em 20,7 s (-2,9 m/s², jerk
-8,7 m/s³) e mais 2 evento(s). O ML suavizou: frear com até 0,6 m/s² e
retomar com até 0,5 m/s², até 19 km/h; 0 evento(s) bruscos e 21 mL a
menos (+1,2 s).
```

## 3. Ver a volta com os agentes no CARLA

Com o CARLA aberto com janela:

```bash
cd src
python -m mover.autonomo.dirigir
```

O comando dá a volta com os agentes. A câmera da janela do CARLA segue o caminhão, e o comando também sobe o painel e abre http://127.0.0.1:8000/ no navegador. Com o script: `./scripts/rodar_gemeo.sh visual`. Outras opções:

| Comando | O que faz |
|---|---|
| `python -m mover.autonomo.dirigir --voltas 3` | três voltas seguidas |
| `python -m mover.autonomo.dirigir --sem-agentes` | o caminhão sozinho, para comparar no painel |
| `python -m mover.autonomo.dirigir --sem-ml` | os agentes sem o modelo de ML (só as opções fixas) |
| `python -m mover.autonomo.dirigir --sem-llm` | os agentes decidem só pela regra |
| `python -m mover.autonomo.dirigir --camera cima` | a janela do CARLA vê de cima |
| `python -m mover.autonomo.dirigir --fator-tempo 2` | duas vezes mais rápido que o real |
| `python -m mover.autonomo.dirigir --provedor falso` | sem LLM instalado (texto-modelo) |
| `python -m mover.autonomo.dirigir --sair-no-fim` | fecha o painel quando a volta acaba |
| `python -m mover.autonomo.dirigir --sem-painel` | sem a página (só grava os dados da corrida) |

A corrida grava os mesmos arquivos da seção 2. No fim, a página e o chat continuam no ar até o Ctrl+C.

## 4. Gravar a volta em vídeo

Com o CARLA aberto com `-quality-level=Epic`:

```bash
cd src
python -m mover.autonomo.dirigir --gravar --sair-no-fim
python -m mover.video.gravar_video
```

O primeiro comando dá a volta com os agentes e liga o gravador do CARLA (`gravacao_carla.log`, na pasta da corrida). O segundo reproduz essa gravação no CARLA em modo síncrono, com passo de 1/30 s, e segue o caminhão com uma câmera de perseguição de 1920 x 1080. Sem `--corrida`, ele usa a última corrida gravada. O vídeo sai em `data/autonomo/corridas/<data>_agentes/video.mp4`. Com o script, `./scripts/rodar_gemeo.sh gravar` faz os dois passos.

O HUD do vídeo mostra:

- a velocidade, com a desejada e o limite;
- a aceleração;
- o diesel e o custo, comparados com o plano sem agentes (previsto) na mesma distância;
- a decisão dos agentes no momento;
- um minimapa com o rastro.

O ffmpeg do `imageio-ffmpeg` codifica em H.264 (libx264, CRF 18, yuv420p). Cada quadro é renderizado pelo CARLA, então o vídeo demora mais que a corrida, conforme a GPU. Opções do `gravar_video` (no script, `./scripts/rodar_gemeo.sh video ...`):

| Opção | O que faz |
|---|---|
| `--camera frontal` | câmera no capô (também há `alto`) |
| `--corrida ../data/autonomo/corridas/<id>` | outra corrida, em vez da última gravada |
| `--modo poses` | se a reprodução falhar, impõe as poses gravadas na telemetria |
| `--inicio 30 --duracao 60` | só um trecho da corrida |
| `--sem-hud` | vídeo limpo, sem o HUD |
| `--teste-hud hud.png` | desenha um quadro do HUD sem o CARLA |

## 5. Interface no localhost

A interface é a página de quatro painéis (simulação, dashboard, log e chat), em http://127.0.0.1:8000/. Ela sobe junto com a corrida:

```bash
cd src
python -m mover.autonomo.dirigir
python -m mover.autonomo.dirigir --sem-carla
```

O primeiro comando usa o CARLA, e o painel 1 mostra a câmera presa ao caminhão. O segundo roda sem o CARLA, num simulador substituto com o mesmo controle, e o painel 1 mostra o mapa 2D. Ele serve para mostrar a interface em qualquer computador; sem o Ollama, junte `--provedor falso`. Com o script: `./scripts/rodar_gemeo.sh painel`.

Para deixar a página no ar entre várias corridas, suba o servidor à parte e rode as corridas em outro terminal. Elas usam o servidor que já estiver na porta 8000:

```bash
python -m mover.servidor.rodar_servidor
```

O que cada painel mostra:

**1. Simulação.** A câmera do CARLA (ou, sem ela, uma visão 2D que segue o caminhão). O HUD traz a velocidade, a desejada e o limite. Um aviso mostra o próximo elemento do plano e o que os agentes decidiram para ele.

**2. Dashboard.** O mapa da volta com as zonas do plano e o rastro que cresce; as zonas com alteração decidida ficam com contorno verde. Os gráficos de velocidade e aceleração se montam ao vivo:

- linha cheia: o que foi medido;
- linha fina: a velocidade desejada;
- tracejado cinza: o limite do plano;
- à frente do instante atual: o plano que o caminhão mandou para o trecho seguinte (vermelho) e o plano com a alteração dos agentes (verde).

O cartão de consumo mostra o diesel e os R$ da corrida e o diesel do plano sem agentes (previsto pelo modelo do caminhão) na mesma distância, com a economia até ali. Ele também mostra as decisões (quantas foram do LLM e a latência), os eventos bruscos medidos (e os previstos sem agentes) e o último benchmark.

**3. Log.** As decisões (a alteração, se foi do LLM ou da regra, o que o plano do caminhão faria, a proposta do ML, o diesel previsto e o tempo extra), os trechos medidos e os problemas. Embaixo, um cartão mostra o próximo trecho.

**4. Chat.** Responde perguntas sobre a corrida, como "quanto diesel os agentes economizaram?", "quais intervenções foram feitas?" e "o que vem pela frente?". As respostas são curtas e citam as entradas (#id). A mesma guarda de números do Supervisor vale aqui.

## 6. Benchmark: N voltas sem agentes e N com agentes

```bash
cd src
python -m mover.autonomo.benchmark --voltas 3
```

Roda no CARLA sem desenhar a cena (`no_rendering_mode`, mais rápido); use `--com-render` para
acompanhar. Primeiro vêm N voltas com o caminhão sozinho, executando o próprio plano sem
alteração. Depois o caminhão volta à largada e faz N voltas com os agentes alterando o plano de
cada bloco. As duas baterias rodam na mesma execução, e uma não lê a outra. A simulação espera
cada decisão, então o resultado não depende da velocidade do computador. Saída em
`data/benchmark/<data>/`:

- `resultados.csv`: uma linha por volta de cada bateria, as médias por volta, a diferença (absoluta e em %) e o total em 100 voltas (`--extrapolar`);
- `resumo.json`: a página mostra o resultado no cartão de consumo;
- `comparativo.png`: velocidade na 1ª volta e diesel por volta;
- `telemetria_sem_agentes.csv`, `telemetria_com_agentes.csv` e `decisoes.jsonl`.

As colunas do `resultados.csv` são:

- tempo, distância e velocidade média;
- diesel (mL), custo (R$) e CO2 (g);
- eventos bruscos de atenção e críticos;
- jerk máximo, aceleração mínima e máxima;
- energia jogada fora nos freios.

A extrapolação para 100 voltas soma a 1ª volta (saída parada), a última (parada final) e 98 vezes
a média das voltas do meio. Por isso o benchmark pede pelo menos 3 voltas.

No simulador substituto (só desenvolvimento: os números do TCC são os do CARLA), 3 + 3 voltas deram:

| Por volta | Sem agentes | Com agentes | Diferença |
|---|---|---|---|
| Diesel | 406 mL | 250 mL | −38 % |
| Tempo | 158 s | 196 s | +24 % |
| Eventos bruscos | 23,7 | 0 | −100 % |

Em 100 voltas, isso dá 39,7 L contra 24,6 L: 15,1 L a menos, R$ 90,63 e 40,5 kg de CO2. O custo
é 61 min a mais. A troca entre diesel e tempo depende do valor do tempo
(`decisao.valor_tempo_r_h`, R$ 30/h) e do teto de tempo extra (`max_tempo_extra_pct`). O modelo
de ML aprende com esse mesmo custo: se mudar esses valores, treine-o de novo (seção 7).

## 7. O modelo de ML que propõe a suavização

```bash
cd src
python -m mover.agentes.suavizador_ml
```

Não é preciso rodar à mão: se `data/modelos/suavizador_ml.joblib` não existir, a primeira corrida com agentes (ou o benchmark) treina o modelo sozinha, uma vez. Rode o comando para treinar de novo, por exemplo depois de mudar o caminhão, o piloto ou o valor do tempo no `config/autonomo.yaml` (seção `suavizador_ml`). Com o script: `./scripts/rodar_gemeo.sh treinar`. O código está em `src/mover/agentes/suavizador_ml.py`.

Entradas, todas tiradas do plano que o caminhão manda:

- a velocidade agora e a de cruzeiro;
- o limite da zona, a distância até ela, o comprimento e, se for parada, a duração;
- a distância até a zona seguinte e o limite dela;
- a desaceleração e a aceleração que o próprio plano traz.

Saídas: com que desaceleração frear antes da zona (0,4 a 1,8 m/s²), com que aceleração retomar depois (0,3 a 1,2 m/s²) e até que velocidade retomar quando a zona seguinte está perto.

O treino sorteia 2000 situações (uma lombada, curva, faixa de pedestre, parada ou partida à frente, às vezes com outra zona logo depois) e simula cada uma com o modelo do caminhão. Em cada situação, uma busca acha a suavização de menor custo (diesel em R$ + tempo × R$ 30/h) sem eventos bruscos e sem passar de +60 % de tempo no trecho: esse é o rótulo. Uma floresta aleatória do scikit-learn (200 árvores, três saídas) aprende a ir das entradas ao rótulo. O treino separa 20 % das situações para teste e grava as métricas em `data/modelos/suavizador_ml.json`.

No treino de 10/10/2026 (2000 situações, 84 s em 8 núcleos):

- R² no teste: 0,44 na desaceleração, 0,85 na retomada e 0,98 no teto;
- simuladas nas situações de teste, 100 % das propostas do modelo ficaram sem eventos bruscos, com custo 0,48 % acima do ótimo da busca;
- a entrada mais importante é a distância até a zona seguinte.

O R² da desaceleração é baixo porque várias desacelerações dão quase o mesmo custo; o que importa é que a proposta não tenha tranco e custe perto do ótimo. Na corrida, toda proposta ainda passa pela conferência do modelo do caminhão antes de valer (seção Como a camada agêntica decide).

## 8. Análise de uma corrida pela camada de logs (sem o CARLA)

```bash
cd src
python -m mover.agentes.rodar_agentes
python -m mover.agentes.treinar_especialista_ml
```

O primeiro comando roda a camada de logs sobre a corrida mais recente de `data/autonomo/corridas/`, bloco a bloco de 10 s, sem abrir o simulador: o especialista de jerk aponta cada evento brusco medido, o especialista de ML dos logs marca os trechos atípicos e o Supervisor (LLM) escreve os logs. Numa corrida sem agentes, aparecem as decisões bruscas do caminhão sozinho; com agentes, elas devem sumir. O resultado sai em `data/agentes/log_agentes.jsonl`, com um resumo ao lado. Outra corrida: `--corrida ../data/autonomo/corridas/<id>`. Sem o Ollama, junte `--provedor falso`. Com o script: `./scripts/rodar_gemeo.sh analise`.

O segundo comando treina o especialista de ML dos logs (IsolationForest) com a corrida e grava `data/modelos/especialista_ml.joblib`. Não é preciso rodá-lo à mão: a análise treina de novo sozinha quando a corrida muda. É outro modelo, só da camada de logs; quem propõe a suavização é o da seção 7.

## Como a camada agêntica decide

A camada agêntica trabalha em blocos de 10 s. Antes da largada, o caminhão manda o plano dos dois
primeiros blocos e a simulação espera a decisão. Depois, a cada bloco k que começa, o caminhão
monta o plano do bloco k+1 a partir do estado medido naquele instante e o manda aos agentes, que
trabalham numa thread enquanto ele anda:

**1. Plano do caminhão.** Os elementos do plano de missão que ele vai alcançar no bloco k+1 (mais 20 m de margem) e que ainda não têm decisão. Para eles, o caminhão simula o que vai fazer: a velocidade desejada que vai pedir em cada zona e a velocidade e a aceleração que isso gera, pelo modelo do controle do Traffic Manager (o TM não expõe a aceleração que vai aplicar). As manobras à frente vêm do próprio TM (`get_all_actions`); sem o CARLA, da curvatura da rota. O código está em `src/mover/autonomo/planejador.py`.

**2. Especialista de jerk.** Aponta as decisões bruscas desse plano (|jerk| ≥ 2,5 m/s³ é atenção; ≥ 5 é crítico), com a zona que causa cada uma.

**3. Especialista de ML.** Para cada zona com decisão brusca, propõe a suavização (seção 7).

**4. Especialista de consumo.** Simula, com o modelo do caminhão (dinâmica + controle no estilo do Traffic Manager), o plano sem alteração, a proposta do ML e as opções fixas, e mede:

- diesel, pelo modelo VT-CPFM;
- tempo;
- desaceleração e jerk máximos;
- eventos bruscos.

**5. Regra.** Sem decisão brusca no plano, nada muda. Com decisão brusca, fica a proposta do ML se ela não tiver mais eventos bruscos que a melhor opção fixa e não passar de +60 % de tempo no trecho; senão, vale a opção fixa de menor custo (diesel em R$ + tempo × R$ 30/h). A recomendação é aplicada na hora.

**6. Supervisor (LLM).** Recebe o plano do caminhão, a proposta do ML, a tabela das alterações simuladas e um rascunho em JSON. Ele aprova ou troca a alteração e escreve o log, com prazo de 6 s:

- se ele trocar a alteração e o caminhão ainda não tiver chegado ao elemento, a escolha dele passa a valer;
- a guarda recusa alterações com mais eventos bruscos que a da regra e números sem respaldo nos fatos;
- o LLM nunca mexe no pedal: ele só escolhe entre alterações já simuladas.

| Alteração | Desaceleração antecipada | Retomada | Teto depois do elemento |
|---|---|---|---|
| ajuste do ML | a proposta do modelo, zona a zona | a proposta do modelo | a proposta do modelo |
| econômica | 0,6 m/s² | 0,5 m/s² | 22 km/h |
| suave | 0,8 m/s² | 0,6 m/s² | — |
| moderada | 1,2 m/s² | 0,9 m/s² | — |
| firme | 1,8 m/s² | 1,2 m/s² | — |

A alteração é real. O governador transforma a alteração num envelope de velocidade com jerk
limitado (1,6 m/s³) e manda a velocidade desejada ao Traffic Manager a cada tick
(`TrafficManager.set_desired_speed`). Se uma decisão chegar depois de o piloto começar a frenagem
bruta, ela não é aplicada àquele elemento, e o log mostra "sem tempo de aplicar".

O buffer de 10 s é o `decisao.horizonte_s` do `config/autonomo.yaml`. Para diminuir, baixe junto o
`decisao.prazo_llm_s` (o LLM precisa responder antes de o bloco acabar) e o `blocos.duracao_s` do
`config/agentes.yaml`.

Em paralelo, o servidor roda a camada de logs sobre a telemetria medida, bloco a bloco. O que de
fato aconteceu (trechos e eventos de jerk) vira log em linguagem natural. Com os agentes, esses
problemas medidos devem quase sumir; sem eles, aparecem todos.

## Consumo e custo

O CARLA não simula combustível. O consumo vem do modelo VT-CPFM (Rakha et al.), aplicado à
velocidade e à aceleração medidas no CARLA e às previstas no plano do caminhão:

```
P_rodas = (m·a + arrasto + rolamento + m·g·rampa) · v   (só a parte positiva)
P_motor = P_rodas / 0,88
diesel (L/s) = marcha lenta + a1 · P_motor · (1 + P_motor / P2)
```

Os parâmetros ficam em `config/autonomo.yaml`, seção `consumo`:

- diesel, com densidade 0,835 kg/L e BSFC de 225 g/kWh;
- marcha lenta de 2 L/h;
- preço de R$ 6,00/L (atualize com o preço médio da ANP);
- 2,68 kg de CO2 por litro.

O caminhão tem 9 t. O termo quadrático (P2 = 1000 kW) é uma suposição documentada. A economia vem
de duas coisas: não acelerar até 30 km/h para logo frear, e frear menos (menos energia jogada fora
nos freios).

## Plano de missão

`config/missao.yaml` tem a rota (~780 m, sentido horário, cruzeiro de 30 km/h), as zonas e a
parada:

- 7 lombadas a 10 km/h;
- a faixa de pedestre do OSM a 15 km/h;
- 4 curvas, com limite por aceleração lateral de 1,6 m/s²;
- a parada de 4 s na portaria (o portão do OSM).

Tudo sai do mapa do CARLA, do OSM do campus e do `config/autonomo.yaml` (seção `gerar_missao`):

- a rota: o itinerário (`rota.vias`) lista as vias do `mapa_final.xodr` na ordem da volta; os conectores das junções são achados no grafo de faixas, e a rota segue o centro das faixas no sentido de tráfego;
- as lombadas: pontos fixos do mapa (`lombadas`), porque o OpenDRIVE não tem lombadas;
- as curvas: a geometria da rota;
- a faixa de pedestre e a portaria: o OSM, projetado no mapa pela georreferência do `.xodr`.

Para gerar o plano de novo:

```bash
cd src
python -m mover.autonomo.rota_mapa --figura vias.png
python -m mover.autonomo.gerar_missao
```

O primeiro comando lista as vias e desenha os ids no mapa, para mudar o itinerário. O segundo grava `config/missao.yaml` e `config/missao.png`. O arquivo também pode ser editado à mão (`s_ini`/`s_fim` em metros ao longo da rota, `v_max_kmh`, paradas com `duracao_s`). O plano de missão só define as zonas: o que o caminhão faz em cada uma ele decide na corrida, a partir do estado medido.

## Configuração

| Arquivo | O que tem |
|---|---|
| `config/autonomo.yaml` | missão, geração do plano (vias, largada, lombadas), caminhão, piloto, governador, opções fixas, decisão (buffer de 10 s, prazo do LLM, valor do tempo), modelo de ML (`suavizador_ml`), consumo, limiares, benchmark, pasta das corridas e CARLA (porta do TM, clima) |
| `config/missao.yaml` | rota, zonas e paradas da volta |
| `config/simulacao.yaml` | mapa, CARLA (host, porta, OpenDRIVE), caminhão, câmeras e servidor do painel |
| `config/agentes.yaml` | LLM (provedor, modelo), camada de logs (entrada `corrida`, a mais recente), especialista de ML dos logs, chat e base de conhecimento do chat |
| `.env` (raiz) | `MOVER_LLM_PROVEDOR`, `MOVER_LLM_MODELO` e `OLLAMA_HOST`, gravados pelo script do Ollama (modelo em `.env.example`) |

O Traffic Manager usa a porta 8100, porque a 8000 é a do servidor da página.

## Estrutura

```
README.md, requirements.txt, .env.example, .gitattributes
config/        autonomo.yaml, missao.yaml (+ missao.png), simulacao.yaml,
               agentes.yaml, tratamento.yaml (legado)
scripts/       rodar_gemeo.sh, preparar_ollama.ps1, preparar_ollama.sh
src/mover/
  autonomo/    rota, rota_mapa, missao, gerar_missao, planejador (o plano
               dos próximos 10 s), veiculo, governador, previsao, consumo,
               simulador (substituto), carla_tm (CARLA), conducao (laço),
               montagem, cliente, dirigir, benchmark
  agentes/     camada agêntica: decisao (alterações), suavizador_ml (modelo
               de ML), grafo/supervisor (logs), rodar_agentes,
               treinar_especialista_ml, llm...
  servidor/    app (FastAPI + WebSocket), sessao, rodar_servidor
  interface/   rotas, cena, chat e estatico/ (página, CSS e 6 módulos JS)
  video/       hud, gravar_video
  simulacao/   opendrive, projecao, camera_painel, fila_envio
               (alinhamento e verificar_mapa: legado)
  tratamento/  legado (só o sinais.py, de filtros, é usado)
tests/         test_planejador, test_autonomo, test_servidor,
               test_interface, test_simulacao, test_video, test_scripts,
               test_agentes
data/          mapas/ (entrada), modelos/, autonomo/, benchmark/ (saídas)
```

## Testes

```bash
python -m pytest tests -q
```

São 93 testes, sem CARLA e sem LLM de verdade (rodam da raiz do repositório). O lado CARLA é testado com um módulo `carla` falso (um Traffic Manager que leva o caminhão pelo caminho) e o LLM, com o provedor falso e um "LLM de roteiro". Eles conferem que:

- o plano do caminhão sai do estado medido na corrida atual, com as decisões bruscas e a zona de cada uma;
- as manobras vêm do Traffic Manager (`get_all_actions`) ou, sem ele, da rota;
- o modelo de ML treina com situações simuladas e propõe suavizações leves, e a proposta só vale se o modelo do caminhão aprovar;
- sem decisão brusca no plano, os agentes não alteram nada;
- na corrida, cada plano parte do estado medido naquela hora, e com as alterações os trancos somem;
- a primeira corrida com agentes treina o modelo de ML sozinha;
- a rota sai do mapa, com os conectores das junções;
- o LLM fica dentro das alterações seguras;
- a corrida ao vivo e o benchmark rodam no simulador substituto;
- o servidor e o chat funcionam;
- o MP4 sai em H.264;
- o `rodar_gemeo.sh` monta o comando certo em cada modo e não coleta nada antes das corridas.

## Limitações (para o TCC)

- **O Traffic Manager não é aprendizado de máquina.** É o piloto automático do CARLA (navegação por regras + PID). As frenagens bruscas vêm do plano em degrau, que é como o TM segue limites de velocidade sozinho.
- **O TM não expõe a velocidade nem a aceleração que vai aplicar.** Ele expõe as manobras que já decidiu (`get_all_actions`), e é isso que o plano usa para as curvas. A velocidade desejada de cada zona é decisão do planejador do caminhão, e a velocidade e a aceleração resultantes vêm do modelo do controle do TM. Por isso o plano é uma previsão; o que vale é o medido no CARLA.
- **O modelo de ML aprendeu no modelo do caminhão, não no CARLA.** As situações de treino são simuladas pelo mesmo modelo que confere as propostas. Se o caminhão do CARLA responder diferente, ajuste `veiculo` e `piloto` no `config/autonomo.yaml` à telemetria medida e treine de novo.
- **O efeito da alteração é previsto antes e medido depois.** Os números que valem são os medidos no benchmark no CARLA.
- **O consumo é estimado** pelo VT-CPFM: o CARLA não tem combustível.
- **Nada foi rodado no CARLA de verdade aqui.** O desenvolvimento foi feito sem CARLA e sem GPU. O lado CARLA foi conferido contra a API do pacote 0.9.16 e testado com um módulo falso. Na primeira corrida no CARLA, confira se o caminhão segue a rota (`set_path`), se `get_all_actions` devolve as manobras (senão, o plano usa a curvatura da rota) e se a reprodução da gravação carrega o mapa OpenDRIVE (senão, use `--modo poses` no vídeo).
- **A latência do LLM depende da máquina.** Sem GPU, o gemma2:2b pode passar do prazo de 6 s, e então as decisões ficam com a regra (o log mostra o motivo).

## Legado

O repositório ainda tem o código das etapas antigas, que usavam a volta gravada com o celular pelo Sensor Logger (`data/csv_maua`). Nada do fluxo acima usa esse código:

- `src/mover/tratamento/` e `config/tratamento.yaml`: o tratamento do Sensor Logger (`python -m mover.tratamento.tratar_dados`);
- `src/mover/simulacao/alinhamento.py` e `verificar_mapa.py`: casavam a gravação com o mapa;
- `python -m mover.agentes.rodar_agentes --entrada ../data/tratado/telemetria_tratada.csv --injetar-eventos`: a análise antiga, com eventos sintéticos.

A exceção é o `src/mover/tratamento/sinais.py` (filtros), usado pelo consumo e pela camada de logs. O resto pode ser apagado sem afetar o gêmeo digital.
