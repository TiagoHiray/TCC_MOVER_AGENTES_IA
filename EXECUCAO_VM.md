# Execução na VM (CARLA 0.9.16) - branch escopo-main

Roteiro: abrir o CARLA, rodar uma volta de teste e depois o benchmark (N voltas sem x N com agentes).
O caminhão (processo do CARLA) fala com os agentes por HTTP (POST /agentes/plano, GET /agentes/decisoes).

Antes de ir para a VM, envie o código daqui:

```powershell
git push -u origin escopo-main
```

## Sobre a velocidade

- `-RenderOffScreen` só tira a janela do CARLA; ele continua desenhando a cena.
- O que acelera de verdade é `--sem-render` (dirigir) ou o padrão do benchmark: o CARLA deixa de desenhar.
- `dirigir` anda em tempo real por padrão (para acompanhar na página). `--sem-espera` tira essa espera;
  a simulação só para quando precisa esperar os agentes. O benchmark já roda assim.
- Sem renderização não há câmera: o painel 1 mostra o mapa 2D.

## 1. Preparar a VM (uma vez, e `git pull` a cada atualização)

```powershell
cd C:\CARLA_0.9.16\PythonAPI\TCC_MOVER_AGENTES_IA
git fetch; git checkout escopo-main; git pull
Set-ExecutionPolicy -Scope Process Bypass     # só se o Activate.ps1 for bloqueado
.\env_carla\Scripts\Activate.ps1              # Python 3.12; se não existir: py -3.12 -m venv env_carla
pip install -r requirements.txt               # já traz carla==0.9.16
powershell -ExecutionPolicy Bypass -File scripts\preparar_ollama.ps1   # opcional: Ollama + gemma2:2b
```

## 2. Abrir o CARLA sem janela

```powershell
Start-Process "C:\CARLA_0.9.16\CarlaUE4.exe" -ArgumentList "-RenderOffScreen","-quality-level=Low","-nosound"
Test-NetConnection 127.0.0.1 -Port 2000       # espere carregar até dar TcpTestSucceeded: True
```

Ajuste o caminho se o CARLA estiver em outra pasta. Para fechar: `Get-Process CarlaUE4* | Stop-Process`.
Para ver o caminhão andando, abra sem `-RenderOffScreen` (o README do projeto usa `-quality-level=Low -world-port=2000`).

## 3. Volta de teste (a partir de `src`)

```powershell
cd src
python -m mover.autonomo.dirigir --sem-painel --sem-render --sem-espera --provedor falso
```

O comando gera o mapa do campus, sobe o servidor dos agentes numa porta livre e o caminhão fala com
ele por HTTP. A corrida fica em `data\autonomo\corridas\<data>_agentes\`. Confira no `resumo.json`:

- `motivo_fim` = `"chegada"` (`"preso"` = parado mais de 30 s; `"tempo_max"` = não chegou);
- `agentes_http` com a URL do servidor e `requisicoes_agentes` > 0;
- `decisoes` > 0 e os eventos bruscos (`total.eventos_atencao`, `total.eventos_criticos`).

Na primeira vez o servidor treina o modelo de ML (~1-2 min) antes da largada.

Com a página (tempo real): `python -m mover.autonomo.dirigir --sem-render` e abra http://127.0.0.1:8000/.

## 4. Benchmark: N voltas sem agentes x N com agentes

```powershell
python -m mover.autonomo.benchmark --voltas 3 --provedor falso
python -m mover.autonomo.benchmark --voltas 3                  # com o Ollama no ar (LLM de verdade)
```

Já roda sem desenhar a cena: a janela do CARLA fica preta, e isso é o esperado (use `--com-render` para ver).
O log mostra o início de cada bateria e o fim de cada volta (cerca de 200 s de simulação; poucos segundos de
relógio cada). Saída em `data\benchmark\<data>\`: `resultados.csv`, `resumo.json`, `comparativo.png` e as
telemetrias das duas baterias. Os números só valem se as duas baterias terminarem com `chegada`.

Agentes noutra máquina (ou num terminal à parte):

```powershell
python -m mover.servidor.rodar_servidor --host 0.0.0.0                    # máquina dos agentes
python -m mover.autonomo.benchmark --voltas 3 --servidor http://<ip>:8000  # máquina do CARLA
```

## LLM

- Com `--provedor falso` o fluxo todo funciona com o texto-modelo; use no primeiro teste.
- Para o LLM de verdade, a VM precisa do Ollama no ar com `gemma2:2b` (`scripts\preparar_ollama.ps1`); aí é só tirar o `--provedor`.
- Sem o Ollama no ar a corrida não quebra: as decisões ficam com a regra.
- Cada decisão tem prazo de 6 s; sem GPU o gemma2:2b pode passar disso e cair na regra (o log mostra o motivo).

## Se algo der errado

- **CARLA travado depois de uma falha no script:** ele fica esperando o próximo passo do modo síncrono.
  Feche e abra de novo. O Ctrl+C normal devolve o CARLA ao modo assíncrono.
- **Tick travado:** se o CARLA (ou o Traffic Manager) não devolver um passo em 90 s, o script encerra sozinho
  com a mensagem "O tick do CARLA não voltou". Feche o CARLA, abra de novo e rode outra vez.
- **`preso`, `removido` ou `sem_resposta` no fim:** a corrida grava o que andou; a linha de erro do log diz
  `d`, `z` e o desvio da rota. Mande essa linha para análise.
- **Porta 8000 ocupada:** o `dirigir` com página usa o servidor que já estiver nela; sem página e no benchmark,
  os agentes sobem numa porta livre. O Traffic Manager usa a 8100 e o CARLA, a 2000.
- **"plano de missão diferente"** ao abrir a sessão dos agentes: o servidor remoto tem outro `config/missao.yaml`;
  faça `git pull` nas duas máquinas.
- **Ver a página do notebook:** `python -m mover.servidor.rodar_servidor --host 0.0.0.0` na VM e, no outro terminal,
  `python -m mover.autonomo.dirigir --servidor http://127.0.0.1:8000`. Acesse `http://<IP da VM>:8000/`.
  Só dentro da rede da Mauá: essas rotas não têm senha.
