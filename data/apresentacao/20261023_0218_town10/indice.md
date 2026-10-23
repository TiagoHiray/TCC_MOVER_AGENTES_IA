# Apresentação: cada situação em cada modo (Town10HD_Opt, 23/10/2026 03:16)

Números: média por volta das 3 voltas do benchmark de cada modo, dentro do trecho da situação (trechos cortados pela chegada ficam de fora). Vídeos: recorte da volta gravada ao vivo com o painel.

## 1. Pedestre atravessando na faixa

Pergunta: Como o caminhão se comporta com um pedestre atravessando?
Trecho: de 74 m a 158 m da volta.

| Modo | Tempo (s) | Diesel (mL) | Eventos bruscos | Críticos | V mín. (km/h) | Pior desacel. (m/s²) | Menor dist. ao figurante (m) | Vídeo |
|---|---|---|---|---|---|---|---|---|
| Só o CARLA (Traffic Manager) | 21.18 | 32.82 | 5.0 | 4.0 | 0.0 | 12.43 | 5.72 | [ver](situacoes/pedestre/01_so_carla.mp4) |
| Só a regra (opções fixas) | 22.27 | 28.26 | 2.33 | 2.33 | 0.0 | 8.21 | 4.52 | [ver](situacoes/pedestre/02_so_regra.mp4) |
| Só o ML (floresta aleatória) | 22.18 | 26.1 | 2.0 | 2.0 | 0.0 | 8.06 | 5.74 | [ver](situacoes/pedestre/03_so_ml.mp4) |
| Só o LLM (sobre as opções fixas) | 22.27 | 28.26 | 2.33 | 2.33 | 0.0 | 8.21 | 4.52 | [ver](situacoes/pedestre/04_so_llm.mp4) |
| Completo: ML + regra + LLM | 22.18 | 26.1 | 2.0 | 2.0 | 0.0 | 8.06 | 5.74 | [ver](situacoes/pedestre/05_completo.mp4) |
| Só o CARLA a 20 km/h (controle) | 21.08 | 32.93 | 2.33 | 2.33 | 0.0 | 7.88 | 3.89 | [ver](situacoes/pedestre/06_devagar_20.mp4) |

## 2. Veículo lento à frente (12 km/h)

Pergunta: Como o caminhão se comporta com outro veículo à frente?
Trecho: de 311 m a 566 m da volta.

| Modo | Tempo (s) | Diesel (mL) | Eventos bruscos | Críticos | V mín. (km/h) | Pior desacel. (m/s²) | Menor dist. ao figurante (m) | Vídeo |
|---|---|---|---|---|---|---|---|---|
| Só o CARLA (Traffic Manager) | 70.73 | 81.08 | 2.33 | 1.67 | 7.72 | 10.87 | 11.68 | [ver](situacoes/veiculo/01_so_carla.mp4) |
| Só a regra (opções fixas) | 70.83 | 70.65 | 4.0 | 3.0 | 8.51 | 9.3 | 11.78 | [ver](situacoes/veiculo/02_so_regra.mp4) |
| Só o ML (floresta aleatória) | 70.82 | 69.98 | 1.33 | 1.0 | 9.07 | 8.69 | 11.64 | [ver](situacoes/veiculo/03_so_ml.mp4) |
| Só o LLM (sobre as opções fixas) | 70.83 | 70.65 | 4.0 | 3.0 | 8.51 | 9.3 | 11.78 | [ver](situacoes/veiculo/04_so_llm.mp4) |
| Completo: ML + regra + LLM | 70.82 | 69.98 | 1.33 | 1.0 | 9.07 | 8.69 | 11.64 | [ver](situacoes/veiculo/05_completo.mp4) |
| Só o CARLA a 20 km/h (controle) | 70.73 | 67.83 | 3.0 | 1.67 | 9.07 | 7.24 | 11.72 | [ver](situacoes/veiculo/06_devagar_20.mp4) |

## Volta inteira (benchmark, por volta)

| Modo | Tempo (s) | Diesel (mL) | Eventos bruscos | Críticos | V mín. (km/h) | Pior desacel. (m/s²) | Menor dist. ao figurante (m) | Vídeo |
|---|---|---|---|---|---|---|---|---|
| Só o CARLA (Traffic Manager) | 174.4 | 258.76 | 20.667 | 15.333 | - | - | - | [ver](modos/01_so_carla/video_painel.mp4) |
| Só a regra (opções fixas) | 185.983 | 213.103 | 12.0 | 9.333 | - | - | - | [ver](modos/02_so_regra/video_painel.mp4) |
| Só o ML (floresta aleatória) | 184.817 | 219.49 | 8.333 | 5.667 | - | - | - | [ver](modos/03_so_ml/video_painel.mp4) |
| Só o LLM (sobre as opções fixas) | 185.983 | 213.103 | 12.0 | 9.333 | - | - | - | [ver](modos/04_so_llm/video_painel.mp4) |
| Completo: ML + regra + LLM | 184.817 | 219.49 | 8.333 | 5.667 | - | - | - | [ver](modos/05_completo/video_painel.mp4) |
| Só o CARLA a 20 km/h (controle) | 184.583 | 219.51 | 19.333 | 14.667 | - | - | - | [ver](modos/06_devagar_20/video_painel.mp4) |

## Modos

- Só o CARLA (Traffic Manager): Linha de base: o piloto automático do CARLA segue o limite de cada zona em degrau e freia em cima.
- Só a regra (opções fixas): A regra escolhe entre as 4 opções fixas, conferidas pelo modelo do caminhão. Sem ML e sem LLM.
- Só o ML (floresta aleatória): O ML propõe a suavização zona a zona e a regra só confere. Sem LLM (também é o caso de o LLM cair).
- Só o LLM (sobre as opções fixas): O LLM escolhe entre as opções fixas já simuladas; a guarda recusa opção com mais eventos bruscos.
- Completo: ML + regra + LLM: O ML propõe, o modelo do caminhão confere, a regra escolhe e o LLM supervisiona e explica no log.
- Só o CARLA a 20 km/h (controle): Responde 'não bastava andar mais devagar?': a mesma volta sem agentes, com cruzeiro de 20 km/h.
