"""Gêmeo digital com caminhão autônomo.

O caminhão dá a volta no campus sozinho, dirigido pelo piloto automático do CARLA (Traffic
Manager). Como todo carro autônomo, ele decide antes de agir: a cada bloco, monta o plano dos
próximos 10 s a partir da corrida atual e o manda à camada agêntica, que o altera levemente quando
ele tem decisões bruscas. O caminhão executa o plano alterado (a velocidade desejada, a cada tick).

- rota: linha central da faixa ao longo da volta, parametrizada pela distância;
- rota_mapa: a volta montada só com o mapa do CARLA (.xodr), a partir de uma lista de vias;
- missao: zonas de velocidade (lombadas, faixas de pedestre, curvas) e paradas;
- gerar_missao: monta o plano a partir do mapa do CARLA, do OSM e do config (nenhuma gravação);
- planejador: o plano dos próximos 10 s que o caminhão manda aos agentes (velocidade, aceleração,
  manobras e decisões bruscas), a partir do estado atual da corrida;
- veiculo: dinâmica longitudinal do caminhão e o controle no estilo do Traffic Manager;
- governador: velocidade desejada a cada tick (plano bruto ou suavizado pelos agentes);
- previsao: o modelo do caminhão que simula os próximos segundos (o plano e as alterações);
- consumo: combustível, custo e CO2 a partir da potência nas rodas (VT-CPFM);
- simulador: simulador substituto (sem CARLA); carla_tm: o lado CARLA (Traffic Manager);
- conducao: o laço da corrida (governador -> piloto -> telemetria, com as decisões dos agentes);
- montagem: monta todas as peças a partir do config/autonomo.yaml;
- cliente: envia a telemetria e as decisões ao servidor do painel;
- dirigir: corrida ao vivo (linha de comando); benchmark: N voltas sem e N com os agentes.
"""
