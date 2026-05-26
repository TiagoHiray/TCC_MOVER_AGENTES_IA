# TCC - MOVER Coordenacao Logistica e Operacional de Veiculos Agricolas Autonomos
#### Realizado por:  
 Tiago Hiray Hisatugo,  
 Daniel Djinishian de Briquez,  
 Eduardo Cunha Santiago,  
 Gabriel Silva Garcia,   
 Eduardo Takase Sawada

# Introdução

&nbsp;&nbsp;&nbsp;&nbsp;O avanço das tecnologias de automação, sensoriamento e inteligência artificial tem impulsionado o desenvolvimento de veículos autônomos aplicados também ao contexto agrícola e off-road. A integração de sensores, sistemas de posicionamento GNSS e técnicas de fusão de dados permite melhorar o monitoramento operacional e a navegação autônoma desses sistemas (BARRILE et al., 2022).   
&nbsp;&nbsp;&nbsp;&nbsp;Além disso, a integração entre comunicação veicular, computação distribuída e inteligência artificial amplia a capacidade de coordenação e supervisão de sistemas autônomos, tornando as operações mais eficientes e seguras.  
&nbsp;&nbsp;&nbsp;&nbsp;No contexto agrícola e logístico, a utilização de Sistemas Multiagentes (SMA) tem se mostrado uma abordagem promissora para lidar com a complexidade operacional de múltiplos veículos autônomos. Nesta arquitetura, cada veículo é modelado como um agente inteligente capaz de perceber o ambiente, tomar decisões locais e interagir cooperativamente com outros agentes.   
&nbsp;&nbsp;&nbsp;&nbsp;Nesse contexto, o projeto MOVER, desenvolvido em parceria entre o Instituto Mauá de Tecnologia, a Escola Politécnica da Universidade de São Paulo, FATEC Santo André, Mercedes-Benz, HYDAC, e FUNDEP, investiga arquiteturas inteligentes aplicadas à logística autônoma em ambientes off-road. Entre as linhas de pesquisa do projeto estão a coordenação operacional entre caminhões e colhedoras autônomas, bem como estratégias de supervisão e monitoramento inteligente.  
&nbsp;&nbsp;&nbsp;&nbsp;Inserido nesse cenário, o presente trabalho, intitulado Projeto MOVER: Infraestrutura Cognitiva Multiagente para Operações com Caminhões Autônomos, tem como objetivo desenvolver uma infraestrutura virtual de simulação baseada no simulador CARLA para treinamento, validação e testes de agentes inteligentes aplicados à condução autônoma de caminhões.  
&nbsp;&nbsp;&nbsp;&nbsp;O simulador CARLA permite reproduzir ambientes complexos e integrar diferentes sensores e algoritmos de percepção, planejamento e controle de forma segura (DOSOVITSKIY et al., 2017). Neste trabalho, o ambiente virtual será utilizado para o treinamento do agente responsável pela condução autônoma do caminhão e para o desenvolvimento de módulos cognitivos voltados à manutenção preditiva e supervisão operacional.  
&nbsp;&nbsp;&nbsp;&nbsp;Além do treinamento do agente autônomo, o trabalho propõe a utilização de dados provenientes dos sensores embarcados para análise de degradação de componentes, previsão de falhas e recomendação de manutenção preventiva. Também serão desenvolvidos mecanismos de supervisão cognitiva, incluindo dashboards inteligentes, geração automática de relatórios e integração com Modelos de Large Language Models (LLMs), permitindo maior transparência e interpretabilidade das decisões tomadas pelo sistema através de linguagem natural em Logs e chat interativo.  
&nbsp;&nbsp;&nbsp;&nbsp;Dessa forma, a utilização de um ambiente virtual baseado no simulador CARLA, associada a arquiteturas multiagentes e técnicas de inteligência artificial, representa uma alternativa segura e escalável para o desenvolvimento e validação de soluções aplicadas à condução autônoma, supervisão cognitiva e manutenção preditiva de caminhões autônomos.  

# Inicialização

#### Versões recomendadas:  
Python 3.12  
Carla 0.9.16  

### 1. Iniciar Servidor
> C:\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen -quality-level=Low -world-port=2000

### 2. Em outro PowerShell, rodar a coleta
> cd C:\CARLA_0.9.16\meu_tcc  
C:\CARLA_0.9.16\venv_carla\Scripts\python.exe coleta_carla_v4.py --town Town03 --duration 60 --vehicles 30 --pedestrians 20 --output .\dataset\run_XYZ

### 3. Gerar vídeos e dashboards
> C:\CARLA_0.9.16\venv_carla\Scripts\python.exe gerar_video.py --input .\dataset\run_XYZ  
C:\CARLA_0.9.16\venv_carla\Scripts\python.exe dashboard.py --input .\dataset\run_XYZ  
 
### 4. Ao terminar, desligar o servidor
> taskkill /F /IM CarlaUE4-Win64-Shipping.exe

# Camada cognitiva

### Representação visual

<img width="684" height="1269" alt="ArquiteturaAgentes" src="https://github.com/user-attachments/assets/5026f4f6-ba20-4fab-9286-82f29de5a588" />   


&nbsp;&nbsp;&nbsp;&nbsp; A arquitetura multiagente proposta é estruturada de forma hierárquica e modular, na qual agentes especializados desempenham funções específicas para apoiar a tomada de decisão do sistema. O fluxo tem início no ambiente do simulador CARLA, responsável por gerar dados de telemetria e sensores, como velocidade, IMU, LiDAR e câmeras. Esses dados são transmitidos via ROS 2 para um monitor de telemetria, implementado em Python, que realiza a ingestão das informações e atualiza continuamente o estado operacional do sistema.  
&nbsp;&nbsp;&nbsp;&nbsp; A partir desse estado, um agente especialista em manutenção analisa os dados recebidos com o objetivo de identificar anomalias (outliers) e gerar eventos associados ao comportamento do veículo. Com base nessa análise, o sistema estabelece caminhos condicionais distintos: em condições normais, os eventos são apenas registrados em log para rastreabilidade e posterior análise. Já em situações críticas, as informações são encaminhadas para um agente supervisor cognitivo.  
&nbsp;&nbsp;&nbsp;&nbsp; Esse agente supervisor atua como uma camada de decisão de alto nível, utilizando um modelo de linguagem (LLM) para interpretar rapidamente o contexto detectado e gerar explicações ou ações associadas ao evento identificado. Como resultado, o sistema pode emitir alertas ao operador e acionar comandos de segurança, como a solicitação de parada do veículo.  
&nbsp;&nbsp;&nbsp;&nbsp; Sob a perspectiva de sistemas multiagentes, a principal característica dessa arquitetura está na especialização funcional dos agentes, em que cada componente possui responsabilidade bem definida — monitoramento, detecção de anomalias, supervisão cognitiva e resposta operacional — contribuindo coletivamente para a construção de uma camada de decisão modular, adaptativa e extensível. Isso favorece manutenção, escalabilidade e evolução progressiva do sistema durante o treinamento em ambiente virtualizado.  

# Resultados gerados por instância
### 1. Vídeo do trajeto  

<img width="800" height="600" alt="001424" src="https://github.com/user-attachments/assets/11250859-29d8-4223-a36b-896b19cc7d08" />


### 2. Dashbords de telemetria
<img width="1389" height="985" alt="dashboard" src="https://github.com/user-attachments/assets/1a356f68-646c-48d8-83a4-36c5667b2c9d" />  

### 3. CSV de telemetria

> frame	tipo	outro  
1472,	lane_invasion,	[carla.libcarla.LaneMarkingType.Broken]  
1473,	lane_invasion,	[carla.libcarla.LaneMarkingType.NONE]  
1719,	lane_invasion,	[carla.libcarla.LaneMarkingType.NONE]  

### 4. Eventos
> Iniciado em 2026-05-05 20:28:42  
[frame 2074] lane_invasion: [carla.libcarla.LaneMarkingType.NONE]  
Finalizado em 2026-05-05 20:30:09  
Total de eventos: 1  

### 5. Metadata  
> {  
  "host": "localhost",  
  "port": 2000,  
  "town": "Town01",  
  "fixed_delta_seconds": 0.05,  
  "vehicle_filter": "vehicle.carlamotors.firetruck",  
  "camera_width": 800,  
  "camera_height": 600,   
  "camera_fov": 90,  
  "lidar_channels": 16,  
  "lidar_range": 30.0,  
  "lidar_points_per_second": 100000,  
  "lidar_rotation_frequency": 20,  
  "gnss_noise_lat_stddev": 1e-05,   
  "gnss_noise_lon_stddev": 1e-05,  
  "weather": "WeatherParameters(cloudiness=5.000000, precipitation=0.000000, precipitation_deposits=0.000000, wind_intensity=10.000000, sun_azimuth_angle=-1.000000, sun_altitude_angle=45.000000,   fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.100000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.030000, rayleigh_scattering_scale=0.033100, dust_storm=0.000000)",  
  "warmup_ticks": 30,  
  "n_vehicles": 30,  
  "n_pedestrians": 45,  
  "try_collision_sensor": true,   
  "try_lane_invasion_sensor": true,   
  "ticks_between_event_sensors": 10,  
  "save_lidar_npy": true,  
  "video_fps": 20,  
  "video_codec": "mp4v",  
  "hud_enabled": true,  
  "started_at": "2026-05-05 21:06:14"  
}  
