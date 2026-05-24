# TCC - MOVER Coordenacao Logistica e Operacional de Veiculos Agricolas Autonomos

&nbsp;&nbsp;&nbsp;&nbsp;O avanço das tecnologias de automação, sensoriamento, inteligência artificial e comunicação veicular tem impulsionado o desenvolvimento de sistemas autônomos aplicados a operações logísticas em ambientes complexos, incluindo cenários agrícolas e off-road (BARRILE et al., 2022).  
&nbsp;&nbsp;&nbsp;&nbsp;  Entretanto, a validação desses sistemas em campo apresenta desafios relacionados a custo, segurança, repetibilidade experimental e exposição a condições dinâmicas de difícil controle.  (PENDLETON et al., 2017; KIRAN et al., 2021).  
&nbsp;&nbsp;&nbsp;&nbsp;Nesse contexto, este trabalho propõe o desenvolvimento de uma infraestrutura cognitiva multiagente para apoio à operação, supervisão e análise preditiva de caminhões autônomos agrícolas no âmbito do Projeto MOVER.  
&nbsp;&nbsp;&nbsp;&nbsp;A solução é baseada em um ambiente virtual de co-simulação utilizando o simulador CARLA, empregado como gêmeo digital para criação, execução e avaliação de cenários operacionais controlados.  
&nbsp;&nbsp;&nbsp;&nbsp;A arquitetura proposta organiza agentes inteligentes em diferentes camadas, contemplando agentes embarcados, responsáveis pelo processamento de dados sensoriais e tomada de decisão local; mecanismos de comunicação cooperativa entre veículos e infraestrutura; e agentes em nuvem, voltados à análise global, supervisão da missão, apoio à coordenação operacional e manutenção preditiva.  
&nbsp;&nbsp;&nbsp;&nbsp;A metodologia envolve revisão bibliográfica, definição de requisitos, modelagem da arquitetura multiagente, integração com sensores virtuais, desenvolvimento de mecanismos de ingestão e armazenamento de dados, além da construção de recursos de visualização e monitoramento, como dashboards e relatórios automáticos.  
&nbsp;&nbsp;&nbsp;&nbsp;Como resultados preliminares, foi configurado o ambiente de simulação no CARLA, com execução de cenários parametrizáveis e integração de sensores como câmeras RGB, LiDAR, GNSS e IMU, permitindo a coleta sincronizada de dados multimodais. Também foi iniciado o desenvolvimento de agentes de inteligência artificial e mecanismos básicos de análise dos dados gerados.  
&nbsp;&nbsp;&nbsp;&nbsp;Os resultados obtidos indicam a viabilidade da infraestrutura proposta como ambiente seguro, escalável e reproduzível para treinamento, validação e supervisão de agentes inteligentes aplicados à condução autônoma, contribuindo para o avanço de soluções em logística autônoma, sistemas multiagentes e manutenção preditiva em veículos off-road.  

# Autores: Tiago Hiray Hisatugo, Daniel Djinishian de Briquez, Eduardo Cunha Santiago, Gabriel Silva Garcia, Eduardo Takase Sawada


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
