# TCC - MOVER Coordenacao Logistica e Operacional de Veiculos Agricolas Autonomos

&nbsp;&nbsp;&nbsp;&nbsp;O avanço das tecnologias de automação, sensoriamento, inteligência artificial e comunicação veicular tem impulsionado o desenvolvimento de sistemas autônomos aplicados a operações logísticas em ambientes complexos, incluindo cenários agrícolas e off-road (BARRILE et al., 2022).  
&nbsp;&nbsp;&nbsp;&nbsp;  Entretanto, a validação desses sistemas em campo apresenta desafios relacionados a custo, segurança, repetibilidade experimental e exposição a condições dinâmicas de difícil controle.  (PENDLETON et al., 2017; KIRAN et al., 2021).  
&nbsp;&nbsp;&nbsp;&nbsp;Nesse contexto, este trabalho propõe o desenvolvimento de uma infraestrutura cognitiva multiagente para apoio à operação, supervisão e análise preditiva de caminhões autônomos agrícolas no âmbito do Projeto MOVER.  
&nbsp;&nbsp;&nbsp;&nbsp;A solução é baseada em um ambiente virtual de co-simulação utilizando o simulador CARLA, empregado como gêmeo digital para criação, execução e avaliação de cenários operacionais controlados.  
&nbsp;&nbsp;&nbsp;&nbsp;A arquitetura proposta organiza agentes inteligentes em diferentes camadas, contemplando agentes embarcados, responsáveis pelo processamento de dados sensoriais e tomada de decisão local; mecanismos de comunicação cooperativa entre veículos e infraestrutura; e agentes em nuvem, voltados à análise global, supervisão da missão, apoio à coordenação operacional e manutenção preditiva.  
&nbsp;&nbsp;&nbsp;&nbsp;A metodologia envolve revisão bibliográfica, definição de requisitos, modelagem da arquitetura multiagente, integração com sensores virtuais, desenvolvimento de mecanismos de ingestão e armazenamento de dados, além da construção de recursos de visualização e monitoramento, como dashboards e relatórios automáticos.  
&nbsp;&nbsp;&nbsp;&nbsp;Como resultados preliminares, foi configurado o ambiente de simulação no CARLA, com execução de cenários parametrizáveis e integração de sensores como câmeras RGB, LiDAR, GNSS e IMU, permitindo a coleta sincronizada de dados multimodais. Também foi iniciado o desenvolvimento de agentes de inteligência artificial e mecanismos básicos de análise dos dados gerados.  
&nbsp;&nbsp;&nbsp;&nbsp;Os resultados obtidos indicam a viabilidade da infraestrutura proposta como ambiente seguro, escalável e reproduzível para treinamento, validação e supervisão de agentes inteligentes aplicados à condução autônoma, contribuindo para o avanço de soluções em logística autônoma, sistemas multiagentes e manutenção preditiva em veículos off-road.  




# Inicialização

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
