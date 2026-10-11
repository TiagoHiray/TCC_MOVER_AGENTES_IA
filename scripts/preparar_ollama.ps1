<#
.SYNOPSIS
  Prepara o LLM local do MOVER no Windows: instala o Ollama (se faltar), liga o serviço, baixa o
  modelo gemma2:2b (o do benchmark do artigo: 25 % de alucinação, ~59 tokens/s), mede a latência e
  grava o .env da raiz do repositório.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\preparar_ollama.ps1
  powershell -ExecutionPolicy Bypass -File scripts\preparar_ollama.ps1 -Modelo qwen2.5:1.5b
#>
param(
  [string]$Modelo = "gemma2:2b",
  [string]$Hospedeiro = "http://localhost:11434",
  [int]$EsperaServicoS = 60
)

$ErrorActionPreference = "Stop"
$Raiz = Split-Path -Parent $PSScriptRoot

function Escrever($texto, $cor = "Cyan") { Write-Host "[MOVER] $texto" -ForegroundColor $cor }

function Servico-NoAr {
  try { Invoke-RestMethod -Uri "$Hospedeiro/api/version" -TimeoutSec 3 | Out-Null; return $true } catch { return $false }
}

# 1) Ollama instalado?
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
  $padrao = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
  if (Test-Path $padrao) {
    $env:Path += ";" + (Split-Path $padrao)
  } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
    Escrever "Instalando o Ollama com o winget (pode pedir confirmação)..."
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    $env:Path += ";" + (Join-Path $env:LOCALAPPDATA "Programs\Ollama")
  } else {
    Escrever "Instale o Ollama em https://ollama.com/download e rode este script de novo." "Yellow"
    exit 1
  }
  $ollama = Get-Command ollama -ErrorAction SilentlyContinue
  if (-not $ollama) { Escrever "O Ollama foi instalado, mas não está no PATH. Abra um terminal novo e rode de novo." "Yellow"; exit 1 }
}
Escrever "Ollama: $($ollama.Source)"

# 2) Serviço no ar?
if (-not (Servico-NoAr)) {
  Escrever "Ligando o serviço do Ollama (ollama serve)..."
  Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden
  $limite = (Get-Date).AddSeconds($EsperaServicoS)
  while (-not (Servico-NoAr)) {
    if ((Get-Date) -gt $limite) { Escrever "O serviço não respondeu em $Hospedeiro." "Red"; exit 1 }
    Start-Sleep -Seconds 1
  }
}
$versao = (Invoke-RestMethod -Uri "$Hospedeiro/api/version").version
Escrever "Serviço do Ollama no ar ($Hospedeiro, versão $versao)."

# 3) Modelo
Escrever "Baixando o modelo $Modelo (na primeira vez, ~1,6 GB)..."
& $ollama.Source pull $Modelo
if ($LASTEXITCODE -ne 0) { Escrever "Falha ao baixar $Modelo." "Red"; exit 1 }

# 4) Aquecimento e latência (a primeira chamada carrega o modelo na memória)
$corpo = @{ model = $Modelo; prompt = "Em uma frase curta: o que é um gêmeo digital?"; stream = $false;
            options = @{ num_predict = 60; temperature = 0.1 } } | ConvertTo-Json -Depth 4
foreach ($rodada in 1..2) {
  $inicio = Get-Date
  $resposta = Invoke-RestMethod -Uri "$Hospedeiro/api/generate" -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($corpo)) `
    -ContentType "application/json; charset=utf-8" -TimeoutSec 300
  $segundos = ((Get-Date) - $inicio).TotalSeconds
  $tokens_s = if ($resposta.eval_duration -gt 0) { [math]::Round($resposta.eval_count / ($resposta.eval_duration / 1e9), 1) } else { 0 }
  $rotulo = if ($rodada -eq 1) { "aquecimento" } else { "medida" }
  Escrever ("Chamada de {0}: {1:N1} s, {2} tokens/s." -f $rotulo, $segundos, $tokens_s)
}
Escrever "Resposta: $($resposta.response.Trim())" "Gray"
if ($segundos -gt 6) {
  Escrever "Acima do prazo de 6 s das decisões (config/autonomo.yaml, decisao.prazo_llm_s): as decisões vão cair na regra. Considere uma GPU ou um modelo menor (qwen2.5:0.5b)." "Yellow"
}

# 5) .env da raiz (mantém as outras linhas)
$env_arquivo = Join-Path $Raiz ".env"
$linhas = @()
if (Test-Path $env_arquivo) { $linhas = Get-Content $env_arquivo | Where-Object { $_ -notmatch '^(MOVER_LLM_PROVEDOR|MOVER_LLM_MODELO|OLLAMA_HOST)=' } }
$linhas += "MOVER_LLM_PROVEDOR=ollama", "MOVER_LLM_MODELO=$Modelo", "OLLAMA_HOST=$Hospedeiro"
Set-Content -Path $env_arquivo -Value $linhas -Encoding UTF8
Escrever "Gravado em $env_arquivo : MOVER_LLM_PROVEDOR=ollama, MOVER_LLM_MODELO=$Modelo." "Green"
Escrever "Pronto. Com o CARLA aberto: cd src; python -m mover.autonomo.dirigir" "Green"
