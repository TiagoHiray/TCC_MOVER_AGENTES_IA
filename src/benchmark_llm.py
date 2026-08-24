import json
import time
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field
import ollama

# --- CONFIGURAÇÕES E PATHS ---
MODELOS_TESTE = ["qwen2.5:0.5b", "qwen2.5:1.5b", "gemma2:2b", "llama3.2:1b"]

# Path resolve para TCC_MOVER_AGENTES_IA/data/logs_essenciais.json
BASE_DIR = Path(__file__).resolve().parents[1]
ARQUIVO_LOGS = BASE_DIR / "data" / "logs_essenciais.json"

PASTA_RESULTADOS = BASE_DIR / "data" / "resultados_pesquisa"
PASTA_RESULTADOS.mkdir(exist_ok=True)

OPCOES_OLLAMA = {
    "num_ctx": 512, 
    "temperature": 0.1 
}

# --- REDIRECIONAMENTO DE LOGS PARA TXT ---
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = PASTA_RESULTADOS / f"benchmark-{timestamp}.txt"
sys.stdout = Logger(log_filename)

# --- MODELOS PYDANTIC ---
class EventoLog(BaseModel):
    frame: float
    tempo_simulacao: float = Field(alias='tempo', default=0.0)
    diagnostico: str
    acao: str

class ResultadoAvaliacao(BaseModel):
    modelo: str
    frame_evento: float
    resumo_gerado: str
    latencia_total_s: float
    tempo_processamento_prompt_ms: float 
    tempo_geracao_ms: float 
    total_tokens_gerados: int
    
    @property
    def throughput_tps(self) -> float:
        tempo_s = self.tempo_geracao_ms / 1e9
        if tempo_s > 0:
            return self.total_tokens_gerados / tempo_s
        return 0.0

# --- PIPELINE PRINCIPAL ---
def carregar_golden_dataset(caminho: Path) -> list[EventoLog]:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    
    with open(caminho, 'r', encoding='utf-8') as f:
        dados_brutos = json.load(f)
        
    return [EventoLog(**item) for item in dados_brutos[:20]] 

def formatar_prompt(evento: EventoLog) -> str:
    return (
        f"Você é um especialista em telemetria veicular. Resuma o evento abaixo de forma concisa "
        f"mantendo a linguagem técnica.\n\n"
        f"Diagnóstico: {evento.diagnostico}\n"
        f"Ação Recomendada: {evento.acao}\n\n"
        f"Resumo Conciso:"
    )

def avaliar_modelo(modelo_nome: str, dataset: list[EventoLog]) -> list[ResultadoAvaliacao]:
    print(f"\n[{modelo_nome}] Carregando modelo na RAM e rodando Warm-up...")
    _ = ollama.generate(model=modelo_nome, prompt="Diga ola", options=OPCOES_OLLAMA)
    
    resultados = []
    print(f"[{modelo_nome}] Iniciando avaliação de {len(dataset)} logs...")
    
    for evento in dataset:
        prompt = formatar_prompt(evento)
        
        inicio_wall_clock = time.time()
        resposta = ollama.generate(
            model=modelo_nome, 
            prompt=prompt,
            options=OPCOES_OLLAMA
        )
        fim_wall_clock = time.time()
        
        latencia = fim_wall_clock - inicio_wall_clock
        
        resultado = ResultadoAvaliacao(
            modelo=modelo_nome,
            frame_evento=evento.frame,
            resumo_gerado=resposta['response'].strip(),
            latencia_total_s=latencia,
            tempo_processamento_prompt_ms=resposta.get('prompt_eval_duration', 0),
            tempo_geracao_ms=resposta.get('eval_duration', 0),
            total_tokens_gerados=resposta.get('eval_count', 0)
        )
        
        resultados.append(resultado)
        
    print(f"[{modelo_nome}] Concluído.")
    return resultados

def executar_pesquisa():
    print(f"Iniciando Pipeline de Avaliação de LLMs - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Arquivo de log: {log_filename}")
    
    try:
        dataset = carregar_golden_dataset(ARQUIVO_LOGS)
    except Exception as e:
        print(f"[ERRO FATAL] {e}")
        sys.exit(1)
        
    todos_resultados = []
    for modelo in MODELOS_TESTE:
        try:
             res_modelo = avaliar_modelo(modelo, dataset)
             todos_resultados.extend(res_modelo)
        except Exception as e:
            print(f"[ERRO] Falha ao testar o modelo {modelo}: {e}")
            
    df_resultados = pd.DataFrame([r.model_dump() for r in todos_resultados])
    df_resultados['throughput_tps'] = [r.throughput_tps for r in todos_resultados]
    
    caminho_csv = PASTA_RESULTADOS / f"metricas_operacionais_{timestamp}.csv"
    df_resultados.to_csv(caminho_csv, index=False)
    
    print(f"\n[OK] Pesquisa operacional concluída! Resultados salvos em: {caminho_csv}")

    print("\nResumo do Throughput (Tokens por Segundo) Mínimo/Médio/Máximo:")
    resumo_tps = df_resultados.groupby('modelo')['throughput_tps'].agg(['min', 'mean', 'max']).round(2)
    print(resumo_tps)

if __name__ == "__main__":
    executar_pesquisa()