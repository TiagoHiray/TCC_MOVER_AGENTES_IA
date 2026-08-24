import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[2]
ARQUIVO_LOGS = BASE_DIR / "data" / "logs_essenciais.json"

load_dotenv(BASE_DIR / ".env")
client = OpenAI()

class AuditoriaAlucinacao(BaseModel):
    contem_alucinacao: bool = Field(description="True se o resumo inventou causas mecânicas ou inseriu fatos ausentes. False se for totalmente fiel.")
    tipo_alucinacao: str
    justificativa: str

def auditar_resultados(caminho_csv: Path) -> pd.DataFrame:
    with open(ARQUIVO_LOGS, 'r', encoding='utf-8') as f:
        dados_brutos = json.load(f)
    
    gabaritos = {float(item['frame']): f"Diagnóstico: {item['diagnostico']} | Ação: {item['acao']}" for item in dados_brutos}
    
    df_resultados = pd.read_csv(caminho_csv)
    auditorias = []
    
    print("\nIniciando auditoria de alucinações via API GPT-4o...")
    
    for index, row in df_resultados.iterrows():
        modelo_local = row['modelo']
        frame = float(row['frame_evento'])
        texto_original = gabaritos.get(frame, "")
        
        prompt = (
            f"Você é um auditor de dados de telemetria.\n\n"
            f"LOG ORIGINAL:\n{texto_original}\n\n"
            f"RESUMO GERADO PELO MODELO:\n{row['resumo_gerado']}\n\n"
            f"Sua tarefa: Identifique se o resumo contém alucinações GRAVES.\n"
            f"- Tolerância (NÃO é alucinação): Mudança na estrutura da frase, uso de sinônimos, ou adições lógicas básicas de contexto (ex: dizer 'o veículo apresentou', ou 'a telemetria detectou').\n"
            f"- Alucinação (É alucinação): Invenção de problemas físicos ou peças que NÃO estavam no log original (ex: citar falha de transmissão, falta de óleo, problemas de suspensão, roda, etc), ou alterar a gravidade/ação recomendada.\n\n"
            f"Responda estritamente se há alucinações GRAVES (true) ou se o sentido técnico original foi mantido (false)."
        )
        
        try:
            resposta = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format=AuditoriaAlucinacao,
                temperature=0.0
            )
            
            resultado = resposta.choices[0].message.parsed
            auditorias.append({
                "modelo": modelo_local,
                "contem_alucinacao": resultado.contem_alucinacao,
            })
        except Exception as e:
            print(f"[ERRO] Falha na auditoria da linha {index}: {e}")
            
    df_auditoria = pd.DataFrame(auditorias)
    
    resumo_alucinacao = df_auditoria.groupby('modelo')['contem_alucinacao'].agg(
        total_alucinacoes='sum',
        taxa_alucinacao_pct=lambda x: (x.sum() / len(x)) * 100
    ).reset_index()
    
    return resumo_alucinacao