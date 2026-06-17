import os
import json
import time
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import deque
from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

# Diretório base dos dados
DATA_DIR = Path(r"C:\Users\danie\Documents\tcc\git\TCC_MOVER_AGENTES_IA\src\data")
st.set_page_config(page_title="POC agentes projeto MOVER", layout="wide")

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

@st.cache_resource
def carregar_modelo():
    try:
        caminho_modelo = DATA_DIR / "modelo_especialista.joblib"
        return joblib.load(caminho_modelo)
    except FileNotFoundError:
        st.error("Erro: Modelo 'modelo_especialista.joblib' não encontrado.")
        st.stop()

modelo_ml = carregar_modelo()
FEATURES_ML = ['speed_mps', 'throttle', 'brake', 'steer', 'acc_x', 'acc_y', 'imu_gyro_z']

TAMANHO_BUFFER = 20
buffer_frames = deque(maxlen=TAMANHO_BUFFER)

def salvar_log_json(dados_log):
    arquivo = DATA_DIR / "logs_essenciais.json"
    logs = []
    if os.path.exists(arquivo):
        with open(arquivo, "r", encoding="utf-8") as f:
            logs = json.load(f)
    
    logs.append(dados_log)
    
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

class AgentState(TypedDict):
    dados_sensores: dict
    lote_contexto: List[dict]
    status: Optional[str]
    diagnostico: Optional[str]
    comandos: Optional[str]

class RespostaSupervisor(BaseModel):
    diagnostico: str = Field(description="Diagnóstico técnico curto (máx 15 palavras)")
    comandos: str = Field(description="Ação corretiva sugerida")

llm_estruturado = llm.with_structured_output(RespostaSupervisor)

def especialista_manutencao(state: AgentState):
    dados_atuais = state["dados_sensores"]
    frame_atual = dados_atuais.get("frame", 0)
    
    buffer_frames.append(dados_atuais)
    
    valores_frame = [dados_atuais.get(f, 0.0) for f in FEATURES_ML]
    df_frame = pd.DataFrame([valores_frame], columns=FEATURES_ML)
    predicao = modelo_ml.predict(df_frame)[0]
    
    ultimo_log = st.session_state.get("ultimo_frame_alerta", -9999)
    
    if predicao == -1 and (frame_atual - ultimo_log >= 20):
        st.session_state.ultimo_frame_alerta = frame_atual
        return {"status": "Outlier", "lote_contexto": list(buffer_frames)}
    
    return {"status": "OK", "lote_contexto": []}

def supervisor_interpreta(state: AgentState):
    lote = state["lote_contexto"]
    prompt = f"IA supervisora. Anomalia severa detectada. Contexto recente (5 frames): {lote}. Forneça diagnóstico e ação."
    resposta = llm_estruturado.invoke(prompt)
    return {"diagnostico": resposta.diagnostico, "comandos": resposta.comandos}

def roteador_status(state: AgentState):
    if state["status"] == "Outlier":
        return "supervisor"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("especialista", especialista_manutencao)
workflow.add_node("supervisor", supervisor_interpreta)
workflow.set_entry_point("especialista")
workflow.add_conditional_edges("especialista", roteador_status)
workflow.add_edge("supervisor", END)
app_grafo = workflow.compile()

st.title("POC Agentes projeto MOVER")

col_grafico, col_logs = st.columns([2, 1])

with col_grafico:
    st.subheader("Telemetria Dinâmica")
    grafico_placeholder = st.empty()

with col_logs:
    st.subheader("Registro de Atividades")
    logs_placeholder = st.empty()

if "simulando" not in st.session_state:
    st.session_state.simulando = False
if "feed_agentes" not in st.session_state:
    st.session_state.feed_agentes = deque(maxlen=20)

def iniciar_simulacao():
    st.session_state.simulando = True
    st.session_state.feed_agentes.clear()

st.sidebar.header("Controles")
st.sidebar.button("Iniciar Simulação", on_click=iniciar_simulacao, type="primary", use_container_width=True)

if st.session_state.simulando:
    try:
        caminho_telemetria = DATA_DIR / "telemetria.csv"
        df = pd.read_csv(caminho_telemetria)
        total_frames = len(df)
    except FileNotFoundError:
        st.error("Arquivo telemetria.csv não encontrado.")
        st.stop()
    
    frames_hist = []
    speed_hist = []
    acc_x_hist = []
    outliers_frames = []
    outliers_vals = []
    
    buffer_frames.clear()
    st.session_state.ultimo_frame_alerta = -9999
    
    with col_grafico:
        st.markdown("**Processando fluxo de telemetria...**")
        barra_progresso = st.progress(0.0)
    
    for index, row in df.iterrows():
        dados_dict = row.to_dict()
        frame_atual = dados_dict["frame"]
        
        frames_hist.append(frame_atual)
        speed_hist.append(dados_dict["speed_mps"])
        acc_x_hist.append(dados_dict["acc_x"])
        
        houve_atualizacao_log = False
        
        for step in app_grafo.stream({"dados_sensores": dados_dict, "lote_contexto": []}):
            
            if "especialista" in step:
                if step["especialista"]["status"] == "Outlier":
                    msg_esp = f"**Especialista (ML):** Alerta crítico detectado no frame {frame_atual}."
                    st.session_state.feed_agentes.appendleft({"papel": "Especialista", "texto": msg_esp})
                    houve_atualizacao_log = True
            
            if "supervisor" in step:
                diag = step["supervisor"]["diagnostico"]
                acao = step["supervisor"]["comandos"]
                msg_sup = f"**Supervisor (LLM):**\n- Diagnóstico: {diag}\n- Ação: {acao}"
                st.session_state.feed_agentes.appendleft({"papel": "Supervisor", "texto": msg_sup})
                houve_atualizacao_log = True
                
                outliers_frames.append(frame_atual)
                outliers_vals.append(dados_dict["speed_mps"])
                
                salvar_log_json({
                    "frame": frame_atual,
                    "tempo": dados_dict.get("sim_time", 0),
                    "diagnostico": diag,
                    "acao": acao
                })

        if houve_atualizacao_log:
            with logs_placeholder.container(height=500):
                for msg in st.session_state.feed_agentes:
                    if msg["papel"] == "Especialista":
                        st.warning(msg["texto"], icon=None)
                    else:
                        st.error(msg["texto"], icon=None)
        
        barra_progresso.progress((index + 1) / total_frames)
            
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames_hist, y=speed_hist, mode='lines', name='Velocidade (m/s)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=frames_hist, y=acc_x_hist, mode='lines', name='Aceleração X', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=outliers_frames, y=outliers_vals, mode='markers', name='Alerta', marker=dict(color='red', size=12, symbol='x')))
    
    fig.update_layout(
        height=400, 
        margin=dict(l=0, r=0, t=30, b=0), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Frames",
        yaxis_title="Valores (m/s e m/s²)"
    )
    
    grafico_placeholder.plotly_chart(fig, use_container_width=True)
    
    st.success("Simulação concluída. Visualização de dados gerada.", icon=None)
    st.session_state.simulando = False