"""Provedores de LLM plugáveis para o Supervisor: Ollama (padrão), OpenAI, Gemini ou falso.

Escolha do provedor, em ordem de prioridade:
  1. argumentos de linha de comando (--provedor, --modelo);
  2. variáveis do .env na raiz do repositório (MOVER_LLM_PROVEDOR, MOVER_LLM_MODELO, OLLAMA_HOST,
     OPENAI_API_KEY, GOOGLE_API_KEY);
  3. seção `llm` do config/agentes.yaml.

Todos os provedores devolvem um objeto pydantic validado (saída estruturada em JSON).
O provedor "falso" não chama modelo nenhum: devolve o rascunho que o Supervisor coloca no
prompt. Serve para testar o pipeline inteiro sem LLM instalado.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from mover.config import RAIZ

log = logging.getLogger("mover.agentes")

M = TypeVar("M", bound=BaseModel)

PROVEDORES = ("ollama", "openai", "gemini", "falso")

# Linha do prompt com o rascunho determinístico, em JSON (lida pelo provedor falso)
MARCADOR_RASCUNHO = "Rascunho (JSON):"


class ClienteLLM(Protocol):
    nome: str

    def gerar(self, sistema: str, usuario: str, esquema: type[M]) -> M:
        """Gera uma resposta que segue o esquema pydantic."""
        ...


def carregar_env() -> None:
    """Lê o .env da raiz do repositório; variáveis já definidas no ambiente têm prioridade."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv é opcional
        return
    load_dotenv(RAIZ / ".env", override=False)


def interpretar_json(texto: str, esquema: type[M]) -> M:
    """Valida a resposta do modelo, tolerando texto em volta do JSON.

    Modelos pequenos às vezes escrevem algo antes ou depois do objeto. Se o esquema tiver
    um único campo e não houver JSON, o texto inteiro vira esse campo.
    """
    try:
        return esquema.model_validate_json(texto)
    except ValidationError:
        pass
    trecho = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    if trecho:
        try:
            return esquema.model_validate_json(trecho.group())
        except ValidationError:
            pass
    campos = list(esquema.model_fields)
    if len(campos) == 1 and texto.strip():
        return esquema.model_validate({campos[0]: texto.strip().strip('"')})
    raise ValueError(f"resposta fora do esquema {esquema.__name__}: {texto[:200]!r}")


class OllamaLLM:
    """Modelo local via Ollama (padrão gemma2:2b, como nos benchmarks do projeto)."""

    def __init__(self, modelo: str, host: str | None = None, temperatura: float = 0.1,
                 num_ctx: int = 2048, timeout_s: float = 30.0):
        try:
            import ollama
        except ImportError as erro:
            raise ImportError("instale o pacote 'ollama' (pip install ollama)") from erro
        self.nome = f"ollama:{modelo}"
        self._modelo = modelo
        self._cliente = ollama.Client(host=host, timeout=timeout_s)
        self._opcoes = {"temperature": temperatura, "num_ctx": num_ctx}

    def gerar(self, sistema: str, usuario: str, esquema: type[M]) -> M:
        resposta = self._cliente.chat(
            model=self._modelo,
            messages=[{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
            format=esquema.model_json_schema(),  # saída estruturada (Ollama >= 0.5)
            options=self._opcoes,
        )
        return interpretar_json(resposta.message.content or "", esquema)


class OpenAILLM:
    """OpenAI via LangChain com saída estruturada, como a POC (gpt-4o-mini)."""

    def __init__(self, modelo: str, temperatura: float = 0.1, timeout_s: float = 30.0):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as erro:
            raise ImportError("instale o pacote 'langchain-openai'") from erro
        self.nome = f"openai:{modelo}"
        self._llm = ChatOpenAI(model=modelo, temperature=temperatura, timeout=timeout_s)
        self._estruturados: dict[type, Any] = {}

    def gerar(self, sistema: str, usuario: str, esquema: type[M]) -> M:
        if esquema not in self._estruturados:
            self._estruturados[esquema] = self._llm.with_structured_output(esquema)
        return self._estruturados[esquema].invoke([("system", sistema), ("human", usuario)])


class GeminiLLM:
    """Gemini via SDK google-genai (chave em GOOGLE_API_KEY ou GEMINI_API_KEY)."""

    def __init__(self, modelo: str, temperatura: float = 0.1):
        try:
            from google import genai
            from google.genai import types
        except ImportError as erro:
            raise ImportError("instale o pacote 'google-genai'") from erro
        self.nome = f"gemini:{modelo}"
        self._modelo = modelo
        self._tipos = types
        self._cliente = genai.Client()
        self._temperatura = temperatura

    def gerar(self, sistema: str, usuario: str, esquema: type[M]) -> M:
        config = self._tipos.GenerateContentConfig(
            system_instruction=sistema,
            temperature=self._temperatura,
            response_mime_type="application/json",
            response_schema=esquema,
        )
        resposta = self._cliente.models.generate_content(model=self._modelo, contents=usuario, config=config)
        return interpretar_json(resposta.text or "", esquema)


class FalsoLLM:
    """Provedor determinístico para testes: devolve o rascunho do prompt, sem chamar modelo."""

    def __init__(self, atraso_s: float = 0.0):
        self.nome = "falso"
        self._atraso_s = atraso_s  # simula a latência de um LLM (testes de tempo real)

    def gerar(self, sistema: str, usuario: str, esquema: type[M]) -> M:
        if self._atraso_s > 0:
            time.sleep(self._atraso_s)
        for linha in usuario.splitlines():
            if linha.startswith(MARCADOR_RASCUNHO):
                return esquema.model_validate(json.loads(linha[len(MARCADOR_RASCUNHO):]))
        raise ValueError("prompt sem rascunho para o provedor falso")


def criar_llm(cfg_llm: dict[str, Any], provedor: str | None = None, modelo: str | None = None) -> ClienteLLM:
    """Cria o cliente do provedor escolhido (linha de comando > .env > YAML)."""
    carregar_env()
    provedor_env = (os.getenv("MOVER_LLM_PROVEDOR") or "").strip().lower() or None
    provedor = (provedor or provedor_env or cfg_llm.get("provedor") or "ollama").strip().lower()
    if provedor not in PROVEDORES:
        raise ValueError(f"provedor de LLM desconhecido: {provedor!r} (use {', '.join(PROVEDORES)})")
    if modelo is None:
        modelo_env = (os.getenv("MOVER_LLM_MODELO") or "").strip() or None
        # O modelo do .env só vale para o provedor do .env (evita pedir gemma2:2b à OpenAI)
        if modelo_env and provedor_env in (None, provedor):
            modelo = modelo_env
        else:
            modelo = cfg_llm.get("modelos_padrao", {}).get(provedor)

    temperatura = float(cfg_llm.get("temperatura", 0.1))
    timeout_s = float(cfg_llm.get("timeout_s", 30))
    if provedor == "ollama":
        cfg_ollama = cfg_llm.get("ollama", {})
        return OllamaLLM(modelo, host=os.getenv("OLLAMA_HOST") or cfg_ollama.get("host"), temperatura=temperatura,
                         num_ctx=int(cfg_ollama.get("num_ctx", 2048)), timeout_s=timeout_s)
    if provedor == "openai":
        return OpenAILLM(modelo, temperatura=temperatura, timeout_s=timeout_s)
    if provedor == "gemini":
        return GeminiLLM(modelo, temperatura=temperatura)
    return FalsoLLM(atraso_s=float(os.getenv("MOVER_LLM_FALSO_ATRASO_S") or 0))
