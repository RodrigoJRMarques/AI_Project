"""
Cliente LLM local via Ollama API.

Requer Ollama instalado e em execução em http://localhost:11434
  → https://ollama.com

Para transferir um modelo:  ollama pull llama3.2
"""

import json
import urllib.request
import urllib.error
from typing import Optional

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Utilitários de conectividade
# ---------------------------------------------------------------------------

def _is_ollama_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _available_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=2) as r:
            data = json.loads(r.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _best_model(preferred: str) -> str:
    """Devolve *preferred* se disponível, caso contrário o primeiro modelo instalado."""
    available = _available_models()
    if not available:
        return preferred
    # Aceita nome parcial (ex: "llama3.2" corresponde a "llama3.2:latest")
    for m in available:
        if m.startswith(preferred):
            return m
    return available[0]


# ---------------------------------------------------------------------------
# Chamada ao LLM
# ---------------------------------------------------------------------------

def get_city_attractions(city: str, model: str = DEFAULT_MODEL) -> str:
    """
    Devolve as 3 principais atrações turísticas de *city* usando um LLM local.

    Usa Ollama como backend. Se não estiver disponível, devolve informação
    pré-definida de fallback.

    Args:
        city:  Nome da cidade portuguesa.
        model: Modelo Ollama a utilizar (padrão: llama3.2).

    Returns:
        String com a descrição das atrações.
    """
    if not _is_ollama_running():
        return _fallback(city)

    model = _best_model(model)
    prompt = (
        f"Indica as 3 principais atrações turísticas de {city}, Portugal. "
        "Responde em português, com uma lista numerada e uma frase curta por atração."
    )
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 250},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
            return data.get("response", _fallback(city)).strip()
    except Exception:
        return _fallback(city)


# ---------------------------------------------------------------------------
# Informações de fallback (quando Ollama não está disponível)
# ---------------------------------------------------------------------------

_FALLBACK: dict[str, str] = {
    "Aveiro": (
        "1. Canais de Aveiro e passeio de moliceiro\n"
        "2. Museu de Aveiro (Convento de Jesus)\n"
        "3. Praias da Costa Nova e Barra"
    ),
    "Beja": (
        "1. Castelo de Beja e Torre de Menagem\n"
        "2. Museu Regional Rainha Dona Leonor\n"
        "3. Convento de Nossa Senhora da Conceição"
    ),
    "Braga": (
        "1. Santuário do Bom Jesus do Monte\n"
        "2. Sé de Braga (catedral mais antiga de Portugal)\n"
        "3. Jardins do Palácio do Raio"
    ),
    "Bragança": (
        "1. Castelo de Bragança e Cidadela Medieval\n"
        "2. Museu Ibérico da Máscara e do Traje\n"
        "3. Parque Natural de Montesinho"
    ),
    "Castelo Branco": (
        "1. Jardim do Paço Episcopal\n"
        "2. Castelo de Castelo Branco\n"
        "3. Museu Francisco Tavares Proença Júnior"
    ),
    "Coimbra": (
        "1. Universidade de Coimbra (Património UNESCO)\n"
        "2. Biblioteca Joanina\n"
        "3. Sé Velha de Coimbra"
    ),
    "Évora": (
        "1. Templo Romano de Évora\n"
        "2. Sé Catedral de Évora\n"
        "3. Cromeleque dos Almendres (megalítico)"
    ),
    "Faro": (
        "1. Cidade Velha (centro histórico murado)\n"
        "2. Ria Formosa e Ilha Deserta\n"
        "3. Museu Municipal de Faro"
    ),
    "Guarda": (
        "1. Sé Catedral da Guarda\n"
        "2. Castelo e Torre de Menagem\n"
        "3. Museu da Guarda"
    ),
    "Leiria": (
        "1. Castelo de Leiria\n"
        "2. Pinhal de D. Dinis (Mata Nacional)\n"
        "3. Sé de Leiria"
    ),
    "Lisboa": (
        "1. Torre de Belém (Património UNESCO)\n"
        "2. Mosteiro dos Jerónimos\n"
        "3. Castelo de São Jorge"
    ),
    "Portalegre": (
        "1. Castelo de Portalegre\n"
        "2. Sé Catedral de Portalegre\n"
        "3. Museu da Tapeçaria de Portalegre"
    ),
    "Porto": (
        "1. Torre dos Clérigos\n"
        "2. Livraria Lello\n"
        "3. Ribeira do Porto (Património UNESCO)"
    ),
    "Santarém": (
        "1. Igreja de Santa Maria da Graça\n"
        "2. Portas do Sol (jardim panorâmico)\n"
        "3. Igreja de São João de Alporão (museu)"
    ),
    "Setúbal": (
        "1. Parque Natural da Arrábida\n"
        "2. Igreja de Jesus de Setúbal\n"
        "3. Forte de São Filipe"
    ),
    "Viana do Castelo": (
        "1. Basílica de Santa Luzia\n"
        "2. Praça da República\n"
        "3. Museu de Viana do Castelo"
    ),
    "Vila Real": (
        "1. Solar de Mateus (arquitetura barroca)\n"
        "2. Ermida de São Lourenço\n"
        "3. Parque Natural do Alvão"
    ),
    "Viseu": (
        "1. Museu de Grão Vasco\n"
        "2. Sé de Viseu\n"
        "3. Parque do Fontelo"
    ),
}


def _fallback(city: str) -> str:
    return _FALLBACK.get(
        city,
        f"1. Centro histórico de {city}\n2. Museu Municipal\n3. Igreja Matriz",
    )
