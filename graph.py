"""
Grafo de cidades portuguesas com distâncias rodoviárias e heurísticas.

Fonte: Trabalho Prático Final – Métodos de Procura, OCR e LLM (2026)
"""

# ---------------------------------------------------------------------------
# Ligações bidirecionais (cidade_a, cidade_b, distância_km)
# ---------------------------------------------------------------------------
_CONNECTIONS = [
    ("Aveiro", "Porto", 68),
    ("Aveiro", "Viseu", 95),
    ("Aveiro", "Coimbra", 68),
    ("Aveiro", "Leiria", 115),
    ("Braga", "Viana do Castelo", 48),
    ("Braga", "Vila Real", 106),
    ("Braga", "Porto", 53),
    ("Bragança", "Vila Real", 137),
    ("Bragança", "Guarda", 202),
    ("Beja", "Évora", 78),
    ("Beja", "Faro", 152),
    ("Beja", "Setúbal", 142),
    ("Castelo Branco", "Coimbra", 159),
    ("Castelo Branco", "Guarda", 106),
    ("Castelo Branco", "Portalegre", 80),
    ("Castelo Branco", "Évora", 203),
    ("Coimbra", "Viseu", 96),
    ("Coimbra", "Leiria", 67),
    ("Évora", "Lisboa", 150),
    ("Évora", "Santarém", 117),
    ("Évora", "Portalegre", 131),
    ("Évora", "Setúbal", 103),
    ("Faro", "Setúbal", 249),
    ("Faro", "Lisboa", 299),
    ("Guarda", "Vila Real", 157),
    ("Guarda", "Viseu", 85),
    ("Leiria", "Lisboa", 129),
    ("Leiria", "Santarém", 70),
    ("Lisboa", "Santarém", 78),
    ("Lisboa", "Setúbal", 50),
    ("Porto", "Viana do Castelo", 71),
    ("Porto", "Vila Real", 116),
    ("Porto", "Viseu", 133),
    ("Vila Real", "Viseu", 110),
]

# ---------------------------------------------------------------------------
# Heurística: distância em linha reta (km) de cada cidade até Faro
# Tabela 2 do enunciado – usada por Procura Sôfrega e A*
# ---------------------------------------------------------------------------
HEURISTIC_TO_FARO: dict[str, int] = {
    "Aveiro": 366,
    "Braga": 454,
    "Bragança": 487,
    "Beja": 99,
    "Castelo Branco": 280,
    "Coimbra": 319,
    "Évora": 157,
    "Faro": 0,
    "Guarda": 352,
    "Leiria": 278,
    "Lisboa": 195,
    "Portalegre": 228,
    "Porto": 418,
    "Santarém": 231,
    "Setúbal": 168,
    "Viana do Castelo": 473,
    "Vila Real": 429,
    "Viseu": 363,
}


def _build_graph() -> dict[str, dict[str, int]]:
    """Constrói grafo bidirecional a partir da lista de ligações."""
    graph: dict[str, dict[str, int]] = {}
    for city_a, city_b, dist in _CONNECTIONS:
        graph.setdefault(city_a, {})[city_b] = dist
        graph.setdefault(city_b, {})[city_a] = dist
    return graph


def get_heuristic(city: str, goal: str) -> int:
    """
    Devolve a estimativa heurística (distância em linha reta) de *city* até *goal*.

    A heurística exata só está disponível quando goal='Faro'.
    Para outros destinos usa-se uma estimativa conservadora baseada na
    diferença de distâncias a Faro, garantindo admissibilidade.
    """
    if city == goal:
        return 0
    if goal == "Faro":
        return HEURISTIC_TO_FARO.get(city, 0)
    # Estimativa conservadora: |h(city→Faro) - h(goal→Faro)|
    h_city = HEURISTIC_TO_FARO.get(city, 0)
    h_goal = HEURISTIC_TO_FARO.get(goal, 0)
    return abs(h_city - h_goal)


# Grafo global e lista ordenada de cidades
GRAPH: dict[str, dict[str, int]] = _build_graph()
CITIES: list[str] = sorted(GRAPH.keys())
