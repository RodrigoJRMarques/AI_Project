"""
Algoritmos de procura implementados:
  1. Custo Uniforme (UCS)
  2. Profundidade Limitada (DLS)
  3. Procura Sôfrega (Greedy Best-First)
  4. A*

Cada função devolve: (caminho | None, custo_total, iterações)

Estrutura de cada iteração:
  {
    "step"  : int,        # número da iteração
    "node"  : str,        # cidade em expansão
    "path"  : list[str],  # caminho até aqui
    "g"     : float,      # custo acumulado g(n)
    "h"     : float,      # heurística h(n)
    "f"     : float,      # f(n) = g(n) + h(n)
    "depth" : int,        # profundidade no grafo
  }
"""

import heapq
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# 1. Custo Uniforme (Uniform Cost Search)
# ---------------------------------------------------------------------------

def uniform_cost_search(
    graph: dict,
    start: str,
    goal: str,
) -> tuple[Optional[list[str]], float, list[dict]]:
    """
    Expande sempre o nó com menor custo acumulado g(n).
    Garante o caminho ótimo.
    """
    iterations: list[dict] = []
    counter = 0  # desempate na heap
    # (g, counter, nó, caminho)
    heap = [(0, counter, start, [start])]
    closed: set[str] = set()

    while heap:
        g, _, node, path = heapq.heappop(heap)

        if node in closed:
            continue
        closed.add(node)

        iterations.append({
            "step": len(iterations) + 1,
            "node": node,
            "path": path[:],
            "g": g,
            "h": 0,
            "f": g,
            "depth": len(path) - 1,
        })

        if node == goal:
            return path, g, iterations

        for neighbor, weight in sorted(graph[node].items()):
            if neighbor not in closed:
                counter += 1
                heapq.heappush(heap, (g + weight, counter, neighbor, path + [neighbor]))

    return None, float("inf"), iterations


# ---------------------------------------------------------------------------
# 2. Profundidade Limitada (Limited Depth Search / DLS)
# ---------------------------------------------------------------------------

def limited_depth_search(
    graph: dict,
    start: str,
    goal: str,
    max_depth: int = 10,
) -> tuple[Optional[list[str]], float, list[dict]]:
    """
    DFS com limite de profundidade *max_depth*.
    Não garante caminho ótimo.
    """
    iterations: list[dict] = []
    # (nó, caminho, profundidade, custo)
    stack = [(start, [start], 0, 0)]

    while stack:
        node, path, depth, cost = stack.pop()

        iterations.append({
            "step": len(iterations) + 1,
            "node": node,
            "path": path[:],
            "g": cost,
            "h": 0,
            "f": cost,
            "depth": depth,
        })

        if node == goal:
            return path, cost, iterations

        if depth < max_depth:
            # Inverter para manter ordem alfabética na exploração
            for neighbor, weight in sorted(graph[node].items(), reverse=True):
                if neighbor not in path:  # evita ciclos no caminho atual
                    stack.append((
                        neighbor,
                        path + [neighbor],
                        depth + 1,
                        cost + weight,
                    ))

    return None, float("inf"), iterations


# ---------------------------------------------------------------------------
# 3. Procura Sôfrega (Greedy Best-First Search)
# ---------------------------------------------------------------------------

def greedy_search(
    graph: dict,
    start: str,
    goal: str,
    heuristic_fn: Callable[[str, str], float],
) -> tuple[Optional[list[str]], float, list[dict]]:
    """
    Expande sempre o nó com menor valor heurístico h(n).
    Rápido mas não garante caminho ótimo.
    """
    iterations: list[dict] = []
    counter = 0
    h_start = heuristic_fn(start, goal)
    # (h, counter, nó, caminho, g)
    heap = [(h_start, counter, start, [start], 0)]
    closed: set[str] = set()

    while heap:
        h, _, node, path, g = heapq.heappop(heap)

        if node in closed:
            continue
        closed.add(node)

        iterations.append({
            "step": len(iterations) + 1,
            "node": node,
            "path": path[:],
            "g": g,
            "h": h,
            "f": h,  # Sôfrega usa apenas h como prioridade
            "depth": len(path) - 1,
        })

        if node == goal:
            return path, g, iterations

        for neighbor, weight in sorted(graph[node].items()):
            if neighbor not in closed:
                counter += 1
                h_n = heuristic_fn(neighbor, goal)
                heapq.heappush(heap, (h_n, counter, neighbor, path + [neighbor], g + weight))

    return None, float("inf"), iterations


# ---------------------------------------------------------------------------
# 4. A* Search
# ---------------------------------------------------------------------------

def astar_search(
    graph: dict,
    start: str,
    goal: str,
    heuristic_fn: Callable[[str, str], float],
) -> tuple[Optional[list[str]], float, list[dict]]:
    """
    Expande o nó com menor f(n) = g(n) + h(n).
    Ótimo com heurística admissível e consistente.
    """
    iterations: list[dict] = []
    counter = 0
    h_start = heuristic_fn(start, goal)
    # (f, counter, nó, caminho, g)
    heap = [(h_start, counter, start, [start], 0)]
    closed: set[str] = set()

    while heap:
        f, _, node, path, g = heapq.heappop(heap)

        if node in closed:
            continue
        closed.add(node)

        h = heuristic_fn(node, goal)
        iterations.append({
            "step": len(iterations) + 1,
            "node": node,
            "path": path[:],
            "g": g,
            "h": h,
            "f": f,
            "depth": len(path) - 1,
        })

        if node == goal:
            return path, g, iterations

        for neighbor, weight in sorted(graph[node].items()):
            if neighbor not in closed:
                counter += 1
                new_g = g + weight
                h_n = heuristic_fn(neighbor, goal)
                heapq.heappush(heap, (new_g + h_n, counter, neighbor, path + [neighbor], new_g))

    return None, float("inf"), iterations
