"""
Trabalho Prático Final – Métodos de Procura, OCR e LLM
IADE / Universidade Europeia – 2026

Ponto de entrada principal da aplicação.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Importações opcionais (rich para output colorido)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box
    from rich.text import Text
    from rich.prompt import Prompt
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None  # type: ignore

from graph import GRAPH, CITIES, get_heuristic
from algorithms import (
    uniform_cost_search,
    limited_depth_search,
    greedy_search,
    astar_search,
)
from ocr import read_license_plate
from llm_client import get_city_attractions


# ---------------------------------------------------------------------------
# Helpers de output (suportam rich e texto simples)
# ---------------------------------------------------------------------------

def _print(msg: str = "", style: str = "") -> None:
    if _RICH:
        console.print(msg, style=style)  # type: ignore
    else:
        print(msg)


def _rule(title: str = "") -> None:
    if _RICH:
        console.rule(title)  # type: ignore
    else:
        print(f"\n{'─' * 60}  {title}")


def _panel(content: str, title: str = "", style: str = "bold cyan") -> None:
    if _RICH:
        console.print(Panel(content, title=title, border_style=style))  # type: ignore
    else:
        print(f"\n[{title}]\n{content}\n")


# ---------------------------------------------------------------------------
# Banner inicial
# ---------------------------------------------------------------------------

_BANNER = r"""
 ____                              ____            _
|  _ \ _ __ ___   ___ _   _ _ __ |  _ \  ___  ___| |_ ___
| |_) | '__/ _ \ / __| | | | '__|| | | |/ _ \/ __| __/ __|
|  __/| | | (_) | (__| |_| | |   | |_| |  __/\__ \ |_\__ \
|_|   |_|  \___/ \___|\__,_|_|   |____/ \___||___/\__|___/

    Métodos de Procura  ·  OCR  ·  LLM          v1.0  2026
"""


def _show_banner() -> None:
    if _RICH:
        console.print(_BANNER, style="bold blue")  # type: ignore
    else:
        print(_BANNER)


# ---------------------------------------------------------------------------
# Selecção de cidade
# ---------------------------------------------------------------------------

def _select_city(prompt_text: str, exclude: str | None = None) -> str:
    """Apresenta lista numerada de cidades e pede selecção ao utilizador."""
    available = [c for c in CITIES if c != exclude]

    if _RICH:
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Num", style="dim", width=4)
        table.add_column("Cidade")
        for i, city in enumerate(available, 1):
            table.add_row(f"{i}.", city)
        console.print(table)  # type: ignore
    else:
        for i, city in enumerate(available, 1):
            print(f"  {i:2}. {city}")

    while True:
        try:
            choice = int(input(f"\n{prompt_text} [1-{len(available)}]: "))
            if 1 <= choice <= len(available):
                return available[choice - 1]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Selecção inválida, tente novamente.")


# ---------------------------------------------------------------------------
# Apresentação das iterações
# ---------------------------------------------------------------------------

def _display_iterations(iterations: list[dict], algo_name: str, show_heuristic: bool) -> None:
    _rule(f"Iterações – {algo_name}")

    if _RICH:
        table = Table(box=box.MINIMAL_DOUBLE_HEAD, show_lines=False)
        table.add_column("Passo", style="dim", justify="right", width=6)
        table.add_column("Cidade", style="cyan", width=20)
        table.add_column("Prof.", justify="right", width=6)
        table.add_column("g(n)", justify="right", width=8)
        if show_heuristic:
            table.add_column("h(n)", justify="right", width=8)
            table.add_column("f(n)", justify="right", width=8)
        table.add_column("Caminho parcial", style="dim")

        for it in iterations:
            path_str = " → ".join(it["path"])
            row = [
                str(it["step"]),
                it["node"],
                str(it["depth"]),
                str(it["g"]),
            ]
            if show_heuristic:
                row += [str(it["h"]), str(it["f"])]
            row.append(path_str)
            table.add_row(*row)

        console.print(table)  # type: ignore
    else:
        header = f"{'Passo':>5}  {'Cidade':<22}  {'Prof':>5}  {'g(n)':>7}"
        if show_heuristic:
            header += f"  {'h(n)':>7}  {'f(n)':>7}"
        header += "  Caminho"
        print(header)
        print("-" * len(header))
        for it in iterations:
            path_str = " → ".join(it["path"])
            row = f"{it['step']:>5}  {it['node']:<22}  {it['depth']:>5}  {it['g']:>7}"
            if show_heuristic:
                row += f"  {it['h']:>7}  {it['f']:>7}"
            row += f"  {path_str}"
            print(row)


# ---------------------------------------------------------------------------
# Apresentação do resultado final
# ---------------------------------------------------------------------------

def _display_result(
    path: list[str] | None,
    cost: float,
    start: str,
    goal: str,
    algo_name: str,
    plate: str,
) -> None:
    _rule("Resultado")

    if path is None:
        msg = f"[bold red]❌  Não foi encontrado caminho de {start} para {goal}.[/bold red]"
        _print(msg)
        return

    path_str = " → ".join(path)
    details = (
        f"[bold]Algoritmo:[/bold] {algo_name}\n"
        f"[bold]Matrícula:[/bold] {plate}\n"
        f"[bold]Origem:[/bold]    {start}\n"
        f"[bold]Destino:[/bold]   {goal}\n"
        f"[bold]Caminho:[/bold]   {path_str}\n"
        f"[bold]Distância:[/bold] {cost} km\n"
        f"[bold]Cidades:[/bold]   {len(path)}"
    )
    if _RICH:
        console.print(Panel(details, title="✅  Caminho Encontrado", border_style="green"))  # type: ignore
    else:
        plain = details.replace("[bold]", "").replace("[/bold]", "")
        print(f"\n✅  Caminho Encontrado\n{plain}")


# ---------------------------------------------------------------------------
# Atrações turísticas via LLM
# ---------------------------------------------------------------------------

def _display_attractions(cities: list[str]) -> None:
    _rule("Atrações Turísticas (LLM)")
    _print("A consultar o LLM local… (pode demorar alguns segundos)\n", style="dim")

    for city in cities:
        attractions = get_city_attractions(city)
        if _RICH:
            console.print(Panel(attractions, title=f"📍 {city}", border_style="yellow"))  # type: ignore
        else:
            print(f"\n📍 {city}\n{attractions}\n")


# ---------------------------------------------------------------------------
# Loop principal de pesquisa
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "1": ("Custo Uniforme (UCS)",       False),
    "2": ("Profundidade Limitada (DLS)", False),
    "3": ("Procura Sôfrega (Greedy)",    True),
    "4": ("A*",                          True),
}


def _run_search(start: str, goal: str, plate: str) -> None:
    """Pede algoritmo, corre pesquisa e mostra resultados."""

    _rule("Selecção do Algoritmo")
    for key, (name, uses_h) in ALGORITHMS.items():
        heuristic_note = " [heurística]" if uses_h else ""
        _print(f"  {key}. {name}{heuristic_note}")

    algo_key = ""
    while algo_key not in ALGORITHMS:
        algo_key = input("\nAlgoritmo [1-4]: ").strip()

    algo_name, uses_heuristic = ALGORITHMS[algo_key]

    # Profundidade limite para DLS
    max_depth = 10
    if algo_key == "2":
        while True:
            try:
                max_depth = int(input("Profundidade máxima [padrão=10]: ").strip() or "10")
                if max_depth > 0:
                    break
            except ValueError:
                pass
            print("  Valor inválido.")

    _print(f"\n▶  A executar {algo_name}: {start} → {goal}\n", style="bold")

    # Executar algoritmo
    if algo_key == "1":
        path, cost, iterations = uniform_cost_search(GRAPH, start, goal)
    elif algo_key == "2":
        path, cost, iterations = limited_depth_search(GRAPH, start, goal, max_depth)
    elif algo_key == "3":
        path, cost, iterations = greedy_search(GRAPH, start, goal, get_heuristic)
    else:
        path, cost, iterations = astar_search(GRAPH, start, goal, get_heuristic)

    # Iterações
    _display_iterations(iterations, algo_name, uses_heuristic)

    # Resultado
    _display_result(path, cost, start, goal, algo_name, plate)

    # Atrações das cidades do caminho
    if path:
        show_att = input("\nMostrar atrações turísticas das cidades do caminho? [S/n]: ").strip().lower()
        if show_att in ("", "s", "sim", "y", "yes"):
            _display_attractions(path)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    _show_banner()

    # --- Login via OCR de matrícula ---
    _rule("Login")
    _print("O login é feito através da leitura OCR da matrícula do veículo.\n")

    image_path: str | None = None
    use_image = input("Tem uma imagem da matrícula? [s/N]: ").strip().lower()
    if use_image in ("s", "sim", "y", "yes"):
        image_path = input("Caminho para a imagem: ").strip().strip('"')

    plate = read_license_plate(image_path)
    _print(f"\n✅  Sessão iniciada · Matrícula: [bold green]{plate}[/bold green]\n")

    # --- Loop principal ---
    while True:
        _rule("Nova Pesquisa")

        _print("\n[bold]Seleccione a cidade de origem:[/bold]")
        start = _select_city("Origem")

        _print("\n[bold]Seleccione a cidade de destino:[/bold]")
        goal = _select_city("Destino", exclude=start)

        _run_search(start, goal, plate)

        _rule()
        again = input("\nRealizar nova pesquisa? [S/n]: ").strip().lower()
        if again not in ("", "s", "sim", "y", "yes"):
            break

    _print("\nAté logo! 👋\n", style="bold blue")


if __name__ == "__main__":
    main()
