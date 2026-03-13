"""Quick smoke test for the search algorithms."""
from graph import GRAPH, CITIES, get_heuristic
from algorithms import (
    uniform_cost_search,
    limited_depth_search,
    greedy_search,
    astar_search,
)

sep = " -> "

pairs = [("Porto", "Faro"), ("Braga", "Faro"), ("Bragança", "Lisboa")]

for start, goal in pairs:
    print(f"\n=== {start} -> {goal} ===")

    path, cost, iters = uniform_cost_search(GRAPH, start, goal)
    print(f"UCS    : {sep.join(path) if path else 'None'}, {cost} km, {len(iters)} iter")

    path2, cost2, iters2 = limited_depth_search(GRAPH, start, goal, max_depth=8)
    print(f"DLS(8) : {sep.join(path2) if path2 else 'None'}, {cost2} km, {len(iters2)} iter")

    path3, cost3, iters3 = greedy_search(GRAPH, start, goal, get_heuristic)
    print(f"Greedy : {sep.join(path3) if path3 else 'None'}, {cost3} km, {len(iters3)} iter")

    path4, cost4, iters4 = astar_search(GRAPH, start, goal, get_heuristic)
    print(f"A*     : {sep.join(path4) if path4 else 'None'}, {cost4} km, {len(iters4)} iter")

print("\nAll algorithms ran successfully.")
