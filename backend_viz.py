import json
import ast
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

JSON_FILE = "qcprov.json"

with open(JSON_FILE, "r") as f:
    data = json.load(f)

connectivity = data["qubit_connectivity"]
cz_errors = ast.literal_eval(data["two_qubit_gate_error"])["cz"]

G = nx.Graph()
for pair in connectivity:
    q1, q2 = pair[0], pair[1]
    err = cz_errors.get((q1, q2)) or cz_errors.get((q2, q1)) or 0.0
    G.add_edge(q1, q2, error=err)

pos = nx.kamada_kawai_layout(G)

errors = [G[u][v]["error"] for u, v in G.edges()]
error_min, error_max = min(errors), max(errors)
cmap = cm.RdYlGn_r
norm = plt.Normalize(vmin=error_min, vmax=error_max)
edge_colors = [cmap(norm(e)) for e in errors]
edge_widths = [1.5 + (e - error_min) / (error_max - error_min) * 4 for e in errors]

fig, ax = plt.subplots(figsize=(18, 14))

nx.draw_networkx_nodes(G, pos, node_size=400, node_color="steelblue", ax=ax)
nx.draw_networkx_labels(
    G, pos,
    labels={q: q.replace("QB", "") for q in G.nodes()},
    font_size=7, font_color="white", font_weight="bold", ax=ax
)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="CZ Gate Error Rate", shrink=0.6, pad=0.02)

ax.set_title(
    f"IQM Aphrodite — Qubit Connectivity & CZ Error Rates\n"
    f"(error range: {error_min:.4f} – {error_max:.4f})",
    fontsize=14
)
ax.axis("off")
plt.tight_layout()
plt.savefig("backend_connectivity.png", dpi=150, bbox_inches="tight")
print("Saved to backend_connectivity.png")
plt.show()
