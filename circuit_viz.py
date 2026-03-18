import json
import ast
import re
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import networkx as nx

JSON_FILE = "qcprov.json"

with open(JSON_FILE, "r") as f:
    data = json.load(f)

# --- Parse compiled QASM for active qubits and CZ gate counts ---
qasm = data["compiled_circuit_qasm"]
qubit_layout = data.get("qubit_layout", {})  # e.g. {"q[0]": 5, "q[1]": 6, ...} (1-based QB)

def qasm_key(reg, idx):
    return f"{reg}[{idx}]"

def logical_to_physical(reg, idx):
    key = qasm_key(reg, int(idx))
    if key in qubit_layout:
        return qubit_layout[key]
    # Fallback: q[i] -> i+1, ancilla[j] -> j + (num_q_regs + 1)
    return int(idx) + 1 if reg == "q" else int(idx) + 7

active_qubits = set()
cz_counts = Counter()

for line in qasm.split("\n"):
    line = line.strip()
    if (not line or line.startswith("qreg") or line.startswith("creg")
            or line.startswith("OPENQASM") or line.startswith("include")
            or line.startswith("gate")):
        continue
    refs = re.findall(r"(q|ancilla)\[(\d+)\]", line)
    for reg, idx in refs:
        active_qubits.add(logical_to_physical(reg, idx))
    if line.startswith("cz"):
        if len(refs) == 2:
            a = logical_to_physical(refs[0][0], refs[0][1])
            b = logical_to_physical(refs[1][0], refs[1][1])
            pair = (min(a, b), max(a, b))
            cz_counts[pair] += 1

print(f"Active physical qubits: {sorted(active_qubits)}")
print(f"CZ pair counts: {dict(cz_counts)}")

# --- Build full topology graph ---
connectivity = data["qubit_connectivity"]
cz_errors = ast.literal_eval(data["two_qubit_gate_error"])["cz"]

G = nx.Graph()
for i in range(1, data["number_of_qubits"] + 1):
    G.add_node(f"QB{i}")

for pair in connectivity:
    q1, q2 = pair[0], pair[1]
    err = cz_errors.get((q1, q2)) or cz_errors.get((q2, q1)) or 0.0
    G.add_edge(q1, q2, error=err)

pos = nx.kamada_kawai_layout(G)

# --- Classify nodes and edges ---
active_nodes = {f"QB{i}" for i in active_qubits}
inactive_nodes = set(G.nodes()) - active_nodes

active_edges = []
active_edge_weights = []
inactive_edges = []

for u, v in G.edges():
    u_num = int(u[2:])
    v_num = int(v[2:])
    pair = (min(u_num, v_num), max(u_num, v_num))
    count = cz_counts.get(pair, 0)
    if count > 0:
        active_edges.append((u, v))
        active_edge_weights.append(count)
    else:
        inactive_edges.append((u, v))

print(f"Active edges found: {active_edges}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(18, 14))

nx.draw_networkx_nodes(G, pos, nodelist=list(inactive_nodes),
                       node_size=250, node_color="#cccccc", ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=inactive_edges,
                       edge_color="#dddddd", width=1.0, ax=ax)

if active_edge_weights:
    cmap = cm.YlOrRd
    norm = plt.Normalize(vmin=min(active_edge_weights), vmax=max(active_edge_weights))
    edge_colors = [cmap(norm(w)) for w in active_edge_weights]
    edge_widths = [3 + (w / max(active_edge_weights)) * 6 for w in active_edge_weights]
    nx.draw_networkx_edges(G, pos, edgelist=active_edges,
                           edge_color=edge_colors, width=edge_widths, ax=ax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="CZ Gate Count", shrink=0.55, pad=0.01)
    edge_labels = {(u, v): cz_counts[(min(int(u[2:]), int(v[2:])), max(int(u[2:]), int(v[2:])))]
                   for u, v in active_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, font_color="#8B0000", ax=ax)
else:
    ax.text(0.5, 0.5, "No active CZ edges found.\nRe-run hw_visualisation.py to regenerate qcprov.json with layout info.",
            ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')

nx.draw_networkx_nodes(G, pos, nodelist=list(active_nodes),
                       node_size=500, node_color="steelblue", ax=ax)
nx.draw_networkx_labels(G, pos,
                        labels={n: n[2:] for n in active_nodes},
                        font_size=7, font_color="white", font_weight="bold", ax=ax)
nx.draw_networkx_labels(G, pos,
                        labels={n: n[2:] for n in inactive_nodes},
                        font_size=6, font_color="#999999", ax=ax)

legend = [
    mpatches.Patch(color="steelblue", label=f"Active qubit ({len(active_nodes)})"),
    mpatches.Patch(color="#cccccc", label=f"Idle qubit ({len(inactive_nodes)})"),
    mpatches.Patch(color="#ff6600", label=f"Active CZ edges ({len(active_edges)}, "
                                          f"total {sum(active_edge_weights)} ops)"),
]
ax.legend(handles=legend, loc="lower left", fontsize=9)
ax.set_title(
    f"IQM Aphrodite — Compiled Circuit on Full Topology\n"
    f"Active qubits: {sorted(active_qubits)}  |  "
    f"CZ pairs: {len(active_edges)}  |  Total CZ ops: {sum(active_edge_weights)}",
    fontsize=13
)
ax.axis("off")
plt.tight_layout()
plt.savefig("circuit_topology.png", dpi=150, bbox_inches="tight")
print("Saved to circuit_topology.png")
plt.show()
