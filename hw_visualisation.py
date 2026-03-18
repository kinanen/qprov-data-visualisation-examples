import math
import mlflow
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, qasm2, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite
#from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
import matplotlib.pyplot as plt
import numpy as np
import time
#
from io import BytesIO, StringIO
import json, os


# Create quantum and classical registers
qubits = 15
qr = QuantumRegister(qubits, 'q')
cr = ClassicalRegister(qubits, 'c')
qc = QuantumCircuit(qr, cr)
shots = 2000

optimization_level = 2

# --- Parameters ---
n_qubits = 15

def build_quantum_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)

    # --- State preparation ---
    qc.h(range(n_qubits))
    for i in range(n_qubits):
        qc.rz(math.pi / (i + 2), i)
    for i in range(0, n_qubits, 3):
        qc.x(i)

    # --- Layer 1: forward CX ladder ---
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # --- Single-qubit rotations after layer 1 ---
    for i in range(n_qubits):
        qc.ry(math.pi / (i + 3), i)

    # --- Layer 2: long-range CX (skip-1 connections) ---
    for i in range(0, n_qubits - 2, 2):
        qc.cx(i, i + 2)

    # --- Single-qubit rotations after layer 2 ---
    for i in range(n_qubits):
        qc.rz(math.pi / (i + 4), i)

    # --- Layer 3: CZ-style entanglement on odd pairs ---
    for i in range(1, n_qubits - 1, 2):
        qc.cz(i, i + 1)

    # --- Single-qubit rotations after layer 3 ---
    for i in range(n_qubits):
        qc.rx(math.pi / (i + 5), i)

    # --- Layer 4: reverse CX ladder ---
    for i in range(n_qubits - 2, -1, -1):
        qc.cx(i + 1, i)

    # --- Final single-qubit layer ---
    for i in range(n_qubits):
        qc.h(i)
        qc.rz(math.pi / (i + 2), i)

    return qc

qc = build_quantum_circuit(n_qubits)

# Measure all qubits
qc.measure(range(n_qubits), range(n_qubits))

# Instantiate the fake IQM backend
# backend = IQMFakeApollo()
backend = IQMFakeAphrodite()
backend_name = getattr(backend, 'name', None) or str(backend)
backend_architecture = getattr(backend, 'architecture', None) or "No architecture info available"
backend_error_profile = getattr(backend, 'error_profile', None) or "No error profile available"

# Transpile for the backend
transpile_start = time.perf_counter()
qc_iqm = transpile(qc, backend=backend, optimization_level=optimization_level)
transpile_time_seconds = time.perf_counter() - transpile_start
print("transpile time: ", transpile_time_seconds)

# Extract physical qubit layout: register+index -> physical QB number (1-based)
qubit_layout = {}
try:
    layout = qc_iqm.layout
    if layout and layout.initial_layout:
        for virt, phys in layout.initial_layout.get_virtual_bits().items():
            reg = virt._register.name if virt._register else "q"
            idx = virt._index
            qubit_layout[f"{reg}[{idx}]"] = phys + 1  # 1-based QB number
except Exception:
    pass


# Run on the IQM fake backend
run_start = time.perf_counter()
job = backend.run(qc_iqm, shots=shots)
run_time_seconds = time.perf_counter() - run_start
result = job.result()
counts = result.get_counts()
probabilities = {state: count/shots for state, count in counts.items()}
print("run time: ", run_time_seconds)

#Run without fake backend to get ideal counts for comparison
ideal_backend = AerSimulator(method='statevector')
ideal_job = ideal_backend.run(qc_iqm, shots=shots)
ideal_result = ideal_job.result()
ideal_counts = ideal_result.get_counts()


# most_likely = max(counts, key=counts.get)
# print("Most likely measured bitstring:")
# print(most_likely)
# print(f"Probability ≈ {counts[most_likely] / shots:.3f}")

# most_likely_ideal = max(ideal_counts, key=ideal_counts.get)
# print("\nMost likely ideal bitstring:")
# print(most_likely_ideal)
# print(f"Ideal Probability ≈ {ideal_counts[most_likely_ideal] / shots:.3f}")

# # Get top 10 most likely bitstrings
# top_10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
# top_10_str = "\n".join([f"{i+1}. {bitstring}: {count} counts ({count/shots:.4f} probability)" 
#                             for i, (bitstring, count) in enumerate(top_10)])
# print("\nTop 10 most likely bitstrings:")
# print(top_10_str)

# plot_histogram(
#     counts,
#     title="Measurement Results",
#     figsize=(10, 4),
#     bar_labels=False
# )
# plt.tight_layout()
# plt.show()

# Export to QASM 2.0
try:
    qasm_string = qasm2.dumps(qc)
except Exception:
    qasm_string = None
try:
    qasm_string_iqm = qasm2.dumps(qc_iqm)
except Exception:
    qasm_string_iqm = None

# store number of qubits and connectivity info
num_qubits = getattr(qc, 'num_qubits', None) or (qc.num_qubits if hasattr(qc, 'num_qubits') else None)

qubit_connectivity = None
if backend_architecture and hasattr(backend_architecture, 'gates') and 'cz' in backend_architecture.gates:
    try:
        qubit_connectivity = list(
            backend_architecture.gates['cz']
            .implementations[backend_architecture.gates['cz'].default_implementation]
            .loci
        )
    except Exception:
        qubit_connectivity = None

# store gate set, gate times and readout fidelities
gate_set = list(backend_architecture.gates.keys()) if backend_architecture and hasattr(backend_architecture, 'gates') else None

gate_times = None
if backend_error_profile and hasattr(backend_error_profile, 'single_qubit_gate_durations') and hasattr(backend_error_profile, 'two_qubit_gate_durations'):
    try:
        gate_times = {
            **backend_error_profile.single_qubit_gate_durations,
            **backend_error_profile.two_qubit_gate_durations,
        }
    except Exception:
        gate_times = None

readout_fidelities = None
if backend_error_profile and hasattr(backend_error_profile, 'readout_errors'):
    try:
        readout_fidelities = {
            qb: 1.0 - (errs.get('0', 0) + errs.get('1', 0)) / 2.0
            for qb, errs in backend_error_profile.readout_errors.items()
        }
    except Exception:
        readout_fidelities = None


quantum_computer = {
        "backend_name": backend_name,
        "number_of_qubits": int(backend.num_qubits) if hasattr(backend, 'num_qubits') else None,
        "t1_times": getattr(backend_error_profile, 't1s', None) if backend_error_profile else None,
        "t2_times": getattr(backend_error_profile, 't2s', None) if backend_error_profile else None,
        "qubit_connectivity": qubit_connectivity,
        "gate_set": gate_set,
        "single_qubit_gate_error": str(getattr(backend_error_profile, 'single_qubit_gate_depolarizing_error_parameters', None)) if backend_error_profile else None,
        "two_qubit_gate_error": str(getattr(backend_error_profile, 'two_qubit_gate_depolarizing_error_parameters', None)) if backend_error_profile else None,
        "gate_times": gate_times,
        "readout_fidelities": readout_fidelities,
        "original_qasm": qasm_string,
        "compiled_circuit_qasm": qasm_string_iqm,
        "qubit_layout": qubit_layout,
        }

qcprov_path = os.path.join(os.getcwd(), 'qcprov.json')
with open(qcprov_path, 'w') as tmp:
    json.dump(quantum_computer, tmp, default=str)
print(f"quantum_computer written to: {qcprov_path}")




# def log_quantum_run(qc, compiled_qc, result, backend, shots=None, optimization_level=None, transpile_time_seconds=None, run_time_seconds=None):
#     """Log a quantum run to MLflow.

#     Parameters:
#     - qc: original QuantumCircuit
#     - compiled_qc: transpiled/compiled QuantumCircuit (iqm-optimized)
#     - result: execution result object (must provide `get_counts()`)
#     - backend: backend object (may provide architecture/error_profile)
#     - shots: optional number of shots (if not provided, inferred from counts)
#     - optimization_level, transpile_time_seconds, run_time_seconds: optional metadata
#     """
#     mlflow.set_experiment("QProv - Q50 Circuit on IQM Fake Backend")  

#     qc_iqm = compiled_qc
#     backend_name = getattr(backend, 'name', None) or str(backend)
#     backend_architecture = getattr(backend, 'architecture', None) or None
#     backend_error_profile = getattr(backend, 'error_profile', None) or None
#     # If an MLflow run is already active, start a nested run to avoid Exception
  
#     with mlflow.start_run(run_name='qprov logging'):

#         # Get counts and shots
#         counts = result.get_counts() if hasattr(result, 'get_counts') else {}
#         if shots is None:
#             shots = int(sum(counts.values())) if counts else None

#         probabilities = {state: count / shots for state, count in counts.items()} if shots else {}

#         most_likely = max(counts, key=counts.get) if counts else None


#         # Export to QASM 2.0
#         try:
#             qasm_string = qasm2.dumps(qc)
#         except Exception:
#             qasm_string = None
#         try:
#             qasm_string_iqm = qasm2.dumps(qc_iqm)
#         except Exception:
#             qasm_string_iqm = None

#         # store number of qubits and connectivity info
#         num_qubits = getattr(qc, 'num_qubits', None) or (qc.num_qubits if hasattr(qc, 'num_qubits') else None)

#         qubit_connectivity = None
#         if backend_architecture and hasattr(backend_architecture, 'gates') and 'cz' in backend_architecture.gates:
#             try:
#                 qubit_connectivity = list(
#                     backend_architecture.gates['cz']
#                     .implementations[backend_architecture.gates['cz'].default_implementation]
#                     .loci
#                 )
#             except Exception:
#                 qubit_connectivity = None

#         # store gate set, gate times and readout fidelities
#         gate_set = list(backend_architecture.gates.keys()) if backend_architecture and hasattr(backend_architecture, 'gates') else None

#         gate_times = None
#         if backend_error_profile and hasattr(backend_error_profile, 'single_qubit_gate_durations') and hasattr(backend_error_profile, 'two_qubit_gate_durations'):
#             try:
#                 gate_times = {
#                     **backend_error_profile.single_qubit_gate_durations,
#                     **backend_error_profile.two_qubit_gate_durations,
#                 }
#             except Exception:
#                 gate_times = None

#         readout_fidelities = None
#         if backend_error_profile and hasattr(backend_error_profile, 'readout_errors'):
#             try:
#                 readout_fidelities = {
#                     qb: 1.0 - (errs.get('0', 0) + errs.get('1', 0)) / 2.0
#                     for qb, errs in backend_error_profile.readout_errors.items()
#                 }
#             except Exception:
#                 readout_fidelities = None

#         # Log all provenance data in structured format to be stored as JSON artifacts in MLFlow
#         quantum_circuit = {
#             "qubits": int(num_qubits) if num_qubits is not None else None,
#             "shots": int(shots) if shots is not None else None,
#             "circuit depth": int(qc.depth()) if hasattr(qc, 'depth') and qc.depth() is not None else None,
#             "gates": dict(qc.count_ops()) if hasattr(qc, 'count_ops') and qc.count_ops() is not None else None,
#             "number of gates": int(len(qc.data)) if hasattr(qc, 'data') and qc.data is not None else None,
#             "circuit transpiled depth": int(qc_iqm.depth()) if hasattr(qc_iqm, 'depth') and qc_iqm.depth() is not None else None,
#             "transpiled gates": dict(qc_iqm.count_ops()) if hasattr(qc_iqm, 'count_ops') and qc_iqm.count_ops() is not None else None,
#             "transpiled number of gates": int(len(qc_iqm.data)) if hasattr(qc_iqm, 'data') and qc_iqm.data is not None else None,
#             "transpile_time_seconds": float(transpile_time_seconds) if transpile_time_seconds is not None else None,
#         }

#         quantum_computer = {
#             "backend_name": backend_name,
#             "number_of_qubits": int(backend.num_qubits) if hasattr(backend, 'num_qubits') else None,
#             "t1_times": getattr(backend_error_profile, 't1s', None) if backend_error_profile else None,
#             "t2_times": getattr(backend_error_profile, 't2s', None) if backend_error_profile else None,
#             "qubit_connectivity": qubit_connectivity,
#             "gate_set": gate_set,
#             "single_qubit_gate_error": str(getattr(backend_error_profile, 'single_qubit_gate_depolarizing_error_parameters', None)) if backend_error_profile else None,
#             "two_qubit_gate_error": str(getattr(backend_error_profile, 'two_qubit_gate_depolarizing_error_parameters', None)) if backend_error_profile else None,
#             "gate_times": gate_times,
#             "readout_fidelities": readout_fidelities,
#             "original_qasm": qasm_string,
#         }

#         compilation = {
#             "transpiled_circuit_qasm": qasm_string_iqm,
#             "optimization_level": optimization_level,
#             "transpile_time_seconds": float(transpile_time_seconds) if transpile_time_seconds is not None else None,
#         }

#         execution = {
#             "input_data": None,
#             "output_data": {
#                 "counts": counts,
#                 "probabilities": probabilities,
#             },
#             "shot": shots,
#             "random_seed": getattr(backend, 'random_seed', None),
#             "execution_time_seconds": float(run_time_seconds) if run_time_seconds is not None else None,
#         }

#         # Log the provenance data to MLFlow
#         try:
#             mlflow.log_dict(quantum_circuit, "quantum_circuit_provenance_data.json")
#             mlflow.log_dict(quantum_computer, "quantum_computer_provenance_data.json")
#             mlflow.log_dict(compilation, "compilation_provenance_data.json")
#             mlflow.log_dict(execution, "execution_provenance_data.json")
#         except Exception:
#             pass

#         # Log the QASM strings and circuit diagrams as artifacts
#         try:
#             if qasm_string:
#                 mlflow.log_text(qasm_string, artifact_file="original_circuit_qasm.txt")
#             if qasm_string_iqm:
#                 mlflow.log_text(qasm_string_iqm, artifact_file="iqm_optimized_circuit_qasm.txt")
#         except Exception:
#             pass

#     mlflow.end_run()

# # Log everything to MLflow
# log_quantum_run(
#     qc=qc,
#     compiled_qc=qc_iqm,
#     result=result,
#     backend=backend,
#     shots=shots,
#     optimization_level=optimization_level,
#     transpile_time_seconds=transpile_time_seconds,
#     run_time_seconds=run_time_seconds
# )






