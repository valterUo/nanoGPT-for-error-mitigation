import random
random.seed(0)
import pickle
import numpy as np
import qiskit
from iqm import qiskit_iqm
from qiskit import QuantumCircuit
from concurrent.futures import ThreadPoolExecutor
from qiskit.primitives import BackendEstimatorV2 as Estimator
from qiskit.primitives import StatevectorEstimator
from qiskit.converters import circuit_to_dagdependency
from utils import rustworkx_to_networkx

from karateclub.graph_embedding import Graph2Vec # type: ignore

class Helmi:
    
    def __init__(self, num_circuits):
        self.backend = qiskit_iqm.IQMFakeAdonis()
        self.num_circuits = num_circuits
        self.operations = {"single": ["r"], "two": ["cz"]}
        self.coupling_map = list(self.backend.coupling_map)
        
        print(f'Native operations of the backend: {self.backend.operation_names}')
        print(f'Coupling map of the backend: {self.backend.coupling_map}')
        print(f'Qubit list of the backend: {self.backend.num_qubits}')
        
        self.circuits = self.generate_random_circuits()
        self.noisy_expectation_values = []
        self.ideal_expectation_values = []
    
    
    def generate_random_circuits(self):
        def create_circuit():
            qc = QuantumCircuit(self.backend.num_qubits)
            for q in range(self.backend.num_qubits):
                theta = random.uniform(-np.pi, np.pi)
                phi = random.uniform(-np.pi, np.pi)
                qc.r(theta, phi, q)
            for _ in range(random.randint(1, 15)):
                op = random.choice(list(self.operations.keys()))
                if op == "single":
                    q = random.choice(range(self.backend.num_qubits))
                    theta = random.uniform(-np.pi, np.pi)
                    phi = random.uniform(-np.pi, np.pi)
                    qc.r(theta, phi, q)
                elif op == "two":
                    q1, q2 = random.choice(self.coupling_map)
                    qc.cz(q1, q2)
            return qc

        with ThreadPoolExecutor() as executor:
            circuits = list(executor.map(lambda _: create_circuit(), range(self.num_circuits)))
        return circuits
    
    
    def run_circuits_on_Adonis(self):
        op = qiskit.quantum_info.SparsePauliOp.from_list([('Z'*self.backend.num_qubits, 1)])
        estimator = Estimator(backend=self.backend, options={'seed_simulator': 0})
        def run_circuit(qc):
            job = estimator.run([(qc, op)])
            pub_result = job.result()[0]
            return pub_result.data.evs

        with ThreadPoolExecutor() as executor:
            self.noisy_expectation_values = list(executor.map(run_circuit, self.circuits))
        return self.noisy_expectation_values
    
    
    def run_circuits_on_Aer(self):
        op = qiskit.quantum_info.SparsePauliOp.from_list([('Z'*self.backend.num_qubits, 1)])
        estimator = StatevectorEstimator(seed = 0)
        def run_circuit(qc):
            job = estimator.run([(qc, op)])
            pub_result = job.result()[0]
            return pub_result.data.evs

        with ThreadPoolExecutor() as executor:
            self.ideal_expectation_values = list(executor.map(run_circuit, self.circuits))
        return self.ideal_expectation_values
    
    def pickle_results(self):
        
        # The default values except attributed=True, min_count=1, seed = 0 and workers = 8
        vectorizer = Graph2Vec(wl_iterations = 2, 
                       attributed = True, 
                       dimensions = 128, 
                       workers = 8, 
                       down_sampling = 0.0001, 
                       epochs = 10, 
                       learning_rate = 0.025, 
                       min_count = 1, 
                       seed = 0, 
                       erase_base_features = False)
        results = []
        nx_graphs = []
        
        for circ in self.circuits:
            dag = circuit_to_dagdependency(circ)
            rxgraph = dag.to_retworkx()
            nxgraph = rustworkx_to_networkx(rxgraph)
            nx_graphs.append(nxgraph)
        
        vectorizer.fit(nx_graphs)    
        embeddings = vectorizer.get_embedding()
        
        for embedding, c, e_noisy, e_idea in zip(embeddings, self.circuits, self.noisy_expectation_values, self.ideal_expectation_values):
            results.append({"dag_embedding": embedding, 
                            "original_circuit": c, 
                            "noisy_expectation": e_noisy, 
                            "ideal_expectation": e_idea})
        with open("results2.pkl", "wb") as f:
            pickle.dump(results, f)
            