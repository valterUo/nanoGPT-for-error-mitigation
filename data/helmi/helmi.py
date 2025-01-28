import copy
import os
import pickle
import random
random.seed(0)
#import pickle
import numpy as np
import qiskit
from iqm import qiskit_iqm
from qiskit import QuantumCircuit
from concurrent.futures import ThreadPoolExecutor
from qiskit.primitives import BackendEstimatorV2 as Estimator
from qiskit.primitives import StatevectorEstimator
from qiskit.converters import circuit_to_dagdependency
from utils import rustworkx_to_networkx

from karateclub.graph_embedding import Graph2Vec

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
            for _ in range(random.randint(1, 40)):
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
    
    def store_results(self):
        
        nx_graphs = []
        
        for circ in self.circuits:
            dag = circuit_to_dagdependency(circ)
            rxgraph = dag.to_retworkx()
            nxgraph = rustworkx_to_networkx(rxgraph)
            nx_graphs.append(nxgraph)

        all_indices = len(nx_graphs)
        assert all_indices == len(self.circuits) == len(self.noisy_expectation_values) == len(self.ideal_expectation_values)

        val_indices = list(random.sample(range(all_indices), int(0.2*all_indices)))

        vectorizer = Graph2Vec(wl_iterations = 2, 
                       dimensions = 768,
                       workers = 8, 
                       down_sampling = 0.0001, 
                       epochs = 10, 
                       learning_rate = 0.025, 
                       min_count = 1, 
                       seed = 0, 
                       erase_base_features = False)

        vectorizer.fit(nx_graphs)    
        embeddings = vectorizer.get_embedding()

        val_context = [np.array(embeddings[i]) for i in val_indices]
        val_context = np.array(val_context)
        val_src = [self.noisy_expectation_values[i] for i in val_indices]
        val_tgt = [self.ideal_expectation_values[i] for i in val_indices]
        val_src = np.array(val_src)
        val_tgt = np.array(val_tgt)
        
        train_context = [np.array(embeddings[i]) for i in range(all_indices) if i not in val_indices]
        train_context = np.array(train_context)
        train_src = [self.noisy_expectation_values[i] for i in range(all_indices) if i not in val_indices]
        train_src = np.array(train_src)
        train_tgt = [self.ideal_expectation_values[i] for i in range(all_indices) if i not in val_indices]
        train_tgt = np.array(train_tgt)

        print(f"Validation context: {val_context.shape}")
        print(f"Validation source: {val_src.shape}")
        print(f"Validation target: {val_tgt.shape}")
        print(f"Training context: {train_context.shape}")
        print(f"Training source: {train_src.shape}")
        print(f"Training target: {train_tgt.shape}")

        with open(os.path.join(os.path.dirname(__file__), 'val_context.pkl'), 'wb') as f:
            pickle.dump(val_context, f)
        with open(os.path.join(os.path.dirname(__file__), 'val_src.pkl'), 'wb') as f:
            pickle.dump(val_src, f)
        with open(os.path.join(os.path.dirname(__file__), 'val_tgt.pkl'), 'wb') as f:
            pickle.dump(val_tgt, f)
        
        with open(os.path.join(os.path.dirname(__file__), 'train_context.pkl'), 'wb') as f:
            pickle.dump(train_context, f)
        with open(os.path.join(os.path.dirname(__file__), 'train_src.pkl'), 'wb') as f:
            pickle.dump(train_src, f)
        with open(os.path.join(os.path.dirname(__file__), 'train_tgt.pkl'), 'wb') as f:
            pickle.dump(train_tgt, f)

        #val_context.tofile(os.path.join(os.path.dirname(__file__), 'val_context.bin'))
        # Pickle the validation and training graphs
        #with open(os.path.join(os.path.dirname(__file__), 'val_context.pkl'), 'wb') as f:
        #    pickle.dump(val_context, f)
        #val_src.tofile(os.path.join(os.path.dirname(__file__), 'val_src.bin'))
        #val_tgt.tofile(os.path.join(os.path.dirname(__file__),'val_tgt.bin'))
        
        #train_context.tofile(os.path.join(os.path.dirname(__file__), 'train_context.bin'))
        #with open(os.path.join(os.path.dirname(__file__), 'train_context.pkl'), 'wb') as f:
        #    pickle.dump(train_context, f)
        #train_src.tofile(os.path.join(os.path.dirname(__file__), 'train_src.bin'))
        #train_tgt.tofile(os.path.join(os.path.dirname(__file__), 'train_tgt.bin'))

        print("Data stored successfully")
        # Example
        #print(f"Validation context: {val_context[0]}")
        #print(f"Validation source: {val_src[0]}")
        #print(f"Validation target: {val_tgt[0]}")
        #print(f"Training context: {train_context[0]}")
        #print(f"Training source: {train_src[0]}")
        #print(f"Training target: {train_tgt[0]}")