import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2

def create_spqc_circuit(t=0, m=2, n=2, r=1):
    """
    Create and return an SPQC circuit with specified parameters.
    
    Args:
        t: number of additional linear terms (can be 0)
        m: address register size  
        n: data register size
        r: number of data registers
        
    Returns:
        QuantumCircuit: The complete SPQC circuit
    """
    # --- hyperparameters and registers ---
    total_qubits = n*r+m+t+1    # total number of qubits (1 for helper ancilla control)
    L = 2**m                    # number of sub-models
    T = 2**t                    # total number of term addresses
    anc = 1                     # single ancilla qubit to help with multi-control
    params_per_model = 2        # e.g. one rotation per data qubit
    term_register = range(0,t)
    address_register = range(t,t+m)
    data_registers = [range(t+m+i*n,t+m+(i+1)*n) for i in range(r)]
    term_addresses = ['1' * i + '0' * (t - i) for i in range(t+1)]
    ancilla = [total_qubits - 1]  # ancilla index

    ## --- SPQC Circuit --- 
    # Feature map
    feature_map = QuantumCircuit(n)
    input_thetas = [Parameter(f"input_theta{i}") for i in range(2)]
    feature_map.ry(input_thetas[0],0)
    feature_map.ry(input_thetas[1],1)
    fm = feature_map.to_gate(label=f"S(X)")

    # Parameterised sub-models
    sub_models = []
    for i in range(L):
        thetas = [Parameter(f"model{i}_theta{j}") for j in range(4)]
        sub_model = QuantumCircuit(n)
        sub_model.ry(thetas[0],0)
        sub_model.rx(thetas[1],0)
        sub_model.ry(thetas[2],1)
        sub_model.rx(thetas[3],1)
        sub_models.append(sub_model.to_gate(label=f"model{i}"))

    # Create SPQC with classical bits for measurements, append Hadamards and feature maps
    num_classical_bits = n * r  # One classical bit per data qubit
    qc = QuantumCircuit(total_qubits, num_classical_bits)
    qc.h(address_register)
    if t > 0:
        qc.h(term_register)

    for i in range(len(data_registers)):
        qc.append(fm, data_registers[i])  # Feature maps

    # Helper function to flip bits
    def flip_bits(qc, register, address):
        for j, bit in enumerate(address):
            if bit == '0':
                qc.x(register[j])

    # Append sub-models
    for i in range(L):
        for p in range(T):
            if t > 0:
                t_address = format(p, f'0{t}b')
            else:
                t_address = ""
            m_address = format(i, f'0{m}b')

            # Apply sub-models for "correct" term addresses
            if t_address in term_addresses:
                # Flip 0s to 1s
                flip_bits(qc, address_register, m_address)
                flip_bits(qc, term_register, t_address)

                # Apply sub-models conditionally
                for k in range(len(data_registers)):
                    data_register = data_registers[k]
                    controlled_sub_model = sub_models[i].control(num_ctrl_qubits=1)
                    control_qubits = list(term_register) + list(address_register)
                    qc.mcx(control_qubits, ancilla) # Use helper ancilla to control sub-model
                    target_qubits = list(data_register)
                    qc.append(controlled_sub_model, ancilla + target_qubits)
                    qc.mcx(control_qubits, ancilla) # Reset ancilla

                # Flip 1s back to 0s
                flip_bits(qc, address_register, m_address)
                flip_bits(qc, term_register, t_address)

            # Create Phi state for "incorrect" term addresses
            else:
                # Inverse hadamards
                qc.h(address_register)
                if t > 0:
                    qc.h(term_register)
                
                # Inverse feature maps
                inverse_fm = QuantumCircuit(n)
                inverse_fm.ry(-input_thetas[0],0)
                inverse_fm.ry(-input_thetas[1],1)
                inv_fm = inverse_fm.to_gate()
                for j in range(len(data_registers)):
                    qc.append(inv_fm, data_registers[j])

    # Measure data registers
    classical_bit_index = 0
    for i, data_register in enumerate(data_registers):
        for qubit in data_register:
            qc.measure(qubit, classical_bit_index)
            classical_bit_index += 1

    # Create address register ansatz
    address_ansatz = EfficientSU2(m, reps=1, parameter_prefix='address_theta')
    qc.append(address_ansatz, address_register)

    # Measure address register
    
    return qc

def get_parameter_mapping(circuit):
    """
    Create a mapping from parameter names to Parameter objects.
    This helps organize parameters by type (input vs model weights).
    """
    input_params = []
    model_params = []
    address_params = []
    
    for param in circuit.parameters:
        if param.name.startswith('input_theta'):
            input_params.append(param)
        elif param.name.startswith('model'):
            model_params.append(param)
        elif param.name.startswith('address_theta'):
            address_params.append(param)
    
    return input_params, model_params, address_params

def create_random_weights(circuit, seed=None):
    """
    Create random weights for the SPQC with disjoint weight sets for each model.
    
    Args:
        circuit: The SPQC circuit
        seed: Random seed for reproducibility (optional)
        
    Returns:
        numpy.ndarray: Combined array of random weights (model + address)
    """
    if seed is not None:
        np.random.seed(seed)
    
    input_params, model_params, address_params = get_parameter_mapping(circuit)
    
    # Group model parameters by model number
    model_groups = {}
    for param in model_params:
        # Extract model number from parameter name like "model0_theta1"
        model_num = int(param.name.split('_')[0].replace('model', ''))
        if model_num not in model_groups:
            model_groups[model_num] = []
        model_groups[model_num].append(param)
    
    # Create disjoint weight ranges for each model
    num_models = len(model_groups)
    params_per_model = len(model_groups[0]) if model_groups else 0
        
    # Define disjoint ranges for each model
    weight_ranges = []
    range_size = 2 * np.pi / num_models  # Divide [-π, π] into disjoint segments
    
    for i in range(num_models):
        start = -np.pi + i * range_size
        end = -np.pi + (i + 1) * range_size
        weight_ranges.append((start, end))
    
    # Generate random weights with disjoint ranges for models
    model_weights = []
    
    for model_num in sorted(model_groups.keys()):
        start, end = weight_ranges[model_num]
        weights = np.random.uniform(start, end, params_per_model)
        model_weights.extend(weights)
    
    # Generate random weights for address ansatz (separate range)
    address_weights = np.random.uniform(-np.pi, np.pi, len(address_params))
    
    # Combine model and address weights
    all_weights = np.concatenate([model_weights, address_weights])
    
    return all_weights

def bind_params(circuit, input_values, random_weights):
    """
    Bind parameters to the circuit.
    
    Args:
        circuit: The SPQC circuit
        input_values: List of input feature values [x, y]
        random_weights: Combined array of model and address weights
    """
    input_params, model_params, address_params = get_parameter_mapping(circuit)
    param_binding = {}

    # Bind input parameters
    for i, param in enumerate(input_params):
        if i < len(input_values):
            param_binding[param] = input_values[i]
        else:
            param_binding[param] = 0.0
    
    # Split random_weights into model and address portions
    num_model_params = len(model_params)
    model_values = random_weights[:num_model_params]
    address_values = random_weights[num_model_params:]
    
    # Bind model parameters  
    for i, param in enumerate(model_params):
        if i < len(model_values):
            param_binding[param] = model_values[i]
        else:
            param_binding[param] = 0.0

    # Bind address parameters
    for i, param in enumerate(address_params):
        if i < len(address_values):
            param_binding[param] = address_values[i]
        else:
            param_binding[param] = 0.0

    return circuit.assign_parameters(param_binding)

def visualize_circuit(qc):
    print(f"Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
    qc.draw(output='mpl', fold=40)
    plt.show()

if __name__ == "__main__":
    qc = create_spqc_circuit(t=0, m=2, n=2, r=1)
    visualize_circuit(qc)