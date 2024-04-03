from qiskit_nature.second_q.operators import FermionicOp
from mindquantum.core.operators import FermionOperator,QubitOperator
from mindquantum.core.parameterresolver import ParameterResolver
from qiskit.quantum_info import SparsePauliOp
from mindquantum.algorithm.nisq import Transform
from qiskit.quantum_info.operators import SparsePauliOp
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.drivers import PySCFDriver
import re
import numpy as np
from typing import List

def qiskit_operator_converter(qiskit_operator:FermionicOp):      
    """
    把Qiskit的UCC 下的FermonicOp 转化成 QEB的operator 
    也就是翻译成Mindquantum后 把Z去掉
    """  
    if len(qiskit_operator.items())==1:
        #只含单个项的operator 如FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2)
        operator_str,coeffient = list(qiskit_operator.items())[0]
        return(FermionOperator(terms=transform_string(operator_str),coefficient=coeffient))
    if len(qiskit_operator.items())==2:
        hw_op=[]
        for i in list(qiskit_operator.items()):
            operator_str,coeffient = i
            hw_op.append(FermionOperator(terms=transform_string(operator_str),coefficient=coeffient))
        return(sum(hw_op))
    


def transform_string(expression):
    elements = expression.split(' ')
    result = []
    for element in elements:
        if element[0] == '+':
            result.append(element[2:] + '^')
        elif element[0] == '-':
            result.append(element[2:])
    return ' '.join(result)

def convert_pauli_string(pauli_string):
    converted_string = ''
    num_qubits = len(pauli_string)

    for i in range(num_qubits):
        if pauli_string[i] == 'X':
            converted_string += f'X{num_qubits - 1 - i} '
        elif pauli_string[i] == 'Y':
            converted_string += f'Y{num_qubits - 1 - i} '
        elif pauli_string[i] == 'Z':
            converted_string += f'Z{num_qubits - 1 - i} '

    return converted_string.strip()

#本函数将qiskit的operator转化成华为框架下的FermionOperator
#针对单个Operator 如FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2)
def qiskit_operator_converter(qiskit_operator:FermionicOp):        
    if len(qiskit_operator.items())==1:
        #只含单个项的operator 如FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2)
        operator_str,coeffient = list(qiskit_operator.items())[0]
        return(FermionOperator(terms=transform_string(operator_str),coefficient=coeffient))
    if len(qiskit_operator.items())==2:
        hw_op=[]
        for i in list(qiskit_operator.items()):
            operator_str,coeffient = i
            hw_op.append(FermionOperator(terms=transform_string(operator_str),coefficient=coeffient))
        return(sum(hw_op))
    
def qiskit_parameterized_operator_convereter(qiskit_operator:FermionicOp,params_name:str=None):
    if len(qiskit_operator.items())==1:
        #只含单个项的operator 如FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2)
        operator_str,coeffient = list(qiskit_operator.items())[0]
        co = np.real(coeffient*1j)
        params = ParameterResolver(data={params_name:co})
        #print(FermionOperator(terms=transform_string(operator_str),coefficient=params))
        return(FermionOperator(terms=transform_string(operator_str),coefficient=params))
    
    if len(qiskit_operator.items())==2:
        hw_op=[]
        for i in list(qiskit_operator.items()):
            operator_str,coeffient = i
            co = np.real(coeffient*1j)
            #print(coeffient*1j)
            params = ParameterResolver(data={params_name:co})
            hw_op.append(FermionOperator(terms=transform_string(operator_str),coefficient=params))
        #print(sum(hw_op))
        return(sum(hw_op))
    
    
def convert_hamiltonian(qiskit_hamiltonian:FermionicOp):
        """
        qiskit_hamiltonian: FermionicOp (Before mapping)
        -------------------------
        This function aim to convert the qiskit hamiltonian to mindquantum hamiltonian
        return type is QubitOperator of Hamiltonian
        
        """
        fermonic_op=[]
        for operator_str,coeffient in qiskit_hamiltonian.items():
            
            fermonic_op.append(FermionOperator(terms=transform_string(operator_str),coefficient=coeffient))
        fermonic_op = sum(fermonic_op)
        return(Transform(fermonic_op).jordan_wigner())
    
def convert_hamiltonian_PauliOP(qiskit_hamiltonian:SparsePauliOp):
    

    mindquantum_hamiltonian=[]
    for pauli_string,coeffient in qiskit_hamiltonian.to_list():
        pauli_str = convert_pauli_string(pauli_string=pauli_string)
        mindquantum_hamiltonian.append(QubitOperator(terms=pauli_str,coefficient=coeffient))
    
    return(sum(mindquantum_hamiltonian))
                
           
    
    
    
def convert_commutors(qiskit_commutators:FermionicOp):
        """
        qiskit_commutorrs: List[SparseOperator]
        -------------------------
        This function aim to convert the qiskit hamiltonian to mindquantum hamiltonian
        return type is QubitOperator of Hamiltonian
        
        """
        #list(fermonicadapt.commutors[0])[0].paulis.to_labels()
        
        commutator_qubit_op=[]
        for commutator in qiskit_commutators:
            qop=[]
            for pauli_string,coeffient in commutator.to_list():
                pauli_str = convert_pauli_string(pauli_string=pauli_string)
                qop.append(QubitOperator(terms=pauli_str,coefficient=coeffient))
            qop=sum(qop)
            commutator_qubit_op.append(qop)
        
        return(sum(commutator_qubit_op))


#统计MindQuantum线路中的CX门数量
def statistics_cx_num(circuit):
    cx_num = 0
    for i in circuit:
        if i.name=='CXGate':
            cx_num+=1
    return(cx_num)


def qiskit_operator_converter_QEB(qiskit_fermonic_op:List[SparsePauliOp]):
    """
    输入是qiskit的UCC FermionicOp 把Z去掉后 生成Mindquantum的QubitOperator
    """
    def convert_pauli_string(pauli_string):
        """
        特制 忽略Z 的版本
        """
        converted_string = ''
        num_qubits = len(pauli_string)

        for i in range(num_qubits):
            if pauli_string[i] == 'X':
                converted_string += f'X{num_qubits - 1 - i} '
            elif pauli_string[i] == 'Y':
                converted_string += f'Y{num_qubits - 1 - i} '
                
        return converted_string.strip()
    QEB_QubitOp=[]
    QEB_QubitOp_params=[]
    for index,i in enumerate(qiskit_fermonic_op):
        
        qop=[]
        qop_params=[]
        for pauli_string,coeffient in i.to_list():
            pauli_str = convert_pauli_string(pauli_string=pauli_string)
            qop.append(QubitOperator(terms=pauli_str,coefficient=coeffient))
            qop_params.append(QubitOperator(terms=pauli_str,coefficient=ParameterResolver(data={'QEB_'+str(index):np.real(coeffient)})))
            #print(f'paulli_str={pauli_str},qop={qop}')
        qop=sum(qop)
        qop_params=sum(qop_params)
        QEB_QubitOp.append(qop)
        QEB_QubitOp_params.append(qop_params)
        
    return QEB_QubitOp,QEB_QubitOp_params
    

def compute_eigenvalue(hamiltonian:QubitOperator,ansatz:Circuit,x0):

    
    simulator = Simulator('mqvector',Hamiltonian(hamiltonian).n_qubits)
    # simulator.apply_circuit(ansatz)
    molecule_pqc = simulator.get_expectation_with_grad(Hamiltonian(hamiltonian), ansatz)
    
    n_params = len(ansatz.params_name)
    if x0 is None:
        p0 = np.zeros(n_params) #参数初始化为0
    else:
        p0 = x0
    #mean_engergy, gradient = molecule_pqc(p0)
    # print(f'接收到的的x0为:{x0},N = {Hamiltonian(hamiltonian).n_qubits}')
    def fun(p0, molecule_pqc, energy_list=None):
        f, g = molecule_pqc(p0)
        f = np.real(f)[0, 0]
        g = np.real(g)[0, 0]
        if energy_list is not None:
            energy_list.append(f)
            # if len(energy_list) % 5 == 0:
            #     print(f"Step: {len(energy_list)},\tenergy: {f}")
        return f, g
    
    energy_list = []
    res = minimize(fun = fun, x0=p0, args=(molecule_pqc, energy_list), method='bfgs', jac=True)
    return res

from mindquantum.io import OpenQASM
def count_cnot(mindquantum_ansatz:Circuit,params:dict):
    """
    统计Mindquantum的ansatz中的CNOT门数量
    mindquantum_ansatz: Circuit 未绑定参数的量子线路
    params: dict 需要绑定的参数
    -------------------------
    """
    bound_circuit = mindquantum_ansatz.apply_value(dict(zip(mindquantum_ansatz.params_name, params)))
    openqasm = OpenQASM()
    openqasm_circuit_string = openqasm.to_string(bound_circuit)
    num  = openqasm_circuit_string.count('cx')
    return num


def Random_Gate_Acativation(ansatz:Circuit):
    num_parameters = ansatz.num_parameters
    pick_index = np.random.choice(a=range(num_parameters),size=(1,num_parameters),replace=False)
    pass


from qiskit.quantum_info import Pauli

def ParticialHamiltonian(hamiltonian: SparsePauliOp,remove_rate:float=0.2):    
    '''
    本函数用于实现Partcial Hamilton approach,筛选规则主要有两条: 
    1.根据系数的绝对值排序
    2.是否只含有I、Z(与论文PHA相关描述一致)
    '''
    num_elements_to_remove = int(len(hamiltonian.coeffs) * remove_rate)
    # 排序列表，根据第二个元素的绝对值
    sorted_data = sorted(hamiltonian.to_list(), key=lambda x: abs(x[1]))
    remaining_data = sorted_data[num_elements_to_remove:]
    remove_data = sorted_data[:num_elements_to_remove]
    
    s=[]
    for term in remaining_data:
        pauli_string, coefficient = term
        s.append(SparsePauliOp(data=Pauli(pauli_string),coeffs=coefficient))
        
    for term in remove_data:
        pauli_string, coefficient = term
        if pauli_string.find('X')==-1 and pauli_string.find('Y')==1:
            s.append(SparsePauliOp(data=Pauli(pauli_string),coeffs=coefficient))
            
    return sum(s)


def count_gate(ansatz:Circuit):
    """
    本函数主要为了统计ansatz里的cx门的数量 因此对于含参的ansatz 随便填充参数
    用以生成qasm编码来获取门的数量
    """
    if ansatz.parameterized == False:
        cir_str = OpenQASM().to_string(ansatz)
        cnot_num = cir_str.count('cx')
        return cnot_num
    
    else:
        ansatz_tmp = ansatz.apply_value(dict(zip(ansatz.params_name,[0]*len(ansatz.params_name))))
        cir_str = OpenQASM().to_string(ansatz_tmp)
        cnot_num = cir_str.count('cx')
        return cnot_num
        

        
    
        