# from openfermion.chem import MolecularData
# from openfermionpyscf import run_pyscf
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit,change_param_name
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import Transform
from mindquantum.algorithm.nisq import get_qubit_hamiltonian
from mindquantum.core.operators import QubitOperator,Hamiltonian
from mindquantum.core.parameterresolver import ParameterResolver
import numpy as np                                          
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ   # 导入量子门H, X, Y, Z, RX, RY, RZ
from mindquantum.core.circuit import Circuit,change_param_name
from mindquantum.simulator import Simulator
import logging,datetime,copy,pickle
from typing import List
from tool import compute_eigenvalue
from mindquantum.core.parameterresolver import PRGenerator


class Sequential_AdaptVQE():
    def __init__(self,hamiltonian: QubitOperator,Layer:int=2)->None:
        self.hamiltonian = hamiltonian
        self.n_qubits = Hamiltonian(hamiltonian=self.hamiltonian).n_qubits
        self.n_layers = Layer
        self.simulator = Simulator('mqvector', self.n_qubits)
        self.EHA = EHA(n_qubit=self.n_qubits,Layer=2)
        self.simulator = Simulator('mqvector',self.n_qubits)
        self.current_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        self.parameter_geneartor = PRGenerator()
        self.logger_init()
        
    def logger_init(self, logger_name:str=None):
            if logger_name is None:
                logger_name = 'Sequential_AdaptVQE' + self.current_time
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.DEBUG)
            myfilter = logging.Filter(logger_name)
            filehandler = logging.FileHandler(filename=logger_name, mode='w')
            filehandler.addFilter(myfilter)
            concolehander = logging.StreamHandler()
            concolehander.setLevel(logging.INFO)
            # 记录器绑定handerler
            self.logger.handlers.clear()
            self.logger.addHandler(filehandler)
            self.logger.addHandler(concolehander)
            self.logger.info('logger初始化完毕!')

    
    def start(self):
        self.logger.info('初始化无报错，开始进行......')
        self.current_ansatz = Circuit()
        self.optimal_parameters=[]
        for each_qubit in range(self.n_qubits):
            self.logger.info(f'当前处理第{each_qubit}个qubit')
            ansatz = copy.deepcopy(self.current_ansatz)
            self.logger.info(ansatz)
            # self.logger.info(f'当前处理的Ansatz为：\n{ansatz}')
            self.three_result = []
            index = 1
            for each_gate in [RX,RY,RZ]:
                ansatz+= each_gate('p_temp').on(each_qubit)
                if len(self.optimal_parameters)==0:
                    result = compute_eigenvalue(hamiltonian=self.hamiltonian,ansatz=ansatz,x0=np.random.uniform(-np.pi, np.pi))
                else:
                    temp_paramters = copy.deepcopy(self.optimal_parameters)
                    temp_paramters = np.append(temp_paramters,np.random.uniform(-np.pi, np.pi))
                    self.logger.info(f'当前载入的热启动参数为：{temp_paramters}')
                    result = compute_eigenvalue(hamiltonian=self.hamiltonian,ansatz=ansatz,x0=temp_paramters)
                self.three_result.append(result)
                self.logger.info('第{}/3次计算结果为：{}'.format(index,result.x))
                index+=1
            max_index = np.argmin([abs(i.fun) for i in self.three_result])
            self.logger.info(f'最优的index为：{max_index},因此选定的gate为：{["RX","RY","RZ"][max_index]}')
            self.logger.info(f'---------------------------')
            self.current_ansatz+= [RX,RY,RZ][max_index](self.parameter_geneartor.new()).on(each_qubit)
            self.optimal_parameters=self.three_result[max_index].x
        self.logger.info(self.current_ansatz)
        self.logger.info('-----------第1轮完毕，开始添加EHA块-------------')
        self.current_ansatz += self.EHA.EHA_ansatz
        self.logger.info(self.current_ansatz)
            
            
            




class EHA():
    """
    Entanglement-variational Hardware-efficient Ansatz (EHA)
    Args:
        n_qubit: int, the number of qubits
        Layer: int, the number of layers, repeat times
    
    """
    def __init__(self,n_qubit:int,Layer:int) -> None:
        self.EHA_ansatz = Circuit()
        self.n_qubit = n_qubit
        self.Layer = Layer
        self.simulator = Simulator('mqvector',self.n_qubit)
        for i in range(Layer):
            rotation_block =  self.rotation_part()
            rename_rb = change_param_name(circuit_fn=rotation_block,name_map=dict(zip(rotation_block.params_name,['Layer'+str(i)+'_theta'+str(j) for j in range(len(rotation_block.params_name))])))
            self.EHA_ansatz+=rename_rb
            
            for j in range(self.n_qubit)[:self.n_qubit-1]:
                tmp= self.Entangle_part(former_qubit=j,latter_qubit=j+1)
                entangle_block = change_param_name(circuit_fn=tmp,name_map=dict(zip(tmp.params_name,['Layer'+str(i)+'_'+str(j)+'_Beta'+str(k) for k in range(3)])))         
                self.EHA_ansatz+=entangle_block        
        
        
        
    def rotation_part(self):
        rotation_part = Circuit()
        for index,i in enumerate(range(self.n_qubit)):
            rotation_part +=RX(f'theta_{index}').on(i)
        return rotation_part
            
        
    def Entangle_part(self,former_qubit:int,latter_qubit:int):
        Entangle_part = Circuit()
        #XX
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        Entangle_part += RX('Beta_1').on(obj_qubits=former_qubit)
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        #YY
        Entangle_part += RX(np.pi/2).on(obj_qubits=former_qubit)
        Entangle_part += RX(np.pi/2).on(obj_qubits=latter_qubit)
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        Entangle_part += RX('Beta_2').on(obj_qubits=latter_qubit)
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        Entangle_part += RX(-1*np.pi/2).on(obj_qubits=former_qubit)
        Entangle_part += RX(-1*np.pi/2).on(obj_qubits=latter_qubit)
        #ZZ
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        Entangle_part += RX('Beta_3').on(obj_qubits=latter_qubit)
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        return Entangle_part
        
        
        
class AdaptResult(object):
    def __init__(self,vqe_result_history:List,
                    gradients_history,picked_history:List,
                    optimial_point:List,
                    cnot_num_history:List,
                    iteration_history:List,
                    oo_value=None) -> None:
        """
        vqe_result_history:存储每一轮的energy value
        optimial_point: 存储每一轮的最优参数
        gradients_history:存储每一轮的梯度
        picked_history: 存储每一轮选中的index
        """
        self.vqe_result_history=vqe_result_history
        self.gradients_history = gradients_history
        self.picked_history =picked_history
        self.cnot_num_history = cnot_num_history
        self.optimial_point = optimial_point
        self.oo_value = oo_value
        self.iteration_history = iteration_history