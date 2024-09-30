#function for adiabatic mps prep project
import numpy as np 
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock3.block2.io import MPSTools, MPOTools,SubTensor
from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.core import SparseTensor
from scipy.linalg import null_space
import re

def get_mps_tensor(ket):
    pyket = MPSTools.from_block2(ket)
    n_sites = pyket.n_sites
    ket = pyket.tensors[0]
    for i in range(n_sites-1):
        ket = np.tensordot(ket,pyket.tensors[i+1],axes = 1)
    return ket

def get_rdo(ket,tracing_list):
    tracing_list = [x+1 for x in tracing_list]
    tracing_list.append(0)
    tracing_list.append(-1)
    if len(tracing_list) > ket.n_blocks+2:
        raise TypeError("Trace too many sites")
    rho = np.tensordot(ket,np.conj(ket),[tracing_list,tracing_list])
    return rho

def quantum_labels2basis(quantum_label,pg):
    if pg == 0:
        det = []
        for qnumber in quantum_label:
            quantum_number = (qnumber.n,qnumber.twos,qnumber.pg)
            if quantum_number==(0,0,0):
                a = 0
            elif quantum_number==(2,0,0):
                a = 3
            elif quantum_number==(1,1,0):
                a = 1
            elif quantum_number==(1,-1,0):
                a = 2
            else:
                raise TypeError("something wrong with the quantum label")
            det.append(a)
        det = tuple(det)
    if pg == 1:
        det = []
        for qnumber in quantum_label:
            quantum_number = (qnumber.n,qnumber.twos)
            if quantum_number==(0,0):
                a = 0
            elif quantum_number==(2,0):
                a = 3
            elif quantum_number==(1,1):
                a = 1
            elif quantum_number==(1,-1):
                a = 2
            else:
                raise TypeError("something wrong with the quantum label")
            det.append(a)
        det = tuple(det)

    return det

def rdo2rdm(rdo):
    rdm = np.zeros((4,) *len(rdo[0].q_labels))
    for number,tensor in enumerate(rdo):
        location = quantum_labels2basis(tensor.q_labels,1)
        value = np.asarray(tensor).item()
        rdm[location] = value
    rdm = rdm.reshape(2**len(rdo[0].q_labels),-1)
    return rdm


def get_rdm_nblock(ket,size_of_block):
    n_sites = ket.n_sites
    rho = {}
    allsites = np.arange(0,ket.n_sites)
    for i in range(ket.n_sites):
        keep_list = np.arange(i,i+size_of_block,1)
        trace_list = [x for x in allsites if x not in keep_list]
        
        if len(keep_list)+len(trace_list)==n_sites:
            mps = get_mps_tensor(ket)
            rdo = get_rdo(mps,trace_list)
            rdm = rdo2rdm(rdo)
            rho["site"+format(keep_list[0])+"to"+format(keep_list[-1])] = rdm
        else:
            break
            ###how to get ride of the fk extra keep_list output
    return rho

def get_rdm_nblock_pbc(ket,size_of_block):
    n_sites = ket.n_sites
    rho = {}
    allsites = np.arange(0,ket.n_sites)
    n = len(allsites)
    for i in range(ket.n_sites):
        block = np.concatenate((allsites[i % n:], allsites[:(i + size_of_block) % n]))[:size_of_block]
        mps = get_mps_tensor(ket)
        trace_list = [x for x in allsites if x not in block]
        rdo = get_rdo(mps,trace_list)
        rdm = rdo2rdm(rdo)
        rho["site"+format(block[0])+"to"+format(block[-1])] = rdm
    return rho

def local_projector_list(rdm, block_size):
    # Compute the kernel (null space) of the reduced density matrix (rdm)
    kernel_space = null_space(rdm).T
    
    # Initialize an empty list to store projectors
    projector_list = []
    # Loop over each vector in the null space
    for vector in kernel_space:
        # Compute the outer product to form the projector
        projector = np.outer(vector, vector.conj())
        # Append the projector to the list
        projector_list.append(projector)
    
    return projector_list, block_size

def local_projector_state_list(rdm):
    # Compute the kernel (null space) of the reduced density matrix (rdm)
    kernel_space = null_space(rdm).T
    
    
    return kernel_space

def local_projector(rdm,block_size):
    return np.sum(local_projector_list(rdm,block_size)[0],axis = 0)
#size of local_projector should be equal to (4**block_size,4**block_size)

def projector_state_all(rdm_dict,block_size,system_size):
    projector = {}
    
    for key in rdm_dict:
        numbers = re.findall(r'\d+', key)
        sites = [int(num) for num in numbers]
        projector_state_local= local_projector_state_list(rdm_dict[key])
        before = np.arange(0,sites[0],1)
        after = np.arange(sites[-1]+1,system_size,1)
        before_state = normalize_vector(np.ones(4**len(before)))
        after_state = normalize_vector(np.ones(4**len(after)))
        project_all_state_list = [np.kron(np.kron(before_state,projectors),after_state) for projectors in projector_state_local]
        projector[key] = project_all_state_list
    return projector

def projector_state_all_pbc(rdm_dict,block_size,system_size):
    projector = {}
    
    for key in rdm_dict:
        numbers = re.findall(r'\d+', key)
        sites = [int(num) for num in numbers]
        projector_state_local= local_projector_state_list(rdm_dict[key])
        before = np.arange(0,sites[0],1)
        after = np.arange(sites[-1]+1,system_size,1)
        before_state = normalize_vector(np.ones(4**len(before)))
        after_state = normalize_vector(np.ones(4**len(after)))
        project_all_state_list = [np.kron(np.kron(before_state,projectors),after_state) for projectors in projector_state_local]
        projector[key] = project_all_state_list
    return projector


def projector_all(rdm_dict,block_size,system_size):

    projector = {}
    
    for key in rdm_dict:
        numbers = re.findall(r'\d+', key)
        sites = [int(num) for num in numbers]
        projector_local,block_size = local_projector_list(rdm_dict[key],block_size)
        before = np.arange(0,sites[0],1)
        after = np.arange(sites[-1]+1,system_size,1)
        before_ham = np.eye(4**len(before))
        after_ham = np.eye(4**len(after))
        project_all_list = [np.kron(np.kron(before_ham,projectors),after_ham) for projectors in projector_local]
        projector[key] = project_all_list
    return projector


def mps2state(mps):
    system_size = len(mps[0].q_labels)-2
    state = np.zeros((4,) *system_size)
    for number,tensor in enumerate(mps):
        location = ab.quantum_labels2basis(tensor.q_labels[1:-1],1)
        value = np.asarray(tensor).item()
        state[location] = value
    state = state.reshape(1,-1)
    return state

#test functions
def is_hermitian(matrix):
    # Compute the conjugate transpose of the matrix
    conjugate_transpose = np.conj(matrix).T
    
    # Check if the matrix is equal to its conjugate transpose
    return np.array_equal(matrix, conjugate_transpose)

def are_diagonal_elements_close_to_one(matrix, tolerance=1e-8):
    # Extract the diagonal elements
    diagonal_elements = np.diag(matrix)
    
    # Check if all diagonal elements are close to 1
    return np.allclose(diagonal_elements, 1, atol=tolerance)


def is_diagonal(matrix, tolerance=1e-8):
    """
    Check if a matrix is diagonal.

    Parameters:
    matrix (ndarray): Input matrix.
    tolerance (float): Tolerance for comparison. Default is 1e-8.

    Returns:
    bool: True if the matrix is diagonal, False otherwise.
    """
    # Create a diagonal matrix from the diagonal elements of the input matrix
    diagonal_matrix = np.diag(np.diag(matrix))
    
    # Check if the input matrix is close to the diagonal matrix
    return np.allclose(matrix, diagonal_matrix, atol=tolerance)



def are_mutually_orthonormal(vectors, tolerance=1e-8):
    """
    Check if a list of vectors are mutually orthonormal.

    Parameters:
    vectors (list of ndarray): List of vectors to be checked.
    tolerance (float): Tolerance for floating point comparison. Default is 1e-8.

    Returns:
    bool: True if the vectors are mutually orthonormal, False otherwise.
    """
    num_vectors = len(vectors)
    
    # Check normalization
    for vec in vectors:
        if not np.isclose(np.linalg.norm(vec), 1, atol=tolerance):
            return False
    
    # Check orthogonality
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            if not np.isclose(np.dot(vectors[i], vectors[j]), 0, atol=tolerance):
                return False
    
    return True

def sort_eigenpairs(eigenvalues, eigenvectors, order='ascending'):
    """
    Sorts eigenvalues and corresponding eigenvectors.

    Parameters:
    eigenvalues (ndarray): Array of eigenvalues.
    eigenvectors (list of ndarray): List where each element is an eigenvector.
    order (str): 'ascending' or 'descending' for sorting order.

    Returns:
    sorted_eigenvalues (ndarray): Sorted eigenvalues.
    sorted_eigenvectors (list of ndarray): Reordered eigenvectors corresponding to sorted eigenvalues.
    """
    # Determine the sorting order
    if order == 'ascending':
        sorted_indices = np.argsort(eigenvalues)
    elif order == 'descending':
        sorted_indices = np.argsort(eigenvalues)[::-1]
    else:
        raise ValueError("Order must be 'ascending' or 'descending'")

    # Sort the eigenvalues using the sorted indices
    sorted_eigenvalues = eigenvalues[sorted_indices]

    # Reorder the eigenvectors using the sorted indices
    sorted_eigenvectors = [eigenvectors[i] for i in sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors


def normalized_random_vector(size):
    """
    Generate a normalized random vector.

    Parameters:
    size (int): The size of the random vector.

    Returns:
    ndarray: A normalized random vector of the specified size.
    """
    # Generate a random vector with elements drawn from a standard normal distribution
    random_vector = np.random.randn(size)
    
    # Compute the norm of the vector
    norm = np.linalg.norm(random_vector)
    
    # Normalize the vector
    normalized_vector = random_vector / norm
    
    return normalized_vector


    
def normalize_vector(vector):
    """
    Normalize a vector.

    Parameters:
    vector (ndarray): Input vector.

    Returns:
    ndarray: Normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def merge_dict_elements(dictionary):
    """
    Merge all elements in a dictionary into one list.

    Parameters:
    dictionary (dict): Dictionary with lists as values.

    Returns:
    list: A single list containing all elements from the dictionary values.
    """
    merged_list = []
    for value_list in dictionary.values():
        merged_list.extend(value_list)
    return merged_list

def cumulative_sum_matrices(matrices):
    """
    Create a cumulative sum list of matrices in place to save memory.

    Parameters:
    matrices (list of ndarray): List of matrices.

    Returns:
    list of ndarray: A list where each element is the cumulative sum of the matrices up to that index.
    """
    matrix_sum = []
    if not matrices:
        return []

    # Initialize the running sum as a zero matrix of the same shape as the first matrix
    running_sum = np.zeros_like(matrices[0])
    
    for i in range(len(matrices)):
        running_sum += matrices[i]
        matrix_sum.append(running_sum)  # Directly store the cumulative sum in the original list

    return matrices

def gap_of_list(values):
    """
    Calculate the difference between the two smallest values in a list using NumPy.

    Parameters:
    values (list or ndarray of numbers): The list of numbers.

    Returns:
    number: The difference between the smallest and the second smallest values.
    """
    if len(values) < 2:
        raise ValueError("Array must contain at least two elements")

    # Convert the list to a NumPy array if it isn't already
    values = np.array(values)
    
    # Use np.partition to get the two smallest elements
    smallest_two = np.partition(values, 1)[:2]
    
    # Calculate the difference
    difference = np.diff(smallest_two)[0]
    
    return difference

def svd_independent(basis):
    U, S, Vt = np.linalg.svd(basis)
    rank = np.sum(S > 1e-10)  # Consider numerical precision
    independent_indices = np.where(S > 1e-9)[0]
    independent_basis = [Vt[i] for i in independent_indices]
    independent_basis = np.array(independent_basis)
    return independent_basis

def gram_matrix(v):
    #v is a set of row 
    v = np.array(v)
    gram_matrix = np.dot(v,v.T)
    return gram_matrix