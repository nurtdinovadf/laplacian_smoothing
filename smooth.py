import numpy as np
import trimesh

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numba
from numba import jit
from numba.types import List, int32, float32

def smooth(mesh, lambdadt = 10, normalize = True):
    
    # normalization
    if normalize:
        vertices = np.array(mesh.vertices)
        vertices = normalize_vertices(vertices)
        mesh.vertices = vertices
    
    # get cotangent matrix 
    matrixcot = create_edge_cotangent_match(mesh)
    
    # get coefficients
    matrix = calculate_coefficients(mesh, lambdadt, matrixcot)
    
    # solve ls
    sol = solve_system(mesh, matrix)
    
    # update verticies
    newmesh = mesh.copy()
    newmesh.vertices = sol
    
    # find outline verticies
    bound_vertices = find_outline(mesh.face_adjacency_edges, mesh.edges)
    
    # restore outline vertices coordinates
    for v in bound_vertices:
        newmesh.vertices[v] = mesh.vertices[v]
        
    return newmesh

@jit(nopython=True, parallel=True)
def normalize_vertices(vertices):
    
    c_max = np.max(np.absolute(vertices))
    c_min = np.min(np.absolute(vertices))
    vertices = 2 * (vertices - c_min) / (c_max - c_min) - 1
    
    return vertices

        

def create_edge_cotangent_match(mesh):
    
    row, col, datacot = create_cotmatrix_components(mesh.face_adjacency_edges, 
                                                    mesh.face_adjacency_unshared, 
                                                    mesh.vertices)
    matrixcot = csr_matrix((datacot, (row, col)), shape=(mesh.vertices.shape[0], mesh.vertices.shape[0]), dtype = 'float16')
    
    return matrixcot

# @jit(numba.types.Tuple((numba.typeof([0]), numba.typeof([0]), numba.typeof([0.0])))(numba.int32[:, :], numba.int32[:,:], numba.float32[:,:]), nopython=True, parallel=True)
@jit(nopython=True, parallel=True)
def create_cotmatrix_components(face_adjacency_edges, face_adjacency_unshared, vertices):
    
    row = []
    col = []
    datacot = []
    l = len(face_adjacency_edges)
    for i in np.arange(0, l, 1):
        point1 = face_adjacency_unshared[i][0]
        point2 = face_adjacency_unshared[i][1]
        pointe1 = face_adjacency_edges[i][0]
        pointe2 = face_adjacency_edges[i][1]
        
        v1 = vertices[pointe1] - vertices[point1]
        v2 = vertices[pointe2] - vertices[point1]
        v3 = vertices[pointe1] - vertices[point2]
        v4 = vertices[pointe2] - vertices[point2]
        
        lv1 = np.linalg.norm(v1)
        lv2 = np.linalg.norm(v2)
        lv3 = np.linalg.norm(v3)
        lv4 = np.linalg.norm(v4)
        
        cosalpha = np.dot(v1, v2) / lv1 / lv2
        cosbeta = np.dot(v3, v4) / lv3 / lv4
        sinalpha = (1 - cosalpha ** 2) ** 0.5
        sinbeta = (1 - cosbeta ** 2) ** 0.5
        cotalpha = cosalpha / sinalpha
        cotbeta = cosbeta / sinbeta
        row.append(pointe1)
        row.append(pointe2)
        col.append(pointe2)
        col.append(pointe1)
#         col += [pointe2, pointe1]
#         datacot += [cotalpha + cotbeta, cotalpha + cotbeta]
        datacot.append(cotalpha + cotbeta)
        datacot.append(cotalpha + cotbeta)
        
    return (row, col, datacot)

    
def calculate_coefficients(mesh, lambdadt, matrixcot):
    # csr matrix components
    row = []
    col = []
    data = []
    
    graph = mesh.vertex_adjacency_graph
    
    # linear system coefficients calculation
    for i, v in enumerate(mesh.vertices):
        m = sum([1 for _ in graph.neighbors(i)])
        cot = sum([matrixcot[i, j] for j in graph.neighbors(i)])
        row += [i]
        col += [i]
        if cot == 0:
            data += [1]
        else:
            data += [1 + lambdadt]
            for j in graph.neighbors(i):
                row += [i]
                col += [j]
                data += [-1 * lambdadt * matrixcot[i, j] / cot]
    matrix = csr_matrix((data, (row, col)), shape=(mesh.vertices.shape[0], mesh.vertices.shape[0]))
    return matrix

def solve_system(mesh, matrix):
    # bias
    b0 = mesh.vertices[:, 0]
    b1 = mesh.vertices[:, 1]
    b2 = mesh.vertices[:, 2]
    
    # solution
    sol0 = spsolve(matrix, b0)
    sol1 = spsolve(matrix, b1)
    sol2 = spsolve(matrix, b2)
    sol = np.transpose(np.array([sol0, sol1, sol2]), (1,0))
    
    return sol


def find_outline(face_adjacency_edges, edges):
    
    adjacent_edges = set([(edge[0] if edge[0] <= edge[1] else edge[1], 
                   edge[0] if edge[0] > edge[1] else edge[1]) for edge in face_adjacency_edges])
    all_edges = set([(edge[0] if edge[0] <= edge[1] else edge[1], 
                       edge[0] if edge[0] > edge[1] else edge[1]) for edge in edges])
    outline = list(all_edges - adjacent_edges)
    bound_vertices = list(set([v for e in outline for v in e]))
    
    return bound_vertices