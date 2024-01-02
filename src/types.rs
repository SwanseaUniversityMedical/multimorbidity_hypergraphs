
use pyo3::pyclass;
use ndarray::{Array1, Array2};
use std::collections::HashSet;
use indexmap::IndexSet;

pub enum Representation {
    Standard,
    Dual,
    Bipartite,
}

#[derive(Debug)]
pub struct HypergraphBase {
    pub incidence_matrix: Array2<u8>, 
    pub edge_weights: Vec<f64>,
    pub node_weights: Vec<f64>,
    pub edge_list: Vec<Vec<usize>>, 
    pub node_list: Vec<usize>, 
}


// TODO - was trying to keep the python specific bits of the code
// to just lib.rs. Decide whether to have a lib.rs and a py_types.rs
// and then eventually r_lib.rs and r_types.rs...
// Alternative is probably a lot of copying...
#[derive(Clone, Debug, PartialEq)]
#[pyclass]
pub struct HyperArc {
    #[pyo3(get, set)]
    pub tail: HashSet<i8>,
    #[pyo3(get, set)]
    pub head: i8
}

#[derive(Debug)]
pub struct DiHypergraphBase {
    pub incidence_matrix: Array2<i8>, 
    pub hyperedge_list: IndexSet<Array1<i8>>,
    pub hyperedge_weights: Array1<f64>,
    pub hyperarc_list: Array1<HyperArc>,
    pub hyperarc_weights: Array1<f64>,
}
