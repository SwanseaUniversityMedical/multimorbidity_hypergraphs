
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


#[derive(Debug, PartialEq)]
pub struct HyperArc {
    pub tail: HashSet<i8>,
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
