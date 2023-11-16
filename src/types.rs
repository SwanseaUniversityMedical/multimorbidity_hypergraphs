
use ndarray::Array2;


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

#[derive(Debug)]
pub struct DiHypergraphBase {
    pub incidence_matrix: Array2<u8>, 
}
