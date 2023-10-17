
use ndarray::Array2;


pub enum Representation {
    Standard,
    Dual,
    Bipartite,
}

#[derive(Debug)]
pub struct Hypergraph {
    pub incidence_matrix: Array2<u8>, 
    pub edge_weights: Vec<f32>,
    pub node_weights: Vec<f32>,
    pub edge_list: Vec<Vec<usize>>, 
    pub node_list: Vec<usize>, 
}
