

pub mod types;
pub mod undirected_hypergraphs;

use numpy::PyArray2;
use pyo3::prelude::*;



#[pyclass]
pub struct HypergraphPy{
    #[pyo3(get, set)]
    pub incidence_matrix: Py<PyArray2<u8>>, 
    #[pyo3(get, set)]
    pub edge_weights: Vec<f32>,
    #[pyo3(get, set)]
    pub node_weights: Vec<f32>,
    #[pyo3(get, set)]
    pub edge_list: Vec<Vec<usize>>, 
    #[pyo3(get, set)]
    pub node_list: Vec<usize>, 
}


#[pymodule]
fn multimorbidity_hypergraphs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HypergraphPy>()?;

    Ok(())
}