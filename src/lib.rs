

pub mod types;
pub mod undirected_hypergraphs;

use numpy::PyArray2;
use pyo3::prelude::*;



#[pyclass]
pub struct Hypergraph{
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

#[pymethods]
impl Hypergraph {
    #[new]
    fn new(data: Option<&PyArray2<u8>>) -> Self {
        
        match data {
            Some(x) => Python::with_gil(|py| 
                Hypergraph{
                    incidence_matrix: PyArray2::zeros(py, [0,0], false).into(), 
                    edge_weights: vec![0.0, 1.0],
                    node_weights: Vec::new(),
                    edge_list: vec![Vec::new()], 
                    node_list: Vec::new(), 
                }
            ),
            None => Python::with_gil(|py| 
                Hypergraph{
                    incidence_matrix: PyArray2::zeros(py, [0,0], false).into(), 
                    edge_weights: Vec::new(),
                    node_weights: Vec::new(),
                    edge_list: vec![Vec::new()], 
                    node_list: Vec::new(), 
                }
            ),
        }
    }
}


#[pymodule]
pub fn multimorbidity_hypergraphs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Hypergraph>()?;
    Ok(())
}