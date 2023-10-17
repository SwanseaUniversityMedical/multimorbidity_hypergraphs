

pub mod types;
pub mod undirected_hypergraphs;

use numpy::PyArray2;
use pyo3::prelude::*;


#[pymodule]
fn multimorbidity_hypergraphs(_py: Python, m: &PyModule) -> PyResult<()> {
    //m.add_class::<Hypergraph>()?;
    
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
        fn new(data: &PyArray2<u8>) -> Self {
            Python::with_gil(|py| 
                Hypergraph{
                    incidence_matrix: PyArray2::zeros(py, [1,1], false).into(), 
                    edge_weights: vec![0.0],
                    node_weights: vec![0.0],
                    edge_list: vec![vec![0]], 
                    node_list: vec![0], 
                }
            )
        }
    }
    

    Ok(())
}