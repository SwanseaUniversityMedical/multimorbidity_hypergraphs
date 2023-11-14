

pub mod types;
pub mod undirected_hypergraphs;

use ndarray::Array2;
use numpy::{ToPyArray, PyArray2};

use pyo3::prelude::*;
use pyo3::{PyAny, PyResult};
use pyo3::types::PyTuple;
use pyo3::exceptions::PyException;


use undirected_hypergraphs::*;
use types::*; 

/*
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
*/

pub fn py_dataframe_to_rust_data(df: &PyAny) -> PyResult<(Vec<String>, Array2<u8>)> {
    
    let cols: Vec<String> = df.getattr("columns")?.extract()?;
    let data: &PyArray2<u8> = df.call_method0("to_numpy")?.extract()?;
    let array: Array2<u8> = data.to_owned_array();

    Ok((cols, array))
}



#[pyclass]
pub struct Hypergraph{
    #[pyo3(get, set)]
    pub incidence_matrix: Py<PyArray2<u8>>, 
    #[pyo3(get, set)]
    pub edge_weights: Vec<f64>,
    #[pyo3(get, set)]
    pub node_weights: Vec<f64>,
    #[pyo3(get, set)]
    pub edge_list: Vec<PyObject>, 
    #[pyo3(get, set)]
    pub node_list: Vec<String>,
}

trait ToRust<T> {
    fn to_rust(&self) -> T;
}

#[pymethods]
impl Hypergraph {
    #[new]
    fn new(data: Option<&PyAny>) -> Self {
        
        match data {
            Some(x) => {
                
                // TODO - figure out if it's possible to call a self method
                // from this constructor. Until then we have C&P code... :(
                
                let (cols, data) = py_dataframe_to_rust_data(x).unwrap();
                let h = compute_undirected_hypergraph(&data);
                
                Python::with_gil(|py| 
                    Hypergraph{
                        incidence_matrix: h.incidence_matrix.to_pyarray(py).to_owned(),
                        edge_weights: h.edge_weights,
                        node_weights: h.node_weights,
                        edge_list: h.edge_list
                            .iter()
                            .map(|edge| 
                                PyTuple::new(py,
                                    edge
                                        .iter()
                                        .map(|ii| cols[*ii].clone())
                                        .collect::<Vec<String>>()
                                ).to_object(py)
                            )
                            .collect::<Vec<PyObject>>(),
                        node_list: h.node_list
                            .iter()
                            .map(|ii| cols[*ii].clone())
                            .collect::<Vec<String>>(),
                })
            },
            None => Python::with_gil(|py| 
                Hypergraph{
                    incidence_matrix: PyArray2::zeros(py, [0,0], false).into(), 
                    edge_weights: Vec::new(),
                    node_weights: Vec::new(),
                    edge_list: vec![PyTuple::new(py, Vec::<String>::new()).to_object(py)], 
                    node_list: Vec::new(), 
                }
            ),
        }
    }
    
    fn compute_hypergraph(
        &mut self, 
        df: &PyAny
    ) {
        
        let (cols, data) = py_dataframe_to_rust_data(df).unwrap();
        let h = compute_undirected_hypergraph(&data);
        
        
        Python::with_gil(|py| {
            self.incidence_matrix = h.incidence_matrix.to_pyarray(py).to_owned();
            self.edge_weights = h.edge_weights;
            self.node_weights = h.node_weights;
            self.edge_list = h.edge_list
                .iter()
                .map(|edge| 
                    PyTuple::new(py,
                        edge
                            .iter()
                            .map(|ii| cols[*ii].clone())
                            .collect::<Vec<String>>()
                    ).to_object(py)
                )
                .collect::<Vec<PyObject>>();
            self.node_list = h.node_list
                .iter()
                .map(|ii| cols[*ii].clone())
                .collect::<Vec<String>>();  
        });
    }
    
    fn eigenvector_centrality(
        &self,
        rep: Option<String>,
        weighted_resultant: Option<bool>,
        tolerance: Option<f64>,
        max_iterations: Option<u32>, 
    ) -> PyResult<Vec<f64>> {
        
        let representation = match rep {
            Some(str_x) => {
                match str_x.as_ref() {
                    "standard" => Representation::Standard,
                    "dual" => Representation::Dual,
                    "bipartite" => Representation::Bipartite,
                    _ => return Err(
                        PyException::new_err(
                            "Error: Requested representation not supported."
                        )
                    )
                }
            }
            None => Representation::Standard
        };
        
        let wr = match weighted_resultant {
            Some(x) => x,
            None => false,
        };
        
        let tol = match tolerance {
            Some(x) => x,
            None => 1e-6,
        };
        
        let iterations = match max_iterations {
            Some(x) => x,
            None => 100,
        };
        
        Ok(
            eigenvector_centrality(
                &self.to_rust(),
                iterations,
                tol,
                representation,
                wr
            )
        )
    }
    
    
    fn degree_centrality(
        &self,
        rep: Option<String>,
        weighted: Option<bool>
    ) -> PyResult<Vec<f64>> {
    
        let representation = match rep {
            Some(str_x) => {
                match str_x.as_ref() {
                    "standard" => Representation::Standard,
                    "dual" => Representation::Dual,
                    _ => return Err(
                        PyException::new_err(
                            "Error: Requested representation not supported."
                        )
                    )
                }
            }
            None => Representation::Standard
        };
        
        let weight = match weighted {
            Some(x) => x,
            None => true
        };
        
        Ok(
            degree_centrality(
                &self.to_rust(),
                representation,
                weight
            )
        )
    }
}

impl ToRust<HypergraphBase> for Hypergraph {
    fn to_rust(&self) -> HypergraphBase {
        
        Python::with_gil(|py|
            HypergraphBase{
                incidence_matrix: self.incidence_matrix
                    .as_ref(py)
                    .to_owned_array(), 
                edge_weights: self.edge_weights.clone(),
                node_weights: self.node_weights.clone(),
                edge_list: self.edge_list
                    .iter()
                    .map(|edge|
                        edge 
                            .as_ref(py)
                            .downcast::<PyTuple>()
                            .unwrap()
                            .iter()
                            .map(|node| 
                                self.node_list
                                    .iter()
                                    .position(|node_str| 
                                        *node_str == node.extract::<String>().unwrap()
                                    )
                                    .unwrap()
                            )
                            .collect::<Vec<usize>>()
                    )
                    .collect::<Vec<_>>(), 
                node_list: (0..self.node_list.len()).collect::<Vec<usize>>(), 
            }
        )
    }
}

#[pymodule]
pub fn multimorbidity_hypergraphs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Hypergraph>()?;
    Ok(())
}