

mod types;
mod undirected_hypergraphs;

use ndarray::Array2;
use numpy::{ToPyArray, PyArray2};

use pyo3::prelude::*;
use pyo3::{PyAny, PyResult};
use pyo3::types::PyTuple;


use undirected_hypergraphs::*;

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
    pub edge_weights: Vec<f32>,
    #[pyo3(get, set)]
    pub node_weights: Vec<f32>,
    #[pyo3(get, set)]
    pub edge_list: Vec<PyObject>, 
    #[pyo3(get, set)]
    pub node_list: Vec<String>, 
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
                let h = compute_hypergraph(&data);
                
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
        let h = compute_hypergraph(&data);
        
        
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
}


#[pymodule]
pub fn multimorbidity_hypergraphs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Hypergraph>()?;
    Ok(())
}