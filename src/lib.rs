

mod types;
mod undirected_hypergraphs;

use ndarray::{Array2, ArrayView2};
use numpy::{ToPyArray, PyArray2};

//use arrow2::ffi;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::{PyAny, PyObject, PyResult};


use undirected_hypergraphs::*;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}


pub fn py_dataframe_to_rust_data(df: &PyAny) -> PyResult<(Vec<String>, Array2<u8>)> {
    
    let cols: Vec<String> = df.getattr("columns")?.extract()?;
    print_type_of(&cols);
    println!("{:?}", cols);
    
    
    let data: &PyArray2<u8> = df.call_method0("to_numpy")?.extract()?;
    let array: Array2<u8> = data.to_owned_array();
    
    print_type_of(&array);
    println!("{:?}", array);
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
    pub edge_list: Vec<Vec<usize>>, 
    #[pyo3(get, set)]
    pub node_list: Vec<usize>, 
}

#[pymethods]
impl Hypergraph {
    #[new]
    fn new(data: Option<&PyArray2<u8>>) -> Hypergraph {
        
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
    
    fn compute_hypergraph(
        &self, 
        df: &PyAny//&PyArray2<u8>
    ) ->  Hypergraph {
        
        println!("{:?}", df);
        
        let (cols, data) = py_dataframe_to_rust_data(df).unwrap();
        
        println!("{:?}", cols);
        println!("{:?}", data);
        
        //println!("{:?}", data["disease_0"]);
        //println!("{:?}", data["disease_1"]);
        //println!("{:?}", data["disease_2"]);
        
        Python::with_gil(|py| 
            Hypergraph{
                incidence_matrix: PyArray2::zeros(py, [0,0], false).into(), 
                edge_weights: Vec::new(),
                node_weights: Vec::new(),
                edge_list: vec![Vec::new()], 
                node_list: Vec::new(), 
            }
        )
        /*
        Python::with_gil(|py| 
            Hypergraph{
                incidence_matrix: PyArray2::from_owned_array(py, h.incidence_matrix).into(), 
                edge_weights: h.edge_weights,
                node_weights: h.node_weights,
                edge_list: h.edge_list, 
                node_list: h.node_list, 
            }
        )
        */
    }
}


#[pymodule]
pub fn multimorbidity_hypergraphs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Hypergraph>()?;
    Ok(())
}