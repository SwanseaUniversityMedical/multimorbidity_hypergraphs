
// directives to be used during development only. 
#[allow(dead_code)]
#[allow(unused_imports)]
// done

//use ndarray::prelude::*;
use ndarray::{array, s, Array2, ArrayView1, ArrayView2, Axis};
use itertools::Itertools;

use std::collections::HashSet;
//use std::convert::From;


/*fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}*/


fn overlap_coefficient(data: ArrayView2<u8>) -> f32 {
    
	let denom = data
        .axis_iter(Axis(1))
        .map(|x| x.sum())
        .min()
        .unwrap() as f32;
		
	if denom < 0.5 { 
	// NOTE(jim): this is an integer count cast to a f32, so if it's less than 
	// 1.0 - eps it's zero and the code should panic.
		panic!("overlap_coefficient: denominator is zero.");
	}
	
    data
        .axis_iter(Axis(0))
        .filter(|data_row| usize::from(data_row.sum()) == data_row.len())
        .count() as f32 / denom
    // NOTE(jim): .fold may be faster than .filter.count
       
}

fn reduced_powerset(data_row: ArrayView1<u8>) -> HashSet<Vec<usize>> {

    // more functional approach. Test for speed later?
    // don't foget automatic returns
	(2..=data_row.iter().map(|x| (x > &0) as usize).sum::<usize>())
	    .map(|ii| 
    	    data_row
    		.iter()
    		.enumerate()
    		.filter(|(_, &r)| r >= 1)
    		.map(|(index, _)| index)
    		.combinations(ii)
    		.collect::<HashSet<_>>()
    	)
    	.flatten()
    	.collect::<HashSet<_>>()
}

fn construct_edge_list(data: Array2<u8>) -> HashSet<Vec<usize>> {
    
    // More functional programming... 
    data
        .axis_iter(Axis(0))
        .map(|x| reduced_powerset(x))
        .flatten()
        .collect::<HashSet<_>>()
    
}


#[derive(Default)]
pub struct Hypergraph {
    incidence_matrix: Array2<u8>, 
    edge_weights: Vec<f32>,
    node_weights: Vec<f32>,
    edge_list: Vec<u8>, // TODO(jim): decide on this type (HashSet<Vec<u8 / usize>>?)
    node_list: Vec<u8>, // TODO(jim): decide on this type (HashSet<u8 / usize>?)
}


// Idiomatic rust is apparently to have tests and code in the same file
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn reduced_powerset_t() {
        // Not part of the python implementation
        // Tests the function to construct the reduced powerset
        
        let data: Array2<u8> = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        let mut expected = HashSet::new();
        expected.insert(vec![1,2,3]);
        expected.insert(vec![1,2]);
        expected.insert(vec![2,3]);
        expected.insert(vec![1,3]);
        
        assert_eq!(
            reduced_powerset(data.row(3)),
            expected
        );
        
    }
    
    #[test]
    fn reduced_powerset_singleton_t() {
        // Not part of the python implementation
        // Tests the function to construct the reduced powerset
        
        let data: Array2<u8> = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        let mut expected = HashSet::new();
        expected.insert(vec![0,2]);
        
        assert_eq!(
            reduced_powerset(data.row(0)),
            expected
        );
        
    }
    
    #[test]
    fn reduced_powerset_emptyset_t() {
        // Not part of the python implementation
        // Tests the function to construct the reduced powerset
        
        let data: Array2<u8> = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        let expected = HashSet::new();
        
        assert_eq!(
            reduced_powerset(data.row(1)),
            expected
        );
        
    }
        
    
    
    #[test]
    fn construct_edge_list_t() {
        // Not part of the python implementation
        // Tests the function to construct the edge list
        
        let data = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        let mut expected = HashSet::new();
        
        expected.insert(vec![0,1,2,3]);
        
        expected.insert(vec![0,1,2]);
        expected.insert(vec![0,1,3]);
        expected.insert(vec![0,2,3]);
        expected.insert(vec![1,2,3]);
        
        expected.insert(vec![0,1]);
        expected.insert(vec![0,2]);
        expected.insert(vec![0,3]);
        expected.insert(vec![1,2]);
        expected.insert(vec![1,3]);
        expected.insert(vec![2,3]);
        
        assert_eq!(
            construct_edge_list(data),
            expected
        );

        
    }
    
    
    #[test]
    fn overlap_coefficient_t() {
        // Not part of the python implementation
        // Tests the computation of the overlap coefficient
        
        let data = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        assert_eq!(
            overlap_coefficient(data.slice(s![.., 1..=2])),
            2.0 / 3.0
        );
        
    }
	
	
    #[test]
	#[should_panic]
    fn overlap_coefficient_divide_by_zero_t() {
        // Not part of the python implementation
        // Tests the computation of the overlap coefficient
        
        let data = array![
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1]
        ];
        
        overlap_coefficient(data.slice(s![.., 1..=2]));
        
    }
    
}
