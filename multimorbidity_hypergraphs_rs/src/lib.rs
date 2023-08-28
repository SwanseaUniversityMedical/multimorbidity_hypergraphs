
// directives to be used during development only. 
#[allow(dead_code)]
#[allow(unused_imports)]
// done

//use ndarray::prelude::*;
use ndarray::{array, Array2, ArrayView1};
use itertools::Itertools;

use std::collections::HashSet;
//use std::convert::From;


/*fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}*/

fn reduced_powerset(data_row: ArrayView1<u8>) -> HashSet<Vec<usize>> {

	let mut out = HashSet::new();
	let indices = data_row
		.iter()
		.enumerate()
		.filter(|(_, &r)| r >= 1)
		.map(|(index, _)| index)
		.collect::<Vec<_>>();
	

	for ii in 2..(indices.len()+1) {
		let combs = indices
			.clone()
			.into_iter()
			.combinations(ii)
			.collect::<HashSet<_>>();
		out.extend(combs);
	}

	out
}


#[derive(Default)]
pub struct Hypergraph {
	incidence_matrix: Option<Array2<u8>>, 
	edge_weights: Option<Vec<f32>>,
	node_weights: Option<Vec<f32>>,
	edge_list: Option<Vec<u8>>, // TODO(jim): decide on this type (HashSet<Vec<u8 / usize>>?)
	node_list: Option<Vec<u8>>, // TODO(jim): decide on this type (HashSet<u8 / usize>?)
}


// Idiomatic rust is apparently to have tests and code in the same file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instantiated() {
		//     ** Copied from the python implementation ** 
		//     Tests the instantiation of the hypergraph object.
		//     Pretty simple test as all internal state is set to None.
		
        let h = Hypergraph{..Default::default()};
		
        assert!(h.incidence_matrix.is_none());
		assert!(h.edge_weights.is_none());
		assert!(h.node_weights.is_none());
		assert!(h.edge_list.is_none());
		assert!(h.node_list.is_none());
    }
	
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
		
	
	/*
	#[test]
	fn construct_edge_list_t() {
		// Not part of the python implementation
		// Tests the function to construct the edge list
		
		/*let data = array![
			[1, 0, 1, 0],
			[0, 1, 0, 0],
			[0, 1, 0, 1],
			[0, 1, 1, 1],
			[1, 1, 1, 1]
		];
		
		println!("{:?}", data.nrows());
		println!("{:?}", data.ncols());*/
		
	}
	
	#[test]
	fn compute_weights_t() {
		// Not part of the python implementation
		// Tests the computation of the weights
		
		
	}
	*/
}
