
// directives to be used during development only. 
#[allow(dead_code)]
// done


#[derive(Default)]
pub struct Hypergraph {
	incidence_matrix: Option<Vec<u8>>, // TODO(jim): decide whether this containiner is right
	edge_weights: Option<Vec<f32>>,
	node_weights: Option<Vec<f32>>,
	edge_list: Option<Vec<u8>>, // TODO(jim): decide on this (internal) type
	node_list: Option<Vec<u8>>, // TODO(jim): decide on this (internal) type 
}

// options for the incidence matrix:
// Vec<u8> and cause myself all sorts of pain with indexing
// Multiarray crate: Array2D::new([3, 2], 0); 
// Vec<Vec<u8>> probably too slow


// Idiomatic rust is apparently to have tests and code in the same file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instantiated() {
		//     Tests the instantiation of the hypergraph object.
		//     Pretty simple test as all internal state is set to None.
		
        let h = Hypergraph{..Default::default()};
		
        assert!(h.incidence_matrix.is_none());
		assert!(h.edge_weights.is_none());
		assert!(h.node_weights.is_none());
		assert!(h.edge_list.is_none());
		assert!(h.node_list.is_none());
    }
}
