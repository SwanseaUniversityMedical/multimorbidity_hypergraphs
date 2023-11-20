
//use crate::types::*;

use ndarray::{
    Array1,
    //Array2,
    //ArrayView1,
    s,
};
use std::collections::HashSet;

/*
pub fn compute_directed_hypergraph(
    data: &Array2<i8>
) -> DiHypergraphBase {
    
    DiHypergraphBase{incidence_matrix: Array2::zeros((1,1))}
    
}
*/

fn compute_single_progset(data_ind: &Array1<i8>) -> HashSet<Array1<i8>> {
    
    // NOTE - we are assuming that there are no duplicates in the ordering
    // ie, this is the simplest possible progression. 
    
    let n_diseases = data_ind
        .iter()
        .map(|&x| x > 0)
        .filter(|&x| x)
        .count();
    
    
    match n_diseases {
        1 => HashSet::from([data_ind.clone()]),
        
        _ => (1..data_ind.len())
            .filter(|&i| data_ind[i] >= 0)
            .map(|i| {
                let mut i_vec = data_ind.slice(s![0..(i + 1)]).to_vec();
                i_vec.extend(&vec![-1; data_ind.len() - 1 - i]);
                Array1::from_vec(i_vec)
            })
            .collect(),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    use ndarray::array;
    
    
    #[test]
    fn di_compute_progression_set_t () {
        
        let data = array![2, 0, 1];
        let expected = HashSet::from([
            array![ 2,  0, -1],
            array![ 2,  0,  1]
        ]);
        
        let out = compute_single_progset(&data);
        
        assert_eq!(out, expected);
    }
    
    
    #[test]
    fn di_compute_progression_set_singleton_t () {
        
        let data = array![2, -1, -1];
        let expected: HashSet<Array1<i8>> = HashSet::from([array![2, -1 ,-1]]);
        
        let out = compute_single_progset(&data);
        
        assert_eq!(out, expected);
    }    
    /*
    #[test]
    fn di_construct_dihypergraph() {
        
        let data = array![[0, 1, 2,],
            [ 0, 1, 2,],
            [ 0, 1, 2,],
            [ 2, 0, 1,],
            [ 1, 2,-1,],
            [ 0,-1,-1,],
            [ 2,-1,-1,],
            [ 1, 0, 2,],
            [ 0, 1,-1,],
            [ 0, 2,-1,],];
        
        let out = compute_directed_hypergraph(&data);
        assert!(false);   
    }
    */
}