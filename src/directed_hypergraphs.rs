
//use crate::types::*;

use ndarray::{
    Array,
    Array1,
    Array2,
    Axis,
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

fn compute_incidence_matrix(data: &HashSet<Array1<i8>>) -> Array2<i8> {
    
    let data_vec: Vec<_> = data.into_iter().collect();
    
    let n_arcs = data_vec.len();
    let n_diseases = data_vec[0].len();

    //let data_array: Array2<_> = data.into_iter().collect();
    let mut data_array = Array2::<i8>::default((n_arcs, n_diseases));
    for (i, mut row) in data_array.axis_iter_mut(Axis(0)).enumerate() {
        for (j, col) in row.iter_mut().enumerate() {
            *col = data_vec[i][j];
        }
    }
    println!("{:?}", data_array);
    
    
    let mut hyperedges: HashSet<Array1<i8>> = HashSet::new();
    
    for a in data_vec.iter() {
        
        let mut edge = Array::zeros(n_diseases);
        
        let n_conds = a
            .iter()
            .map(|&x| x >= 0)
            .filter(|&x| x)
            .count();
        
        for j in 0..n_conds-1 {
            edge[[a[j] as usize]] = -1;
        }
        edge[[a[n_conds-1] as usize]] = 1;

        println!("{:?}, {:?}", a.to_vec(), edge.to_vec());
        hyperedges.insert(edge);
    }
    
    let n_edges = hyperedges.len();
    hyperedges
        .into_iter()
        .flat_map(|x| x)
        .collect::<Array1<_>>()
        .into_shape((n_edges, n_diseases))
        .unwrap()
       
        
}

fn compute_progset(data: &Array2<i8>) -> HashSet<Array1<i8>> {
    
    let n_rows = data.nrows();

    (0..n_rows)
        //.into_iter()
        .flat_map(|i| compute_single_progset(&data.index_axis(Axis(0), i).to_owned()))
        .collect()
    
}

fn compute_single_progset(data_ind: &Array1<i8>) -> HashSet<Array1<i8>> {
    
    // NOTE - we are assuming that there are no duplicates in the ordering
    // ie, this is the simplest possible progression. 
    
    let n_diseases = data_ind
        .iter()
        .map(|&x| x >= 0)
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
    
    #[test]
    fn di_compute_progression_set_cohort_t() {
        
        let data = array![
            [2, 0, 1],
            [2, -1, -1],
        ];
        
        let expected = HashSet::from([
            array![ 2,  0, -1],
            array![ 2,  0,  1],
            array![2, -1 ,-1]
        ]);
        
        
        let out = compute_progset(&data);
        
        assert_eq!(out, expected);
        
    }
    
    
    #[test]
    fn di_compute_progression_set_cohort_duplicates_t() {
        
        let data = array![
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, 1],
        ];
        
        let expected = HashSet::from([
            array![ 2,  0, -1],
            array![ 2,  0,  1],
        ]);
        
        
        let out = compute_progset(&data);
        
        assert_eq!(out, expected);
        
    }
    
    #[test]
    fn di_compute_incidence_matrix_t() {
        
        let data = array![
            [ 0,  1,  2,],
            [ 0,  1,  2,],
            [ 0,  1,  2,],
            [ 2,  0,  1,],
            [ 1,  2, -1,],
            //[ 0, -1, -1,],
            //[ 2, -1, -1,],
            [ 1,  0,  2,],
            [ 0,  1, -1,],
            [ 0,  2, -1,],
        ];
        
        let expected = array![
            [-1,  1,  0],
            [-1, -1,  1],
            [ 1,  0, -1],
            [-1,  1, -1],
            [ 0, -1,  1],
            [ 1, -1,  0],
            [-1,  0,  1]
        ];
        
        let ps = compute_progset(&data);
        let out = compute_incidence_matrix(&ps);
        
        // NOTE - I think this is working, but the order of the edges 
        // is arbitrary so the test fails (because of the hashset). 
        // TODO - figure out how to test this meaningfully. is there an ndarray
        // sort_rows method or something?
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