
//use crate::types::*;

use ndarray::{
    array,
    Array,
    Array1,
    Array2,
    ArrayView1,
    Axis,
    s,
};
use std::collections::{HashSet, HashMap};

/*
pub fn compute_directed_hypergraph(
    data: &Array2<i8>
) -> DiHypergraphBase {
    
    DiHypergraphBase{incidence_matrix: Array2::zeros((1,1))}
    
}
*/


fn compute_incidence_matrix(progset: &HashSet<Array1<i8>>) -> Array2<i8> {
    
    let progset_vec: Vec<_> = progset.into_iter().collect();
    
    let n_diseases = progset_vec[0].len();
    //let max_hyperedges = 2_usize.pow(n_diseases as u32);
    
    let mut hyperedges: HashSet<Array1<i8>> = HashSet::new();
    
    for a in progset_vec.iter() {
        
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

/*
fn compute_prev_matrices(
    progset: HashSet<Array1<i8>>
) -> (Array1<i8>, Array2<i8>) {
    

    
}
*/

fn compute_progset(data: &Array2<i8>) -> HashSet<Array1<i8>> {
    
    let n_rows = data.nrows();

    (0..n_rows)
        //.into_iter()
        .flat_map(|i| compute_single_progset(&data.index_axis(Axis(0), i).to_owned()).0 )
        .collect()
    
}

fn bincount(arr: &ArrayView1<usize>) -> HashMap<usize, usize> {
    arr.iter().fold(HashMap::new(), |mut acc, &value| {
        *acc.entry(value).or_insert(0) += 1;
        acc
    })
}

fn compute_single_progset(
    data_ind: &Array1<i8>
) -> (
    HashSet<Array1<i8>>, // single prog_set
    Array1<usize>, // bin_tail
    Array1<i8>, // head_node 
    Array1<usize>, // bin_headtail
    Array1<f64> // contribution 
) {
    
    // NOTE - we are assuming that there are no duplicates in the ordering
    // ie, this is the simplest possible progression. 
    
    let n_diseases = data_ind
        .iter()
        .map(|&x| x >= 0)
        .filter(|&x| x)
        .count();
    
    
    match n_diseases {
        // people that only have one disease have to be treated spearately
        1 => {
            (
                HashSet::from([data_ind.clone()]),
                array![0],
                array![data_ind[0]],
                array![2_usize.pow(data_ind[0] as u32)],
                array![1.0],
            )
        },
        
        _ => {
            let out:HashSet<Array1<i8>> = (1..data_ind.len())
                .filter(|&i| data_ind[i] >= 0)
                .map(|i| {
                    let mut i_vec = data_ind.slice(s![0..(i + 1)]).to_vec();
                    i_vec.extend(&vec![-1; data_ind.len() - 1 - i]);
                    Array1::from_vec(i_vec)
                })
            .collect();
            
            let bin_tail:Array1<_>  = out 
                .iter()
                .map(
                    |arr| arr
                        .iter()
                        .filter(|&x| x >= &0)
                        .fold(0, |acc, x| acc + 2_usize.pow(*x as u32))
                )
                .collect();
            
            let head_node: Array1<_> = out
                .iter()
                .map(
                    |arr| arr
                        .iter()
                        .enumerate()
                        .filter(|(_, &r)| r >= 0)
                        .max()
                        .map(|(index, _)| arr[index])
                        .unwrap()
                )
                .collect();
                
            let bin_headtail: Array1<_> = bin_tail 
                    .iter()
                    .zip(head_node.clone())
                    .map(|(x, y)| x + 2_usize.pow(y as u32))
                    .collect();
        
            let n_conds_prog: Array1<_> = out
                .iter()
                .map(
                    |x| x
                        .iter()
                        .filter(|&x| x >= &0)
                        .count()
                )
                .collect();
                
            let cond_cnt = bincount(&n_conds_prog.view());
            
            let contribution: Array1<f64> = n_conds_prog
                .iter()
                .map(|x| 1.0 / (cond_cnt[x] as f64))
                .collect();
            
            (    
                out, // single prog_set
                
                bin_tail, 
                
                head_node,
                
                bin_headtail,
                    
                contribution,
            )
        },
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
        
        assert_eq!(out.0, expected);
        
        //assert!(false);
    }
    
    
    #[test]
    fn di_compute_progression_set_singleton_t () {
        
        let data = array![2, -1, -1];
        let expected: HashSet<Array1<i8>> = HashSet::from([array![2, -1 ,-1]]);
        
        let out = compute_single_progset(&data);
        
        assert_eq!(out.0, expected);
        
        //assert!(false);
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
        
        // NOTE - the order of axes does not matter, so use an iterator over
        // rows and collect them into a HashSet for comparison.
        assert_eq!(
            out
                .axis_iter(Axis(0))
                .collect::<HashSet<_>>(), 
            expected
                .axis_iter(Axis(0))
                .collect::<HashSet<_>>()
        );
        
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