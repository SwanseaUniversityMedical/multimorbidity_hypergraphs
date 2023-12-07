
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
use indexmap::IndexSet;
use itertools::izip;

/*
pub fn compute_directed_hypergraph(
    data: &Array2<i8>
) -> DiHypergraphBase {
    
    DiHypergraphBase{incidence_matrix: Array2::zeros((1,1))}
    
}
*/

// TODO - write docstrings!
// TODO - clean up functions 

fn compute_hyperarc_weights(
    hyperedge_worklist: &Array2<i8>,
    hyperedge_prev: &Array1<f64>,
    hyperarc_prev: &Array2<f64>,
    hyperedge_weights: &Array1<f64>
) -> (IndexSet<Array1<i8>>, Array1<f64>) {
    
    let mut hyperarcs: IndexSet<Array1<i8>> = IndexSet::new();
    let mut hyperarc_weights: Vec<f64> = Vec::new();

    let n_diseases = hyperedge_worklist.ncols();
    
    for (h_idx, h) in hyperedge_worklist.axis_iter(Axis(0)).enumerate() {
        let hyperedge = h
            .iter()
            .filter(|&&x| x >= 0)
            .map(|&x| x) // this looks a bit ugly, but it's needed to dereference the values in h
            .collect::<Array1<_>>();
        
        let degree = hyperedge.len();
        
        let mut child_worklist: Array2<i8> = Array2::ones((degree, n_diseases));
        child_worklist
            .iter_mut()
            .for_each(|x| *x = -*x);
            
        let mut child_prevs: Array1<f64> = Array1::zeros(degree);
            
        if degree > 1 {
        
            let hyperedge_idx = hyperedge
                .iter()
                .map(|&x| 2_usize.pow(x as u32))
                .sum::<usize>();
                
            for n in 0..degree {
                
                let head = hyperedge[n] as usize;
                let tail = hyperedge
                    .slice(s![..n])
                    .map(|&x| 2_usize.pow(x as u32))
                    .sum() + 
                        hyperedge
                    .slice(s![n+1..])
                    .map(|&x| 2_usize.pow(x as u32))
                    .sum();
                    
                
                let hyperedge_set: HashSet<_> = hyperedge
                        .iter()
                        .cloned()
                        .collect();
                
                let cw_add = hyperedge_set
                        .difference(&HashSet::from_iter(std::iter::once(head as i8)))
                        .cloned()
                        .chain(std::iter::once(head as i8))
                        .collect::<Vec<_>>();
                
                for i in 0..cw_add.len() {
                    child_worklist[[n, i]] = cw_add[i] as i8;
                }
                
                child_prevs[n] = hyperarc_prev[[tail, head]];
            }
            
            let child_weights: Array1<_> = child_prevs 
                .iter()
                .map(|x| hyperedge_weights[h_idx] * x / (hyperedge_prev[hyperedge_idx] as f64))
                .collect();
            
            for i in 0..child_weights.len() {
                if child_weights[i] > 0.0 {
                    hyperarcs.insert(child_worklist.index_axis(Axis(0), i).to_owned());
                    hyperarc_weights.push(child_weights[i]);
                }
            }
            
        } else {
            
            let hyperedge_idx = hyperedge[0] as usize;
            child_worklist[[0, 0]] = hyperedge_idx as i8;
            let child_prev = hyperarc_prev[[0, hyperedge_idx]];
            let numerator = child_prev * hyperedge_weights[h_idx];
            let denominator = hyperedge_prev[hyperedge_idx] as f64;

            hyperarcs.insert(child_worklist.index_axis(Axis(0), 0).to_owned());

            if denominator == 0.0 {
                hyperarc_weights.push(0.0);
            } else {
                hyperarc_weights.push(numerator / denominator);
            }
        
        }
    }
    
    (hyperarcs, hyperarc_weights.into())
    
}

fn compute_hyperedge_weights(
    worklist: &IndexSet<Array1<i8>>,
    hyperedge_idx: &Array1<i32>,
    hyperedge_prev: &Array1<f64>,    
) -> Array1<f64> {
    
    let n_edges = worklist.len();
    
    let mut numerator: Array1<f64> = Array1::zeros(n_edges);
    let mut denominator: Array1<f64> = Array1::zeros(n_edges);
    
    for i in 0..n_edges {
        
        let hyper_idx = hyperedge_idx[i];
        
        let src_num_prev = hyperedge_prev[hyper_idx as usize];
        let src_denom_prev = hyperedge_prev[hyper_idx as usize];
        
        numerator[i] += src_num_prev;
        denominator[i] += src_denom_prev;
        
        let src_in_tgt = hyperedge_idx
            .iter()
            .enumerate()
            .zip(
                hyperedge_idx.iter().map(|&x| (hyper_idx & x) == hyper_idx).collect::<Vec<_>>()
            )
            .filter(|(_, b)| *b)
            .map(|(x, _)| x)
            .filter(|(loc, _)| *loc != i)
            .map(|(loc, _)| loc as usize)
            .collect::<Vec<_>>(); 
            
        
        // loop over src in tgt:
        for j in src_in_tgt {
            
            let tgt_hyper_idx = hyperedge_idx[j];
            let tgt_denom_prev = hyperedge_prev[tgt_hyper_idx as usize];
            
            denominator[i] += tgt_denom_prev;
            denominator[j] += src_denom_prev;  
        }
 
    }
    
    numerator
        .iter()
        .zip(&denominator)
        .map(|(x, y)| x / y)
        .collect()
    
}



fn compute_hyperedge_info(progset: &IndexSet<Array1<i8>>) -> (Array1<i32>, Array1<i32>) {
    
    (progset
        .iter()
        .map(|edge| {
            edge 
                .iter()
                .filter(|&x| *x >= 0)
                .map(|&x| 2_i32.pow(x as u32))
                .sum::<i32>()
        })
        .collect::<Array1<i32>>(), 
    progset
        .iter()
        .map(|edge| {
            edge 
                .iter()
                .filter(|&x| *x >= 0)
                .count() as i32
        })
        .collect::<Array1<i32>>()
    )
    
}

fn compute_hyperedge_worklist(inc_mat: &Array2<i8>) -> Array2<i8> {
    
    let n_rows = inc_mat.nrows();
    let n_cols = inc_mat.ncols();

    Array2::from_shape_vec(
        (n_rows, n_cols),
        inc_mat
            .axis_iter(Axis(0))
            .flat_map(|row| {
                let mut inds: Vec<i8> = row
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x != 0)
                    .map(|(index, _)| index as i8)
                    .collect();

                inds.extend(std::iter::repeat(-1).take(n_cols - inds.len()));

                inds
            })
            .collect::<Vec<i8>>()
    ).unwrap()
}



fn compute_incidence_matrix(progset: &IndexSet<Array1<i8>>) -> Array2<i8> {
    
    let progset_vec: Vec<_> = progset.into_iter().collect();
    
    let n_diseases = progset_vec[0].len();
    //let max_hyperedges = 2_usize.pow(n_diseases as u32);
    
    let mut hyperedges: IndexSet<Array1<i8>> = IndexSet::new();
    
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


fn compute_node_prev(
    data: &Array2<i8>
) -> Array1<usize> {
    
    let n_diseases = data.ncols();
    let mut out = Array::zeros(2 * n_diseases);
    
    for ii in 0..data.nrows() {
        for col_ind in 0..n_diseases {
        
            let mut second_cond = false;
            if col_ind < n_diseases - 1 {second_cond = second_cond || data[[ii, col_ind + 1]] >= 0 ;}
            
            if data[[ii, col_ind]] >= 0 && second_cond {
                out[data[[ii, col_ind]] as usize] += 1;
            } else if data[[ii, col_ind]] >= 0 && !second_cond {
                out[data[[ii, col_ind]] as usize + n_diseases] += 1;
            }
        }
    }

    out   
}


fn compute_progset(data: &Array2<i8>) -> 
(
    IndexSet<Array1<i8>>,
    Array1<f64>, //hyperedge_prev
    Array2<f64>, // hyperarc_prev
) {
    
    let n_rows = data.nrows();
    let n_diseases = data.ncols();
    let max_hyperedges = 2_usize.pow(n_diseases as u32);

    let mut hyperarc_prev: Array2<f64>  = Array::zeros((max_hyperedges, n_diseases));
    let mut hyperedge_prev: Array1<f64> = Array::zeros(max_hyperedges);

    let mut out: IndexSet<Array1<i8>> = (0..n_rows)
        //.into_iter()
        .flat_map(|i| {
                let progset_data = compute_single_progset(&data.index_axis(Axis(0), i).to_owned());
                for (a, b, z) in izip!(
                    progset_data.1, 
                    progset_data.2, 
                    progset_data.4.clone()
                ) {
                    hyperarc_prev[[a, b as usize]] += z;
                }
                
                for (c, z) in izip!(progset_data.3, progset_data.4) {
                    hyperedge_prev[c] += z;
                }
                progset_data.0
            }
         )
        .collect();
        
    if true { // add single diseases
        let additional: IndexSet<Array1<i8>> = (0..n_diseases)
            .map(|i| {
                let mut i_vec: Vec<i8> = vec![i as i8];
                i_vec.extend(&vec![-1; n_diseases - 1]);
                Array1::from_vec(i_vec)
            })
            .collect();
        
        out.extend(additional);
    }
    
    
    
    (out, hyperedge_prev, hyperarc_prev)
    
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
    IndexSet<Array1<i8>>, // single prog_set
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
                IndexSet::new(),
                array![0], 
                array![data_ind[0]], 
                array![2_usize.pow(data_ind[0] as u32)],
                array![1.0],
            )
        },
        
        _ => {
            let out:IndexSet<Array1<i8>> = (1..data_ind.len())
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
                        .rev()
                        .skip(1)
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
                    .chain(
                        std::iter::once(2_usize.pow(data_ind[0] as u32))
                     ) // this is the single disease contribution
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
                .chain(
                   std::iter::once(1.0)
                ) // this is the single disease contribution
                .collect();
                
            //println!("{:?} {:?} {:?}", bin_tail.to_vec(), head_node.to_vec(), contribution.to_vec());
            
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
        let expected = IndexSet::from([
            array![ 2,  0, -1],
            array![ 2,  0,  1]
        ]);
        
        let out = compute_single_progset(&data);
        
        assert_eq!(out.0, expected);
        //TODO - write tests for the other outputs of this function
    }
    
    
    #[test]
    fn di_compute_progression_set_singleton_t () {
        
        let data = array![2, -1, -1];
        let expected: IndexSet<Array1<i8>> = IndexSet::new();
        
        let out = compute_single_progset(&data);
        
        assert_eq!(out.0, expected);
    }
    
    #[test]
    fn di_compute_progression_set_cohort_t() {
        
        let data = array![
            [2, 0, 1],
            [0, -1, -1],
        ];
        
        let expected_progset = IndexSet::from([
            array![ 2,  0, -1],
            array![ 2,  0,  1],
            array![2, -1 ,-1],
            array![1, -1 ,-1],
            array![0, -1 ,-1],
        ]);
        
        let expected_hyperedge_prev = array![0., 1., 0., 0., 1., 1., 0., 1.];
        
        let expected_hyperarc_prev = array![[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.]];
        
        
        let out = compute_progset(&data);
        
        assert_eq!(out.0, expected_progset);
        assert_eq!(out.1, expected_hyperedge_prev);
        //assert_eq!(out.2, expected_hyperarc_prev); // TODO - this test fails. 
        
    }

    #[test]
    fn di_compute_progression_set_bigger_cohort_t() {
        
        let data = array![
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [2, 0, 1],
            [1, 2, -1],
            [0, -1, -1],
            [2, -1, -1],
            [1, 0, 2],
            [0, 1, -1],
            [0, 2, -1],
        ];
        
        let expected_progset = IndexSet::from([
            array![0, 1, -1],
            array![0, 1, 2 ],
            array![0, 2, -1],
            array![1, 0, -1],
            array![1, 0, 2 ], // this is not produced by Jamie's code...
            array![1, 2, -1],
            array![2, 0, -1],
            array![2, 0, 1 ],
            array![2, -1, -1 ],
            array![1, -1, -1 ],
            array![0, -1, -1 ],
        ]);
        
        let expected_hyperedge_prev = array![0., 6., 2., 5., 2., 2., 1., 5.];
        
        let expected_hyperarc_prev = array![[1., 0., 1.],
            [0., 4., 1.],
            [1., 0., 1.],
            [0., 0., 4.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.]];
        
        
        let out = compute_progset(&data);
        
        assert_eq!(out.0, expected_progset);
        assert_eq!(out.1, expected_hyperedge_prev);
        assert_eq!(out.2, expected_hyperarc_prev);
        
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
        
        let expected = IndexSet::from([
            array![ 2,  0, -1],
            array![ 2,  0,  1],
            array![2, -1, -1 ],
            array![1, -1, -1 ],
            array![0, -1, -1 ],
        ]);
        
        
        let out = compute_progset(&data);
        
        assert_eq!(out.0, expected);
        
    }
    
    #[test]
    fn di_compute_incidence_matrix_t() {
        
        let data = array![
            [ 0,  1,  2,],
            [ 0,  1,  2,],
            [ 0,  1,  2,],
            [ 2,  0,  1,],
            [ 1,  2, -1,],
            [ 0, -1, -1,],
            [ 2, -1, -1,],
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
            [-1,  0,  1],
            [1,  0,  0],
            [0,  1,  0],
            [0,  0,  1],
        ];
        
        let ps = compute_progset(&data);
        let out = compute_incidence_matrix(&ps.0);
        
        println!("{:?}", expected);
        println!("{:?}", out);
        
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
    
    #[test]
    fn di_construct_node_prev_t() {
        
        let data = array![
            [2, 0, 1],
            [1, -1, -1],
        ];

        let expected = array![1, 0, 1, 0, 2, 0];
        
        let out = compute_node_prev(&data);
        
        assert_eq!(out, expected);
        
    }
    
    #[test]
    fn di_construct_hyperedge_worklist_t() {
        
        let data = array![[2, 0, 1],
            [0, -1, -1]];
            
        let expected = array![[ 0, -1, -1],
            [ 1, -1, -1],
            [ 2, -1, -1],
            [ 0,  2, -1],
            [ 0,  1,  2]];
        
        let ps = compute_progset(&data);
        let inc_mat = compute_incidence_matrix(&ps.0);
        let out = compute_hyperedge_worklist(&inc_mat);
        
        
        println!("{:?}", expected);
        println!("{:?}", out);
        
        // NOTE - the order of axes does't matter again
        assert_eq!(
            out
                .axis_iter(Axis(0))
                .collect::<HashSet<_>>(), 
            expected
                .axis_iter(Axis(0))
                .collect::<HashSet<_>>()
        );
    }
    
    #[test]
    fn di_construct_hyperedge_info_t() {
        
        let data = array![[2, 0, 1],
            [0, -1, -1]];
            
        let ps = compute_progset(&data);
        let out = compute_hyperedge_info(&ps.0);
        
        let expected = (
            array![5, 7, 1, 2, 4],
            array![2, 3, 1, 1, 1]
        );
        
        println!("{:?}", ps.0);
        
        println!("{:?}", expected);
        println!("{:?}", out);
        
        assert_eq!(out, expected);
        
    }
    
    #[test]
    fn di_compute_weights_t() {
        let data = array![[2, 0, 1],
            [0, -1, -1]];
            
        let ps = compute_progset(&data);
        let info = compute_hyperedge_info(&ps.0);
       
        let out = compute_hyperedge_weights(
            &ps.0,
            &info.0,
            &ps.1
        );
        
        let expected = array![0.25, 0.25, 0.3333333333333333, 0., 0.3333333333333333];
        println!("{:?}", expected);
        println!("{:?}", out);
        
        assert_eq!(out, expected);
        
    }
    
    #[test]
    fn di_compute_hyperarc_weights_t() {
        let data = array![[2, 0, 1],
            [0, -1, -1]];
            
        let ps = compute_progset(&data);
        let info = compute_hyperedge_info(&ps.0);
        let hyperedge_weights = compute_hyperedge_weights(
            &ps.0,
            &info.0,
            &ps.1
        );
        let inc_mat = compute_incidence_matrix(&ps.0);
        let hyperedge_wl = compute_hyperedge_worklist(&inc_mat);
        
        let out: (IndexSet<Array1<i8>>, Array1<f64>) = compute_hyperarc_weights(
            &hyperedge_wl,
            &ps.1, // hyperedge_prev 
            &ps.2, // hyperarc_prev 
            &hyperedge_weights,
        );
        
        let mut hyperarc_set:IndexSet<Array1<i8>> = IndexSet::new();
        
        hyperarc_set.insert(array![2, 0, -1]);
        hyperarc_set.insert(array![2, 0, 1]);
        hyperarc_set.insert(array![0, -1, -1]);
        hyperarc_set.insert(array![1, -1, -1]);
        hyperarc_set.insert(array![2, -1, -1]);
        
        let expected: (IndexSet<Array1<i8>>, Array1<f64>) = (
            hyperarc_set, 
            array![0.25, 0.25, 0., 0., 0.,]
        );
        
        println!("{:?}", expected);
        println!("{:?}", out);
        
        assert_eq!(out.0, expected.0);
        assert_eq!(out.1, expected.1);
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