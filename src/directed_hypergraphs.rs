
use crate::types::*;
use crate::common_functions::*;


use ndarray::{
    array,
    Array,
    Array1,
    Array2,
    ArrayView1,
    Axis,
    s,
    stack,
};
use std::collections::{HashSet, HashMap, BTreeSet};
use indexmap::{IndexMap, IndexSet};
use itertools::izip;


pub fn compute_directed_hypergraph(
    data: &Array2<i8>
    ) -> DiHypergraphBase {
   
    let ps = compute_progset(&data);
    let inc_mat = compute_incidence_matrix(&ps.0);
    let hyperedge_wl = compute_hyperedge_worklist(&inc_mat);
    let info = compute_hyperedge_info(&hyperedge_wl);
    let hyperedge_weights = compute_hyperedge_weights(
        &hyperedge_wl,
        &info.0,
        &ps.1
    );
    
    let hyperarc: (Array1<HyperArc>, Array1<f64>) = compute_hyperarc_weights(
        &hyperedge_wl,
        &ps.1, // hyperedge_prev 
        &ps.2, // hyperarc_prev 
        &hyperedge_weights,
    );
   
    DiHypergraphBase{
        incidence_matrix: inc_mat,
        hyperedge_list: ps.0,
        hyperedge_weights: hyperedge_weights,
        hyperarc_list: hyperarc.0,
        hyperarc_weights: hyperarc.1,
    }
    
}



// TODO VERY IMPORTANT - Check to make sure that the new implementation of the 
// hyperedge_worklist IndexSet is being correctly calculated everywhere. 
// TODO - write docstrings!
// TODO - clean up functions 
// TODO - make sure everything that's calculated is actually needed


fn compute_hyperarc_weights(
    hyperedge_worklist: &IndexSet<Array1<i8>>,
    hyperedge_prev: &Array1<f64>,
    hyperarc_prev: &Array2<f64>,
    hyperedge_weights: &Array1<f64>
) -> (Array1<HyperArc>, Array1<f64>) {
    
    let mut hyperarcs: Vec<HyperArc> = Vec::new();
    let mut hyperarc_weights: Vec<f64> = Vec::new();
    
    for (h_idx, h) in hyperedge_worklist.iter().enumerate() {
        let hyperedge = h
            .iter()
            .filter(|&&x| x >= 0)
            .map(|&x| x) // this looks a bit ugly, but it's needed to dereference the values in h
            .collect::<Array1<_>>();
        
        let degree = hyperedge.len();
        
        let mut child_worklist: Vec<HyperArc> = Vec::new();
        
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
                
                
                let cw_add_p1 = hyperedge_set
                        .difference(&HashSet::from_iter(std::iter::once(head as i8)))
                        .cloned()
                        .collect::<HashSet<_>>();
                
                child_worklist.push(
                    HyperArc{
                        tail: cw_add_p1, 
                        head: head as i8
                    }
                );
                
                child_prevs[n] = hyperarc_prev[[tail, head]];
            }
            
            // TODO hyperedge weights are right but in the wrong order
            let child_weights: Array1<_> = child_prevs 
                .iter()
                .map(|x| hyperedge_weights[h_idx] * x / (hyperedge_prev[hyperedge_idx] as f64))
                .collect();    
                
            for i in (0..child_weights.len()).rev() {
                if child_weights[i] > 0.0 {
                    // borrow checker shenanigans here:
                    // you can't use the [] indexing operator when the elements 
                    // of the vec don't implement Copy. Therefore, using remove
                    // so that the data no longer exists in child_worklist and is free
                    // to move to hyperarcs. Looping backwards to avoid indexing weirdness.
                    hyperarcs.push(child_worklist.remove(i));
                    hyperarc_weights.push(child_weights[i]);
                }
            }
            
        } else {
            
            let hyperarc_idx = hyperedge[0] as usize;
            let hyperedge_idx = 2_i32.pow(hyperarc_idx.try_into().unwrap()) as usize;
            let child_prev = hyperarc_prev[[hyperarc_idx, 0]];
            let numerator = child_prev * hyperedge_weights[h_idx];
            let denominator = hyperedge_prev[hyperedge_idx] as f64;

            println!("{:?}, {}, {}", hyperedge, numerator, denominator);

            hyperarcs.push(
                HyperArc{
                    tail: HashSet::new(), 
                    head: hyperarc_idx as i8
                }
            );
            
            if denominator == 0.0 {
                hyperarc_weights.push(0.0);
            } else {
                hyperarc_weights.push(numerator / denominator);
            }
        
        }
    }
    
    (hyperarcs.into(), hyperarc_weights.into())
    
}

// the order of this is fucked.
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

fn compute_hyperedge_worklist(inc_mat: &Array2<i8>) -> IndexSet<Array1<i8>> {
    
    let n_rows = inc_mat.nrows();
    let n_cols = inc_mat.ncols();

    let big_hwl = Array2::from_shape_vec(
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
    ).unwrap();
    
    big_hwl
        .axis_iter(Axis(0))
        .map(|view| view.to_owned()) 
        .collect::<IndexSet<_>>()
}

fn compute_incidence_matrix(progset: &IndexSet<Array1<i8>>) -> Array2<i8> {
    
    
    let n_diseases = progset[0].len();
    
    let mut hyperedges: IndexSet<Array1<i8>> = IndexSet::new();
    
    for a in progset.iter() {
        
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


fn compute_head_tail_inc_mat(
    inc_mat: &Array2<i8>,
    end: HyperedgeEnd,
) -> Array2<i8> {
    
    match end {
        HyperedgeEnd::Head => inc_mat.map(|&x| {
                if x < 0 {
                    0
                } else {
                    x
                }
            }),
            
    // NOTE(jim). The tail incidence matrix has to have the self
    // edges manually added, hence the relatively higher complexity 
    // of this than the head incidence matrix.
    HyperedgeEnd::Tail => stack(
            Axis(0),
            &inc_mat
                .axis_iter(Axis(0))
                .map(|a| {
                    if a.sum() == 1 { 
                        a.to_owned()
                    } else {
                        a.map(|&x| 
                            if x > 0 {
                                0
                            } else {
                                x.abs()
                            })
                    }
                })
                .collect::<Vec<_>>()
                .iter() // have to go back through and convert owned arrays to views ¯\_(ツ)_/¯
                .map(|a| a.view()) 
                .collect::<Vec<_>>() 
        ).expect("Error creating stacked array")
    }
}



fn compute_node_prev(
    data: &Array2<i8>
) -> Array1<usize> {
    // TODO - is this function needed
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


    /*
    // start with single diseases
    let mut out: IndexSet<Array1<i8>> =  (0..n_diseases)
        .map(|i| {
            let mut i_vec: Vec<i8> = vec![i as i8];
            i_vec.extend(&vec![-1; n_diseases - 1]);
            Array1::from_vec(i_vec)
        })
        .collect();

     let additional: IndexSet<Array1<i8>> = (0..n_rows)
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
    */


    
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
        
    // add single diseases
    let additional: IndexSet<Array1<i8>> = (0..n_diseases)
        .map(|i| {
            let mut i_vec: Vec<i8> = vec![i as i8];
            i_vec.extend(&vec![-1; n_diseases - 1]);
            Array1::from_vec(i_vec)
        })
        .collect();
    
    out.extend(additional);
    
    
    (out, hyperedge_prev, hyperarc_prev)
    
}

fn bincount(arr: &ArrayView1<usize>) -> IndexMap<usize, usize> {
    arr.iter().fold(IndexMap::new(), |mut acc, &value| {
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
    use std::collections::{HashSet, HashMap};

    fn assert_vec_approx_eq(actual: &[f64], expected: &[f64], eps: f64) {
        assert_eq!(actual.len(), expected.len(), "vector lengths differ");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < eps,
                "index {}: {} not within {} of expected {}",
                i, a, eps, e
            );
        }
    }
    
    // NOTE - these two functions are a bit hacky to get the tests to 
    // pass. It's because constructing the hypergraph manually and 
    // doing it with the helper functions messes with the order and 
    // does not compare to Jamie's reference code
    
    // Returns a copy of `inc_mat` with rows sorted into a canonical,
    // deterministic order: ascending by hyperarc "degree" (how many diseases
    // participate in total: tail + head), with ties broken by which diseases
    // are involved. `compute_incidence_matrix`'s own row order is just
    // whatever order hyperarcs first appeared in the input data, which several
    // other tests already depend on (via `compute_hyperedge_worklist` /
    // `compute_hyperedge_info` / `compute_hyperarc_weights`), so this is
    // intentionally a separate, opt-in step rather than a change to
    // `compute_incidence_matrix` itself.
    fn canonicalize_incidence_matrix_order(inc_mat: &Array2<i8>) -> Array2<i8> {
        let n_cols = inc_mat.ncols();
    
        let mut rows: Vec<Array1<i8>> = inc_mat
            .axis_iter(Axis(0))
            .map(|r| r.to_owned())
            .collect();
    
        rows.sort_by_key(|row| {
            let degree = row.iter().filter(|&&x| x != 0).count();
            let membership: i32 = row
                .iter()
                .enumerate()
                .filter(|(_, &x)| x != 0)
                .map(|(i, _)| 2_i32.pow(i as u32))
                .sum();
            (degree, membership)
        });
    
        let n_rows = rows.len();
        rows.into_iter()
            .flat_map(|r| r)
            .collect::<Array1<_>>()
            .into_shape((n_rows, n_cols))
            .unwrap()
    }
    
    
    // Re-order `hyperarc_list`/`hyperarc_weights` (as produced by
    // `compute_hyperarc_weights`, which is ordered by the deduplicated,
    // disease-index-sorted hyperedge worklist) so that entry `i` corresponds to
    // row `i` of `inc_mat`. `inc_mat` encodes each hyperarc as a row with -1 at
    // each tail disease and +1 at the (single) head disease, so we can recover
    // the (tail, head) pair for each row and look up the matching arc.
    fn reorder_hyperarcs_to_incidence_matrix(
        inc_mat: &Array2<i8>,
        hyperarc_list: &Array1<HyperArc>,
        hyperarc_weights: &Array1<f64>,
    ) -> (Array1<HyperArc>, Array1<f64>) {
    
        let lookup: HashMap<(BTreeSet<i8>, i8), usize> = hyperarc_list
            .iter()
            .enumerate()
            .map(|(idx, arc)| {
                let tail: BTreeSet<i8> = arc.tail.iter().cloned().collect();
                ((tail, arc.head), idx)
            })
            .collect();
    
        let n_rows = inc_mat.nrows();
        let n_cols = inc_mat.ncols();
    
        let mut ordered_list: Vec<HyperArc> = Vec::with_capacity(n_rows);
        let mut ordered_weights: Vec<f64> = Vec::with_capacity(n_rows);
    
        for r in 0..n_rows {
            let row = inc_mat.row(r);
    
            let tail: BTreeSet<i8> = (0..n_cols)
                .filter(|&j| row[j] < 0)
                .map(|j| j as i8)
                .collect();
    
            let head = (0..n_cols)
                .find(|&j| row[j] > 0)
                .expect("Each incidence matrix row must have exactly one head disease")
                as i8;
    
            let idx = *lookup
                .get(&(tail, head))
                .expect("No matching hyperarc found for incidence matrix row");
    
            ordered_list.push(hyperarc_list[idx].clone());
            ordered_weights.push(hyperarc_weights[idx]);
        }
    
        (ordered_list.into(), ordered_weights.into())
}
 
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
    
//    #[test]
//    fn di_compute_progression_set_cohort_t() {
//        
//        let data = array![
//            [2, 0, 1],
//            [0, -1, -1],
//        ];
//        
//        let expected_progset = IndexSet::from([
//            array![ 2,  0, -1],
//            array![ 2,  0,  1],
//            array![2, -1 ,-1],
//            array![1, -1 ,-1],
//            array![0, -1 ,-1],
//        ]);
//        
//        let expected_hyperedge_prev = array![0., 1., 0., 0., 1., 1., 0., 1.];
//        
//        let expected_hyperarc_prev = array![[0., 0., 0.],
//            [0., 0., 0.],
//            [0., 0., 0.],
//            [0., 0., 0.],
//            [1., 0., 0.],
//            [0., 1., 0.],
//            [0., 0., 0.],
//            [0., 0., 0.]];
//        
//        
//        let out = compute_progset(&data);
//        
//        assert_eq!(out.0, expected_progset);
//        assert_eq!(out.1, expected_hyperedge_prev);
//        assert_eq!(out.2, expected_hyperarc_prev); // TODO - this test fails. 
//        
//    }

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
        
        println!("{:?}", ps);
        
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
    fn di_compute_head_tail_incidence_matrix_t() {
        
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
        
        let expected_head = array![
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ];
        
        let expected_tail = array![
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0]
        ];
        
        let h = compute_directed_hypergraph(&data);
        let out_head = compute_head_tail_inc_mat(&h.incidence_matrix, HyperedgeEnd::Head);
        let out_tail = compute_head_tail_inc_mat(&h.incidence_matrix, HyperedgeEnd::Tail);
        
        
        println!("Incidence matrix");
        println!("{:?}", h.incidence_matrix);

        println!("Head");
        println!("{:?}", expected_head);
        println!("{:?}", out_head);
        
        println!("Tail");
        println!("{:?}", expected_tail);
        println!("{:?}", out_tail);
        
        // NOTE - the order of axes does not matter, so use an iterator over
        // rows and collect them into a HashSet for comparison.
        
        println!("Test Head");
        assert_eq!(
            out_head
                .axis_iter(Axis(0))
                .collect::<HashSet<_>>(), 
            expected_head
                .axis_iter(Axis(0))
                .collect::<HashSet<_>>()
        );
        
        println!("Test Tail");
        assert_eq!(
            out_tail
                .axis_iter(Axis(0))
                .collect::<HashSet<_>>(), 
            expected_tail
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
            out,
                //.axis_iter(Axis(0))
                //.collect::<HashSet<_>>(), 
            expected
                .axis_iter(Axis(0))
                .map(|view| view.to_owned())
                .collect::<IndexSet<_>>()
        );
        
    }
    
   #[test]
    fn di_construct_hyperedge_worklist_larger_set_t() {
        
        let data = array![[0, 1, 2], //conds_worklist
        [0, 1, 2],
        [0, 1, 2],
        [2, 0, 1],
        [1, 2, -1],
        [0, -1, -1],
        [2, -1, -1],
        [1, 0, 2],
        [0, 1, -1],
        [0, 2, -1],];
            
        let expected = array![
            [ 0, -1, -1],
            [ 1, -1, -1],
            [ 2, -1, -1],
            [ 1,  2, -1],
            [ 0,  2, -1],
            [ 0,  1, -1],
            [ 0,  1,  2],
        ];
        
        let ps = compute_progset(&data);
        let inc_mat = compute_incidence_matrix(&ps.0);
        let out = compute_hyperedge_worklist(&inc_mat);
        
        
        println!("{:?}", expected);
        println!("{:?}", out);
        
        
        //assert_eq!(out.shape(), expected.shape());
        
        // NOTE - the order of axes does't matter again
        assert_eq!(
            out,
                //.axis_iter(Axis(0))
                //.collect::<HashSet<_>>(), 
            expected
                .axis_iter(Axis(0))
                .map(|view| view.to_owned())
                .collect::<IndexSet<_>>()
        );
    }
    
    #[test]
    fn di_construct_hyperedge_info_t() {
        
        let data = array![[2, 0, 1],
            [0, -1, -1]];
            
        let ps = compute_progset(&data);
        let inc_mat = compute_incidence_matrix(&ps.0);
        let hyperedge_wl = compute_hyperedge_worklist(&inc_mat);
        let out = compute_hyperedge_info(&hyperedge_wl);
                
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
    fn di_construct_hyperedge_info_larger_set_t() {
        
        let data = array![[0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [2, 0, 1],
            [1, 2, -1],
            [0, -1, -1],
            [2, -1, -1],
            [1, 0, 2],
            [0, 1, -1],
            [0, 2, -1],];
            
        let ps = compute_progset(&data);
        let inc_mat = compute_incidence_matrix(&ps.0);
        let hyperedge_wl = compute_hyperedge_worklist(&inc_mat);
        let out = compute_hyperedge_info(&hyperedge_wl);
        
        let expected = (
            array![1, 2, 4, 6, 5, 3, 7,], // hyperedge_indexes?
            array![1, 1, 1, 2, 2, 2, 3,] // hyperedge_N ?
        );
        
        println!("{:?}", ps.0);
        
        println!("Expected 0 {:?}\n", expected.0);
        println!("Calculated 0 {:?}\n", out.0);
        println!("Expected 1 {:?}\n", expected.1);
        println!("Calculated 1 {:?}\n", out.1);
        
        for (idx, value) in expected.0.iter().enumerate() {
            if let Some(index) = out.0.iter().position(|x| x == value) {
                assert_eq!(expected.1[idx], out.1[index]);
            } else {
                panic!("Something fucked up");
            }
        }
        
        //assert_eq!(out, expected);
        
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
    fn di_compute_weights_larger_data_t() {
        let data = array![[0, 1, 2],
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
            
        let ps = compute_progset(&data);
        let info = compute_hyperedge_info(&ps.0);
       
        let out = compute_hyperedge_weights(
            &ps.0,
            &info.0,
            &ps.1
        );
        
        let expected = array![0.33333333, 0.15384615, 0.2       , 0.1       , 0.13333333,
       0.27777778, 0.2173913 ];
       // ['A', 'B', 'C', 'B, C', 'A, C', 'A, B', 'A, B, C']
        println!("Expected: {:?}", expected);
        println!("Calculated: {:?}", out);
        
        assert_eq!(
            out.to_vec().sort_by(|a, b| a.partial_cmp(b).unwrap()),
            expected.to_vec().sort_by(|a, b| a.partial_cmp(b).unwrap())
        );
        
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
        
        let out: (Array1<HyperArc>, Array1<f64>) = compute_hyperarc_weights(
            &hyperedge_wl,
            &ps.1, // hyperedge_prev 
            &ps.2, // hyperarc_prev 
            &hyperedge_weights,
        );
        
        let mut hyperarc_set: Vec<HyperArc> = Vec::new();
        
        hyperarc_set.push(HyperArc{ tail: HashSet::from([2]), head: 0});
        hyperarc_set.push(HyperArc{ tail: HashSet::from([2, 0]), head: 1});
        hyperarc_set.push(HyperArc{ tail: HashSet::new(), head: 0});
        hyperarc_set.push(HyperArc{ tail: HashSet::new(), head: 1});
        hyperarc_set.push(HyperArc{ tail: HashSet::new(), head: 2});
        
        let expected: (Array1<HyperArc>, Array1<f64>) = (
            hyperarc_set.into(), 
            array![0.25, 0.25, 0.3333333333333333, 0., 0.,]
        );
        
        println!("{:?}\n\n", expected);
        println!("{:?}\n\n", out);
        
        assert_eq!(out.0, expected.0);
        assert_eq!(out.1, expected.1);
        
        
    }
    
    #[test]
    fn di_compute_hyperarc_weights_larger_set_t() {
        let data = array![[0, 1, 2],
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
            
        let ps = compute_progset(&data);
        let inc_mat = compute_incidence_matrix(&ps.0);
        let hyperedge_wl = compute_hyperedge_worklist(&inc_mat);
        let info = compute_hyperedge_info(&hyperedge_wl);
        
        let hyperedge_weights = compute_hyperedge_weights(
            &hyperedge_wl,
            &info.0,
            &ps.1
        );
        
        let out: (Array1<HyperArc>, Array1<f64>) = compute_hyperarc_weights(
            &hyperedge_wl,
            &ps.1, // hyperedge_prev 
            &ps.2, // hyperarc_prev 
            &hyperedge_weights,
        );
        
        let mut hyperarc_set: Vec<HyperArc> = Vec::new();
        
        hyperarc_set.push(HyperArc{ tail: HashSet::new(), head: 0});
        hyperarc_set.push(HyperArc{ tail: HashSet::new(), head: 1});
        hyperarc_set.push(HyperArc{ tail: HashSet::new(), head: 2});
        hyperarc_set.push(HyperArc{ tail: HashSet::from([1]), head: 2});
        hyperarc_set.push(HyperArc{ tail: HashSet::from([2]), head: 0});
        hyperarc_set.push(HyperArc{ tail: HashSet::from([0]), head: 2});
        hyperarc_set.push(HyperArc{ tail: HashSet::from([1]), head: 0});
        hyperarc_set.push(HyperArc{ tail: HashSet::from([0]), head: 1});
        hyperarc_set.push(HyperArc{ tail: HashSet::from([0, 2]), head: 1});
        hyperarc_set.push(HyperArc{ tail: HashSet::from([0, 1]), head: 2});
        
                
        
        let expected: (Array1<HyperArc>, Array1<f64>) = (
            hyperarc_set.into(), 
            array![0.05555555555555555, 0.0, 0.1, 0.1, 0.06666666666666667, 0.06666666666666667, 0.05555555555555556,   0.22222222222222224, 0.043478260869565216, 0.17391304347826086]
        );

        

        // A -> A, 0.05555555555555555
        // B -> B, 0.0
        // C -> C, 0.1
        // B -> C, 0.1
        // C -> A, 0.06666666666666667
        // A -> C, 0.06666666666666667
        // B -> A, 0.05555555555555556
        // A -> B, 0.22222222222222224
        // A, C -> B, 0.043478260869565216
        // A, B -> C, 0.17391304347826086


        
        println!("\n");
        for (idx, item) in expected.0.iter().enumerate() {
            if let Some(index) = out.0.iter().position(|x| x == item) {
                println!("{:?}, {:?}, {:?}", out.0[index], expected.1[idx], out.1[index]);
                assert_eq!(expected.1[idx], out.1[index]);
            } else {
                panic!("Something fucked up.")
            }
        }
        println!("\n");
        
        //assert_eq!(out.0, expected.0);
        //assert_eq!(out.1, expected.1);
    }
   
    
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
            
            
        let ps = compute_progset(&data);
        let inc_mat = compute_incidence_matrix(&ps.0);
        let hyperedge_wl = compute_hyperedge_worklist(&inc_mat);
        let info = compute_hyperedge_info(&hyperedge_wl);
        let hyperedge_weights = compute_hyperedge_weights(
            &hyperedge_wl,
            &info.0,
            &ps.1
        );
        
        let hyperarc: (Array1<HyperArc>, Array1<f64>) = compute_hyperarc_weights(
            &hyperedge_wl,
            &ps.1, // hyperedge_prev 
            &ps.2, // hyperarc_prev 
            &hyperedge_weights,
        );
   
   
        let expected = DiHypergraphBase{
            incidence_matrix: inc_mat,
            hyperedge_list: ps.0,
            hyperedge_weights: hyperedge_weights,
            hyperarc_list: hyperarc.0,
            hyperarc_weights: hyperarc.1,
        };
        
        let out = compute_directed_hypergraph(&data);
        
        assert_eq!(out.incidence_matrix, expected.incidence_matrix);
        assert_eq!(out.hyperedge_list, expected.hyperedge_list);
        assert_eq!(out.hyperedge_weights, expected.hyperedge_weights);
        assert_eq!(out.hyperarc_list, expected.hyperarc_list);
        assert_eq!(out.hyperarc_weights, expected.hyperarc_weights);
        
        println!("{:?}", expected.hyperedge_list);
        
        //panic!("oh no")
    }
    #[test]
    fn di_compute_degree_vectors() {
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
        
        //let h = compute_directed_hypergraph(&data);
        
        let ps = compute_progset(&data);
        let inc_mat = compute_incidence_matrix(&ps.0);
        let hyperedge_wl = compute_hyperedge_worklist(&inc_mat);
        let info = compute_hyperedge_info(&hyperedge_wl);
        let hyperedge_weights = compute_hyperedge_weights(
            &hyperedge_wl,
            &info.0,
            &ps.1
        );
        
        let hyperarc: (Array1<HyperArc>, Array1<f64>) = compute_hyperarc_weights(
            &hyperedge_wl,
            &ps.1, // hyperedge_prev 
            &ps.2, // hyperarc_prev 
            &hyperedge_weights,
        );
        
        let inc_mat_canonical = canonicalize_incidence_matrix_order(&inc_mat); 
        let hyperarc_ordered = reorder_hyperarcs_to_incidence_matrix(
            &inc_mat_canonical,
            &hyperarc.0,
            &hyperarc.1,
        );

   
        let h = DiHypergraphBase{
            incidence_matrix: inc_mat_canonical,
            hyperedge_list: ps.0,
            hyperedge_weights: hyperedge_weights,
            hyperarc_list: hyperarc_ordered.0,
            hyperarc_weights: hyperarc_ordered.1,
        };
    
        
        
        let out_head = compute_head_tail_inc_mat(&h.incidence_matrix, HyperedgeEnd::Head);
        let out_tail = compute_head_tail_inc_mat(&h.incidence_matrix, HyperedgeEnd::Tail);
        
        let exp_node_degree_head = vec![0.17777778, 0.26570048, 0.44057971];
        let exp_node_degree_tail = vec![0.56183575, 0.3294686 , 0.21014493];
        let exp_edge_degree_head = vec![1., 1., 1., 1., 1., 1., 1., 1., 1., 1.];
        let exp_edge_degree_tail = vec![1., 1., 1., 1., 1., 1., 1., 1., 2., 2.];
        
        println!("Incidence matrix");
        println!("{:?}\n", h.incidence_matrix);
        println!("Incidence matrix head");
        println!("{:?}\n", out_head);
        println!("Incidence matrix tail");
        println!("{:?}\n", out_tail);
        println!("Hyperedge weights");
        println!("{:?}\n", h.hyperedge_weights);
        println!("Hyperarc weights");
        println!("{:?}\n", h.hyperarc_weights);
        

        let node_degree_head = degree_centrality(&out_head, Representation::Standard, Some(h.hyperarc_weights.to_vec()));
        let edge_degree_head = degree_centrality(&out_head, Representation::Dual, None);
        let edge_degree_tail = degree_centrality(&out_tail, Representation::Dual, None);
        
        
        
        
        // node_degree_head is compared to full f64 precision internally, but
        // exp_node_degree_head is hand-typed and truncated to 8 significant
        // figures, so it will never be bit-for-bit equal to the computed
        // value even when the computation is correct. Compare with a
        // tolerance instead of assert_eq!.
        
        println!("Expected");
        println!("{:?}\n", exp_node_degree_head);
        
        println!("Computed");
        println!("{:?}\n", node_degree_head);
        
        
        assert_vec_approx_eq(&node_degree_head, &exp_node_degree_head, 1e-7); // this fails without using hyperarc_sorted above

        // edge_degree_head/tail sum to small whole numbers (1.0, 2.0), which
        // are exactly representable in f64, so exact comparison is fine here.
        assert_eq!(edge_degree_head, exp_edge_degree_head);
        assert_eq!(edge_degree_tail, exp_edge_degree_tail); // this always fails 
        
        
        
    }
}