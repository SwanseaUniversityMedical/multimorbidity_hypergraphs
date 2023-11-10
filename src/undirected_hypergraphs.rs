
#[allow(dead_code)]
#[allow(unused_imports)]

extern crate intel_mkl_src;

use crate::types::*;

// TODO -clean up the datatypes
use ndarray::{
    Array,
    Array1,
    Array2,
    ArrayView1, 
    ArrayView2, 
    Axis,
    arr1
};
use sprs::{CsMat, TriMat};
use rand::Rng;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashSet;
use log::warn;
use std::cmp;

pub fn compute_hypergraph(data: &Array2<u8>) -> HypergraphBase {
    
    // Computes the hypergraph from an array of data.
    
    let edge_list = construct_edge_list(&data);
    let inc_mat = incidence_matrix(&edge_list);
    let node_list = (0..inc_mat.ncols()).collect::<Vec<_>>();
    
    let node_w = node_list
        .iter()
        .map(|x| overlap_coefficient(data.select(Axis(1), [*x].as_slice()).view()))
        .collect::<Vec<_>>();
    
    //println!("{} {} {}", &edge_list.len(), &node_list.len(), &edge_list.len()*&node_list.len());
    
    HypergraphBase {
        incidence_matrix: inc_mat, 
        edge_weights: compute_edge_weights(&data, &edge_list),
        node_weights: node_w,
        edge_list: edge_list, 
        node_list: node_list, 
    }
}

fn compute_edge_weights(data: &Array2<u8>, edge_list: &Vec<Vec<usize>>) -> Vec<f64> {
    
    // Loops over the edges to calculate the edge weights. Currently only the overlap
    // coefficient is supported.
    
    edge_list
        .into_par_iter()
        .map(|x| overlap_coefficient(data.select(Axis(1), x.as_slice()).view()))
        .collect::<Vec<_>>()
    
}

fn incidence_matrix(edge_list: &Vec<Vec<usize>>) -> Array2<u8> {
    
    // Generates the incidence matrix from the edge list
    
    let max_column = edge_list
        .iter()
        .flat_map(|edge| edge.iter())
        .copied()
        .max()
        .unwrap() + 1;

    Array::from_shape_fn(
        (edge_list.len(), max_column),
        |(ii, jj)| {
            edge_list[ii].contains(&jj).then(|| 1).unwrap_or(0)
        }
    )
}

fn overlap_coefficient(data: ArrayView2<u8>) -> f64 {
    
    // Calculates the overlap coefficient for an edge, or the prevalence for 
    // data on only one disease. 
    
    match data.ncols() {
        
        1 => {
            data
                .iter()
                .map(|x| *x as u32)
                .sum::<u32>() as f64 / 
            data.nrows() as f64
        }
    
        
        _ => {
            let denom = data
                .axis_iter(Axis(1))
                .map(|x| 
                    x
                        .iter()
                        .map(|y| *y as u32)
                        .sum::<u32>()
                )
                .min()
                .unwrap() as f64;
                
            if denom < 0.5 { 
            // NOTE(jim): this is an integer count cast to a f64, so if its less than 
            // 1.0 - eps its zero and the code should panic.
                panic!("overlap_coefficient: denominator is zero.");
            }
            
            data
                .axis_iter(Axis(0))
                .filter(|data_row| usize::from(data_row.sum()) == data_row.len())
                .count() as f64 / denom
            // NOTE(jim): .fold may be faster than .filter.count
        }
    
    }
       
}

fn reduced_powerset(
    data_row: ArrayView1<u8>,
    all_diseases: usize,
) -> HashSet<Vec<usize>> {

    // Returns the "reduced" powerset of a single set of diseases (ie the powerset
    // without the empty or singleton sets. 
    
    let n_diseases = data_row.iter().map(|x| (x > &0) as usize).sum::<usize>();
    let upper_const = cmp::min(all_diseases - 1, n_diseases);
    
    (2..=upper_const) 
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

fn construct_edge_list(data: &Array2<u8>) -> Vec<Vec<usize>> {
    
    // Constructs the full edge list from the original data array.
    
    let all_diseases = data.ncols();
    
    data
        .axis_iter(Axis(0))
        .map(|x| reduced_powerset(x, all_diseases))
        .flatten()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
    
}


fn adjacency_matrix_times_vector(
    incidence_matrix: &ArrayView2<f64>,
    weight: &[f64],
    vector: &[f64],
) -> Vec<f64> {

    // ndarray-linalg / sprs.
    if let [outer, inner] = &incidence_matrix.shape() {
        
        let w: CsMat<f64> = CsMat::new(
            (*outer, *outer),
            (0..*outer+1).collect::<Vec<_>>(),
            (0..*outer).collect::<Vec<_>>(),
            weight.to_vec()
        );
        
        let weighted_incidence = &w * incidence_matrix;
        
        let intermediate = weighted_incidence.dot(&arr1(vector));
        
        let term_1 = intermediate.dot(incidence_matrix);
        
        let mut subt = vec![0.0; vector.len()];
    
        for k in 0..*outer * *inner {
                subt[k % inner] += incidence_matrix[[k / inner, k % inner]] * weighted_incidence[[k / inner, k % inner]] * vector[k % inner];
        }
        
        let mut result = vec![0.0; vector.len()];

        for i in 0..vector.len() {
            result[i] = term_1[i] - subt[i];
        }
        
        return result;

    } else {
        panic!("The incidence matrix has the wrong shape.");
    }
}



fn evc_iteration(
    inc_mat: ArrayView2<f64>, 
    weight: &Vec<f64>, 
    eigenvector: Vec<f64>,
    tolerance: f64,
    iter_no: u32,
    max_iterations: u32,
) -> Vec<f64> {
    
    let mut eigenvector_new = adjacency_matrix_times_vector(
        &inc_mat,
        weight,
        &eigenvector,
    );
    
    let evnew_norm = eigenvector_new
        .iter()
        .map(|x| x.powf(2.0))
        .sum::<f64>()
        .sqrt();
    
    eigenvector_new = eigenvector_new
        .iter() 
        .map(|x| x / evnew_norm)
        .collect::<Vec<_>>();
    
    let err_estimate = eigenvector_new
        .iter()
        .zip(&eigenvector)
        .map(|(&x, &y)| (x - y).powf(2.0))
        .sum::<f64>()
        .sqrt();
  
    if (err_estimate < tolerance) | (iter_no > max_iterations) {
        eigenvector_new 
    } else {
        evc_iteration(
            inc_mat,
            weight,
            eigenvector_new,
            tolerance,
            iter_no + 1,
            max_iterations
        )
    }   
}


fn evc_iteration_sparse(
    adj_mat: &CsMat<f64>,
    eigenvector: &Array1<f64>,
    tolerance: f64,
    iter_no: u32,
    max_iterations: u32,
) -> Array1<f64> {
    
    // NOTE(jim): The power iteration method for the bipartite rep
    // is a terrible choice because the adjacency matrix is almost always
    // singular. That means the non-zero eigenvalues form real +/- pairs 
    // and the algorithm fails to converge. To get around that I am adding a 
    // small positive offset to all entries in the adjacency matrix, but 
    // that means I'm not strictly calculating the EVC of the adjacency matrix
    // any more, and the tests fail. Moreover, because the offset is small the 
    // convergence is really slow. On the bright side, the result is a pretty close 
    // approximation to the correct answer.
    
    // BiCGSTAB is currently being worked on by the sprs developers, so hopefully 
    // I will be able to make use of that soon...
    
    let offset = eigenvector.sum() * 0.01;    
    let mut eigenvector_new = adj_mat * eigenvector + offset;

    let evnew_norm = eigenvector_new
        .iter()
        .map(|x| x.powf(2.0))
        .sum::<f64>()
        .sqrt();
    
    eigenvector_new = eigenvector_new
        .iter() 
        .map(|x| x / evnew_norm)
        .collect::<Array1<_>>();    
    
    let err_estimate = eigenvector_new
        .iter()
        .zip(eigenvector)
        .map(|(&x, &y)| (x - y).powf(2.0))
        .sum::<f64>()
        .sqrt();

    

    if (err_estimate < tolerance) | (iter_no > max_iterations) {
        eigenvector_new 
    } else {
        evc_iteration_sparse(
            adj_mat,
            &eigenvector_new,
            tolerance,
            iter_no + 1,
            max_iterations,
        )
    }   
}


fn normalised_vector_init(len: usize) -> Vec<f64> {
    
    let mut rng = rand::thread_rng();
    
    let vector: Vec<f64> = (0..len)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();
    let norm = vector
        .iter()
        .fold(0., |sum, &num| sum + num.powf(2.0))
        .sqrt();
        
    vector.iter().map(|&b| b / norm).collect()
}

fn bipartite_eigenvector_centrality(
    incidence_matrix: &Array2<u8>, 
    edge_weights: &Vec<f64>,
    tolerance: f64,
    max_iterations: u32,
) -> Array1<f64> {
    
    warn!("WARNING: This currently calculates the eigenvector centrality of the adjacency matrix
    plus a small offset. This is because of limitations in the external libraries used. The 
    eigenvector centralities returned are not exactly correct, though the ordering of the 
    centralities should be correct. Do not rely on this if it's mission critical.");
    
    let m_size = incidence_matrix.shape();
    let n_edges = m_size[0]; let n_nodes = m_size[1];
    
    let total_elems: usize = n_edges + n_nodes;

    let weighted_inc = incidence_matrix
        .mapv(|x| f64::from(x))
        .t()
        .dot(&Array2::from_diag(&arr1(edge_weights)));

    let rows: Vec<_> = weighted_inc
        .indexed_iter()
        .filter(|&(_, &value)| value != 0.0)
        .map(|((row, _), _)| row)
        .collect();
    let cols: Vec<_> = weighted_inc
        .indexed_iter()
        .filter(|&(_, &value)| value != 0.0)
        .map(|((_, col), _)| col)
        .collect();
    let data: Vec<_> = weighted_inc
        .iter()
        .filter(|&&value| value != 0.0)
        .cloned()
        .collect();
    
    
    let adjacency_matrix: CsMat<_> = {
        let mut a = TriMat::new((total_elems, total_elems));
        
        for (i, (x, y)) in rows.into_iter().zip(&cols).enumerate() {
            a.add_triplet(x, n_nodes + y, data[i]);
            a.add_triplet(n_nodes + y, x, data[i]);
        }
        a.to_csr()
    };
    
    
    let eigenvector = Array::from_vec(
        normalised_vector_init(total_elems)
    );
    
    
    // NOTE(jim): At the moment, sprs doesn't support a lot of linear algebra
    // operations and eig is one of them. We're going to use the iterative method
    // to find the eigenvector, but at some point in the future we will probably 
    // use an accelerated method.
    // Also note, this matrix really does need to be sparse because it's potentially
    // millions square but most entries are zero.
    
    evc_iteration_sparse(
        &adjacency_matrix,
        &eigenvector,
        tolerance,
        1,
        max_iterations,
    )
    


}

pub fn eigenvector_centrality(
    h: &HypergraphBase, 
    max_iterations: u32,
    tolerance: f64,
    rep: Representation,
    weighted_resultant: bool,
) -> Vec<f64> {
    
    let tmp_inc_mat = &h.incidence_matrix;
    let im_dims = tmp_inc_mat.shape();

    let (eigenvector, inc_mat, weights) = match (rep, weighted_resultant) {
        
        (Representation::Standard, true) => (
            normalised_vector_init(im_dims[1]),
            
            h.incidence_matrix
                .mapv(|x| f64::from(x))
                .dot(&Array::from_diag(&arr1(
                    &h.node_weights
                        .iter()
                        .map(|x| x.sqrt())
                        .collect::<Vec<_>>()
                    )
                )
            ),
            
            &h.edge_weights,
        ),
        
        (Representation::Standard, false) => (
            normalised_vector_init(im_dims[1]),
            
            h.incidence_matrix
                .mapv(|x| f64::from(x)),
            
            &h.edge_weights,
        ),
        
        (Representation::Dual, true) => (
            normalised_vector_init(im_dims[0]),
            
            h.incidence_matrix.t()
                .mapv(|x| f64::from(x))
                .dot(&Array::from_diag(&arr1(
                    &h.edge_weights
                        .iter()
                        .map(|x| x.sqrt())
                        .collect::<Vec<_>>()
                    )
                )
            ),
            
            &h.node_weights,
        ),
        
        (Representation::Dual, false) => (
            normalised_vector_init(im_dims[0]),
            
            h.incidence_matrix.t()
                .mapv(|x| f64::from(x)),
            
            &h.node_weights,
        ),
        
        // the bipartite representation needs to be handled 
        // separately with an implementation using sparse matrices
        (Representation::Bipartite, _) => return bipartite_eigenvector_centrality(
            &h.incidence_matrix,
            &h.edge_weights,
            tolerance,
            max_iterations,
        ).into_iter().collect::<Vec<_>>(),
    };

    evc_iteration(
        inc_mat.view(),
        &weights,
        eigenvector,
        tolerance,
        0,
        max_iterations,
    )
}

fn diag_sprs(
    v: &Vec<f64>,
) -> CsMat<f64> {
    
    let mut a = TriMat::new((v.len(), v.len()));
    
    for (i, x) in v.iter().enumerate() {
        a.add_triplet(i, i, *x);
    }
    a.to_csr()
}

pub fn degree_centrality(
    h: &HypergraphBase, 
    rep: Representation,
    weighted: bool
) -> Vec<f64> {
    
    match (rep, weighted) {
        (Representation::Standard, true) => {
            let inc_mat: CsMat<_> = CsMat::csr_from_dense(
                h.incidence_matrix.mapv(|x| x as f64).view(), 0.0
            );
            let w = diag_sprs(&h.edge_weights);
            let m = &w * & inc_mat;
            m.to_dense().sum_axis(Axis(0)).to_vec()
        }
        (Representation::Standard, false) => {
            h.incidence_matrix
                .mapv(|x| x as f64)
                .sum_axis(Axis(0))
                .to_vec()
        }
        (Representation::Dual, true) => {
            let inc_mat: CsMat<_> = CsMat::csr_from_dense(
                h.incidence_matrix.mapv(|x| x as f64).view(), 0.0
            );
            let w = diag_sprs(&h.node_weights);
            let m = &inc_mat * &w;
            m.to_dense().sum_axis(Axis(1)).to_vec()
        }
        (Representation::Dual, false) => {
            h.incidence_matrix
                .mapv(|x| x as f64)
                .sum_axis(Axis(1))
                .to_vec()
        }
        (_, _) => panic!("This set of optional arguments not supported"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use ndarray::{
        array, 
        s,
    };
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_linalg::Eig;
    
    #[test]
    fn reduced_powerset_t() {
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
            reduced_powerset(data.row(3), 4),
            expected
        );
        
    }
    
    #[test]
    fn reduced_powerset_singleton_t() {
        // Tests the function to construct the reduced powerset for a person with exactly 
        // 2 diseases. 
        
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
            reduced_powerset(data.row(0), 4),
            expected
        );
        
    }

    #[test]
    fn reduced_powerset_emptyset_t() {
        // Tests the function to construct the reduced powerset for a person that has exactly 
        // one disease.
        
        let data: Array2<u8> = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        let expected = HashSet::new();
        
        assert_eq!(
            reduced_powerset(data.row(1), 4),
            expected
        );
        
    }
        
    
    
    #[test]
    fn construct_edge_list_t() {
        // Tests the function to construct the edge list
        
        let data = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        let mut expected = Vec::new();
        
        //expected.push(vec![0,1,2,3]);
        
        expected.push(vec![0,1,2]);
        expected.push(vec![0,1,3]);
        expected.push(vec![0,2,3]);
        expected.push(vec![1,2,3]);
        
        expected.push(vec![0,1]);
        expected.push(vec![0,2]);
        expected.push(vec![0,3]);
        expected.push(vec![1,2]);
        expected.push(vec![1,3]);
        expected.push(vec![2,3]);
        
        assert_eq!(
            construct_edge_list(&data).sort(),
            expected.sort()
        );

        
    }
    
    
    #[test]
    fn overlap_coefficient_t() {
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
        // Tests the computation of the overlap coefficient - should panic as /0.
        
        let data = array![
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1]
        ];
        
        overlap_coefficient(data.slice(s![.., 1..=2]));
        
    }
    
    #[test]
    fn overlap_coefficient_one_column_prevalence_t() {
        // Tests the computation of the prevalence by the overlap coefficient function 
        
        let data = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        assert_eq!(
            overlap_coefficient(data.slice(s![.., 1..2])),
            4.0 / 5.0
        );
        
    }

    #[test]
    #[should_panic]
    fn overlap_coefficient_one_column_divide_by_zero_t() {
        // Tests whether the overlap coefficient function errors when given an empty array
        
        let data = array![[]];
        
        overlap_coefficient(data.slice(s![.., ..]));
        
    }
    
    #[test]
    fn incidence_matrix_t() {
        // tests the construction of the incidence matrix 
        
        let data = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        let edge_list = construct_edge_list(&data);
        
        let expected = array![
            [1,1,0,0],
            [1,0,1,0],
            [0,1,1,0],
            [1,1,0,1],
            [1,1,1,0],
            [1,0,0,1],
            [1,0,1,1],
            [0,1,0,1],
            [0,0,1,1],
            [0,1,1,1]
        ];
        
        let inc_mat = incidence_matrix(&edge_list);
        
        assert_eq!(
            inc_mat.axis_iter(Axis(0)).len(),
            expected.axis_iter(Axis(0)).len()
        );
        
        for x in inc_mat.axis_iter(Axis(0)) {
            
            assert!(expected.axis_iter(Axis(0)).contains(&x));
            
        }
        
    }
    
    #[test]
    fn compute_edge_weights_t() {
        
        // tests the computation of edge weights for all edges
        
        let data = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        let expected = HashMap::from([     
            (vec![0, 1], 1./2.),
            (vec![0, 2], 2./2.),
            (vec![1, 2], 2./3.),
            (vec![0, 3], 1./2.),
            (vec![1, 3], 3./3.),
            (vec![2, 3], 2./3.),
            (vec![0, 1, 2], 1./2.),
            (vec![0, 1, 3], 1./2.),
            (vec![0, 2, 3], 1./2.),
            (vec![1, 2, 3], 2./3.),
        ]);
        
        let edge_list = construct_edge_list(&data);
        let weights = compute_edge_weights(&data, &edge_list);
        
        assert_eq!(
            weights.len(),
            expected.len()
        );
        
        for (ii, edge) in edge_list.into_iter().enumerate() {
            assert_eq!(weights[ii], expected[&edge]);
        }
        
    }
    
    #[test]
    fn compute_edge_weights_zero_sets_t() {

        // tests to make sure edges with zero weight are appropriately discarded.

        let data = array![
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1]
        ]; // there is no one with disease 0 and disease 1

        let expected = HashMap::from([
            (vec![0, 2], 1./1.),
            (vec![1, 2], 2./3.),
        ]);
     
        let edge_list = construct_edge_list(&data);
        let weights = compute_edge_weights(&data, &edge_list);
        
        println!("{:?}", edge_list);
        println!("{:?}", weights);
        
        assert_eq!(
            weights.len(),
            expected.len()
        );
        
        for (ii, edge) in edge_list.into_iter().enumerate() {
            assert_eq!(weights[ii], expected[&edge]);
        }
     
    }

    #[test]
    fn compute_edge_weights_big_t() {

        // tests to make sure a bigish dataset runs at all.

        let n_diseases = 10;
        let n_subjects = 5000;
        
        let data = Array::random((n_subjects, n_diseases), Uniform::new(0.5, 1.5))
            .mapv(|x| x as u8);
        
        let h = compute_hypergraph(&data);
        
        println!("{}", h.edge_weights.len());
        
        println!("Yay");
     
    }
    
    #[test]
    fn compute_hypergraph_remaining_fns_t() {
        
        // Test the remaining functionality of compute_hypergraph 
        // (node_list, node_weights)
        
        let data = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        let h = compute_hypergraph(&data);
        
        let expected = vec![2./5., 4./5., 3./5., 3./5.];
        
        assert_eq!(h.node_list.len(), 4);
        
        for ii in h.node_list.iter() {
            assert_eq!(h.node_weights[*ii], expected[*ii]);
        }
    }
    
    // hypergraph methods
    
    #[test]
    fn adjacency_matrix_times_vector_t() {
        
        let inc_mat = array![[1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0]];
        
        let w_e = vec![0.5, 0.6666667, 0.5, 0.5, 1.0, 0.5, 0.6666667, 0.5, 0.6666667, 0.5, 1.0];
        let vector = vec![0.5; 4];
        
        let big_mess = inc_mat.t()
                .dot(&Array::from_diag(&arr1(&w_e)))
                .dot(&inc_mat);
      
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());
        let expected: Vec<f64> = adj.dot(&Array::from_vec(vector.clone()))
            .into_iter()
            .map(|x| x)
            .collect();
        
        
        // now, function that's being tested. 
        
        let res = adjacency_matrix_times_vector(&inc_mat.view(), &w_e, &vector);
        
        println!("{:?}", expected);
        println!("{:?}", res);
        
        for (x, y) in expected.iter().zip(&res) {
            println!("{:?}", (x - y).abs());
            assert!((x - y).abs() < 1e-6);
        }
        
    }
    
    #[test]
    fn adjacency_matrix_times_vector_random_inputs_t() {
        
        
        let n_cols = 10;
        let n_rows = 15;
        
        let mut rng = rand::thread_rng();
        
        let inc_mat = Array::random((n_rows, n_cols), Uniform::new(0., 10.));
        let weight: Vec<f64> = (0..n_rows)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        
        let vector: Vec<f64> = (0..n_cols)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        
        let big_mess = inc_mat.t()
                .dot(&Array::from_diag(&arr1(&weight)))
                .dot(&inc_mat);
      
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());
        let expected: Vec<f64> = adj.dot(&Array::from_vec(vector.clone()))
            .into_iter()
            .map(|x| x)
            .collect();
        
        let res = adjacency_matrix_times_vector(&inc_mat.view(), &weight, &vector);

        println!("{:?}", expected);
        println!("{:?}", res);
        
        for (x, y) in expected.iter().zip(&res) {
            println!("{:?}", (x - y).abs());
            assert!((x - y).abs() < 1e-3); // :\
        }
        
        
    }
    
    #[test]
    fn eigenvector_centrality_manual_t () {
        
        // test the computation of the eigenvector centrality
        // NOTE(jim): Only calculating the weighted resultant: sqrt(w_n) * M^T * w_e * M * sqrt(w_n)
        
        let h = HypergraphBase {
            incidence_matrix: array![[0, 1, 0, 1],
                 [1, 1, 0, 0],
                 [1, 0, 1, 1],
                 [0, 0, 1, 1],
                 [1, 1, 0, 1],
                 [0, 1, 1, 0],
                 [1, 0, 0, 1],
                 [1, 1, 1, 1],
                 [0, 1, 1, 1],
                 [1, 1, 1, 0],
                 [1, 0, 1, 0]],
            edge_weights: vec![1.0, 0.5, 0.5, 0.6666667, 0.5, 0.6666667, 0.5, 0.5, 0.6666667, 0.5, 1.0],
            node_weights: vec![0.4, 0.8, 0.6, 0.6],
            edge_list: vec![vec![1, 3], vec![0, 1], vec![0, 2, 3], vec![2, 3], vec![0, 1, 3], vec![1, 2], vec![0, 3], vec![0, 1, 2, 3], vec![1, 2, 3], vec![0, 1, 2], vec![0, 2]],
            node_list: vec![0, 1, 2, 3],
        };


        let inc_mat = h.incidence_matrix.mapv(|x| f64::from(x));

        let big_mess = Array::from_diag(&arr1(&
                h.node_weights
                    .iter()
                    .map(|x| x.sqrt())
                    .collect::<Vec<_>>()
                )
            )
            .dot(&inc_mat.t())
            .dot(&Array::from_diag(&arr1(&h.edge_weights)))
            .dot(&inc_mat)
            .dot(&Array::from_diag(&arr1(&
                    h.node_weights
                        .iter()
                        .map(|x| x.sqrt())
                        .collect::<Vec<_>>()
                    )
                )
            );
      
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());
        let (eig_vals, eig_vecs) = adj.eig().unwrap();
        
        let max_val = eig_vals
            .mapv(|x| x.re)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let max_index = eig_vals
            .mapv(|x| x.re)
            .iter()
            .position(|x| *x == max_val)
            .unwrap();
        
        let expected = eig_vecs
            .index_axis(Axis(1), max_index)
            .iter().map(|x| x.re.abs())
            .collect::<Vec<_>>();
        
        let tol = 0.00001;
        let max_iterations = 50;
        
        let centrality = eigenvector_centrality(
            &h,
            max_iterations, 
            tol, 
            Representation::Standard,
            true,
        );
        
        println!("{:?}", expected);
        println!("{:?}", centrality);
        
        let rms_error = expected.iter()
            .zip(&centrality)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f64>()
            .sqrt() / expected.len() as f64;
            
        println!("{:?}", rms_error);
        
        println!("{:?}", expected.iter() 
            .zip(centrality)
            .map(|(x, y)| (x - y))
            .collect::<Vec<_>>()
        );
        
        assert!(rms_error < tol);
    }
    
    #[test]
    fn eigenvector_centrality_t () {
        
        let n_diseases = 10;
        let n_subjects = 15;
        
        let data = Array::random((n_subjects, n_diseases), Uniform::new(0.5, 1.5))
            .mapv(|x| x as u8);
        
        let h = compute_hypergraph(&data);
        
        let big_mess = Array::from_diag(&arr1(&
                h.node_weights
                    .iter()
                    .map(|x| x.sqrt())
                    .collect::<Vec<_>>()
                )
            )
            .dot(&h.incidence_matrix.mapv(|x| f64::from(x)).t())
            .dot(&Array::from_diag(&arr1(&h.edge_weights)))
            .dot(&h.incidence_matrix.mapv(|x| f64::from(x)))
            .dot(&Array::from_diag(&arr1(&
                    h.node_weights
                        .iter()
                        .map(|x| x.sqrt())
                        .collect::<Vec<_>>()
                    )
                )
            );        
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());
        let (eig_vals, eig_vecs) = adj.eig().unwrap();
        
        let max_val = eig_vals
            .mapv(|x| x.re)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let max_index = eig_vals
            .mapv(|x| x.re)
            .iter()
            .position(|x| *x == max_val)
            .unwrap();
        
        let expected = eig_vecs
            .index_axis(Axis(1), max_index)
            .iter().map(|x| x.re.abs())
            .collect::<Vec<_>>();        
        let tol = 0.00001;
        let max_iterations = 50;
        
        let res = eigenvector_centrality(
            &h,
            max_iterations, 
            tol,
            Representation::Standard,
            true
        );
        
        println!("{:?}", expected);
        println!("{:?}", res);
        
        let rms_error = expected.iter()
            .zip(&res)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f64>()
            .sqrt() / expected.len() as f64;
            
        println!("{:?}", rms_error);
        
        println!("{:?}", expected.iter() 
            .zip(res)
            .map(|(x, y)| (x - y))
            .collect::<Vec<_>>()
        );
        
        assert!(rms_error < tol);
    }
    
    #[test]
    fn eigenvector_centrality_not_wr_t () {
        
        let n_diseases = 10;
        let n_subjects = 15;
        
        let data = Array::random((n_subjects, n_diseases), Uniform::new(0.5, 1.5))
            .mapv(|x| x as u8);
        
        let h = compute_hypergraph(&data);
        
        let big_mess = h.incidence_matrix.mapv(|x| f64::from(x)).t()
            .dot(&Array::from_diag(&arr1(&h.edge_weights)))
            .dot(&h.incidence_matrix.mapv(|x| f64::from(x)));
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());

        let (eig_vals, eig_vecs) = adj.eig().unwrap();
        
        let max_val = eig_vals
            .mapv(|x| x.re)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let max_index = eig_vals
            .mapv(|x| x.re)
            .iter()
            .position(|x| *x == max_val)
            .unwrap();
        
        let expected = eig_vecs
            .index_axis(Axis(1), max_index)
            .iter().map(|x| x.re.abs())
            .collect::<Vec<_>>();        
        let tol = 0.00001;
        let max_iterations = 50;
        
        let res = eigenvector_centrality(
            &h,
            max_iterations, 
            tol,
            Representation::Standard,
            false
        );
        
        println!("{:?}", expected);
        println!("{:?}", res);
        
        let rms_error = expected.iter()
            .zip(&res)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f64>()
            .sqrt() / expected.len() as f64;
            
        println!("{:?}", rms_error);
        
        println!("{:?}", expected.iter() 
            .zip(res)
            .map(|(x, y)| (x - y))
            .collect::<Vec<_>>()
        );
        
        assert!(rms_error < tol);
    }
    
    
   #[test]
   fn eigenvector_centrality_dual_rep_not_wr_not_rand_t() {
        let data = array![[0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 1],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0]];
           
        let h = compute_hypergraph(&data);
        
        let big_mess = h.incidence_matrix.mapv(|x| f64::from(x)).t()
            .dot(&Array::from_diag(&arr1(&h.edge_weights)))
            .dot(&h.incidence_matrix.mapv(|x| f64::from(x)));
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());

        let (eig_vals, eig_vecs) = adj.eig().unwrap();
        
        let max_val = eig_vals
            .mapv(|x| x.re)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let max_index = eig_vals
            .mapv(|x| x.re)
            .iter()
            .position(|x| *x == max_val)
            .unwrap();
        
        let expected = eig_vecs
            .index_axis(Axis(1), max_index)
            .iter().map(|x| x.re.abs())
            .collect::<Vec<_>>();        
        let tol = 0.00001;
        let max_iterations = 50;
        
        let res = eigenvector_centrality(
            &h,
            max_iterations, 
            tol,
            Representation::Standard,
            false
        );
        
        
        println!("Rust, Expected: {:?}", expected);
        println!("Rust, Result: {:?}", res);
        
        let rms_error = expected.iter()
            .zip(&res)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f64>()
            .sqrt() / expected.len() as f64;
            
        println!("{:?}", rms_error);
        
        println!("{:?}", expected.iter() 
            .zip(res)
            .map(|(x, y)| (x - y))
            .collect::<Vec<_>>()
        );
        
        assert!(rms_error < tol);        
        assert!(false);
   }
    
   #[test]
   fn eigenvector_centrality_dual_rep_t () {
        
        let n_diseases = 10;
        let n_subjects = 15;
        
        let data = Array::random((n_subjects, n_diseases), Uniform::new(0.5, 1.5))
            .mapv(|x| x as u8);
        
        let h = compute_hypergraph(&data);
        
        let big_mess = Array::from_diag(&arr1(&
                h.edge_weights
                    .iter()
                    .map(|x| x.sqrt())
                    .collect::<Vec<_>>()
                )
            )
            .dot(&h.incidence_matrix.mapv(|x| f64::from(x)))
            .dot(&Array::from_diag(&arr1(&h.node_weights)))
            .dot(&h.incidence_matrix.t().mapv(|x| f64::from(x)))
            .dot(&Array::from_diag(&arr1(&
                    h.edge_weights
                        .iter()
                        .map(|x| x.sqrt())
                        .collect::<Vec<_>>()
                    )
                )
            ); 
            
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());
        let (eig_vals, eig_vecs) = adj.eig().unwrap();
        
        let max_val = eig_vals
            .mapv(|x| x.re)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let max_index = eig_vals
            .mapv(|x| x.re)
            .iter()
            .position(|x| *x == max_val)
            .unwrap();
        
        let expected = eig_vecs
            .index_axis(Axis(1), max_index)
            .iter().map(|x| x.re.abs())
            .collect::<Vec<_>>();        
        let tol = 0.00001;
        let max_iterations = 50;
        
        let res = eigenvector_centrality(
            &h,
            max_iterations, 
            tol,
            Representation::Dual,
            true
        );
        
        let rms_error = expected.iter()
            .zip(&res)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f64>()
            .sqrt() / expected.len() as f64;
            
        println!("{:?}", expected);
        println!("{:?}", res);
        
        
        println!("\n {:?}", rms_error);
        
        println!("{:?}", expected.iter() 
            .zip(res)
            .map(|(x, y)| (x - y))
            .collect::<Vec<_>>()
        );
        
        assert!(rms_error < tol);
   }
   
   #[test]
   fn eigenvector_centrality_dual_rep_not_wr_t () {
        
        let n_diseases = 10;
        let n_subjects = 15;
        
        let data = Array::random((n_subjects, n_diseases), Uniform::new(0.5, 1.5))
            .mapv(|x| x as u8);
        
        let h = compute_hypergraph(&data);
        
        let big_mess = h.incidence_matrix.mapv(|x| f64::from(x))
            .dot(&Array::from_diag(&arr1(&h.node_weights)))
            .dot(&h.incidence_matrix.t().mapv(|x| f64::from(x))); 
            
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());
        let (eig_vals, eig_vecs) = adj.eig().unwrap();
        
        let max_val = eig_vals
            .mapv(|x| x.re)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let max_index = eig_vals
            .mapv(|x| x.re)
            .iter()
            .position(|x| *x == max_val)
            .unwrap();
        
        let expected = eig_vecs
            .index_axis(Axis(1), max_index)
            .iter().map(|x| x.re.abs())
            .collect::<Vec<_>>();        
        let tol = 0.00001;
        let max_iterations = 50;
        
        let res = eigenvector_centrality(
            &h,
            max_iterations, 
            tol,
            Representation::Dual,
            false
        );
        
        let rms_error = expected.iter()
            .zip(&res)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f64>()
            .sqrt() / expected.len() as f64;
            
        println!("{:?}", expected);
        println!("{:?}", res);
        
        
        println!("\n {:?}", rms_error);
        
        println!("{:?}", expected.iter() 
            .zip(res)
            .map(|(x, y)| (x - y))
            .collect::<Vec<_>>()
        );
        
        assert!(rms_error < tol);
   }   
   
   #[test]
   fn eigenvector_centrality_bipartite_rep_t () {
       
        let h = HypergraphBase {
            incidence_matrix: array![[0, 1, 0, 1],
                 [1, 1, 0, 0],
                 [1, 0, 1, 1],
                 [0, 0, 1, 1],
                 [1, 1, 0, 1],
                 [0, 1, 1, 0],
                 [1, 0, 0, 1],
                 [1, 1, 1, 1],
                 [0, 1, 1, 1],
                 [1, 1, 1, 0],
                 [1, 0, 1, 0]],
            edge_weights: vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            node_weights: vec![0.4, 0.8, 0.6, 0.6],
            edge_list: vec![vec![1, 3], vec![0, 1], vec![0, 2, 3], vec![2, 3], vec![0, 1, 3], vec![1, 2], vec![0, 3], vec![0, 1, 2, 3], vec![1, 2, 3], vec![0, 1, 2], vec![0, 2]],
            node_list: vec![0, 1, 2, 3],
        };
        
        let adjacency_matrix: Array2<f64> = array![[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        
        let (eig_vals, eig_vecs) = adjacency_matrix.eig().unwrap();
        
        let max_val = eig_vals
            .mapv(|x| x.re)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let max_index = eig_vals
            .mapv(|x| x.re)
            .iter()
            .position(|x| *x == max_val)
            .unwrap();
        
        let mut expected = eig_vecs
            .index_axis(Axis(1), max_index)
            .iter().map(|x| x.re.abs())
            .collect::<Vec<_>>();
           
        let ex_norm = expected
            .iter()
            .map(|x| x.powf(2.0))
            .sum::<f64>()
            .sqrt();
        
        expected = expected
            .iter() 
            .map(|x| x / ex_norm)
            .collect::<Vec<_>>();             
           
        let tol = 0.00001;
        let res = eigenvector_centrality(
            &h,
            50, 
            tol,
            Representation::Bipartite,
            true
        );    

        println!("\nTest printlns");
        println!("Expected: {:?}", expected);
        println!("Calculated: {:?}", res);  

        let rms_error = expected.iter()
            .zip(&res)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f64>()
            .sqrt() / expected.len() as f64;  
            
        println!("RMS error = {}", rms_error);            
        
        assert!(rms_error < tol);
   }
   
   #[test]
   fn eigenvector_centrality_bipartite_rep_rand_t () {
       
        let n_diseases = 4;
        let n_subjects = 50;
        
        let data = Array::random((n_subjects, n_diseases), Uniform::new(0.5, 1.5))
            .mapv(|x| x as u8);
        
        let h = compute_hypergraph(&data);
        let m_size = h.incidence_matrix.shape();
        let n_edges = m_size[0]; let n_nodes = m_size[1];
        let total_elems: usize = n_edges + n_nodes;
        
        let weighted_inc = h.incidence_matrix
            .mapv(|x| f64::from(x))
            .t()
            .dot(&Array2::from_diag(&arr1(&h.edge_weights)));
        
        let mut adjacency_matrix: Array2<f64> = Array::zeros((total_elems, total_elems));
        adjacency_matrix
            .slice_mut(s!(n_nodes..total_elems, 0..n_nodes))
            .assign(&weighted_inc.t());
        adjacency_matrix
            .slice_mut(s!(0..n_nodes, n_nodes..total_elems))
            .assign(&weighted_inc);
            
        println!("{:?}", adjacency_matrix.shape());
        
        let (eig_vals, eig_vecs) = adjacency_matrix.eig().unwrap();
        
        let max_val = eig_vals
            .mapv(|x| x.re)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let max_index = eig_vals
            .mapv(|x| x.re)
            .iter()
            .position(|x| *x == max_val)
            .unwrap();
        
        let mut expected = eig_vecs
            .index_axis(Axis(1), max_index)
            .iter().map(|x| x.re.abs())
            .collect::<Vec<_>>();
           
        let ex_norm = expected
            .iter()
            .map(|x| x.powf(2.0))
            .sum::<f64>()
            .sqrt();
        
        expected = expected
            .iter() 
            .map(|x| x / ex_norm)
            .collect::<Vec<_>>();             
           
        let tol = 0.00001;
        let res = eigenvector_centrality(
            &h,
            200, 
            tol,
            Representation::Bipartite,
            true
        );    

        println!("Expected: {:?}", expected);
        println!("Calculated: {:?}", res);  

        let rms_error = expected.iter()
            .zip(&res)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f64>()
            .sqrt() / expected.len() as f64;        
            
        println!("RMS error = {}", rms_error);
        
        assert!(rms_error < tol);
   }   
}
