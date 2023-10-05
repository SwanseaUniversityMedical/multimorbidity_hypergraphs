
#[allow(dead_code)]
#[allow(unused_imports)]

extern crate intel_mkl_src;
// old comments from cargo.toml
//# ndarray 0.15.0
//# blas-src 0.8
//# ndarray-linalg 0.14
//# blas-src = { version = "0.4.0", features = ["intel-mkl"] }

use ndarray::{
    array, 
    s, 
    Array,
    Array1,
    Array2,
    ArrayView1, 
    ArrayView2, 
    Axis,
    arr1
};
use sprs::{CsMat, CsVec};
use rand::Rng;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashSet;

pub fn compute_hypergraph(data: &Array2<u8>) -> Hypergraph {
    
    // Computes the hypergraph from an array of data.
    
    let edge_list = construct_edge_list(&data);
    let inc_mat = incidence_matrix(&edge_list);
    let node_list = (0..inc_mat.ncols()).collect::<Vec<_>>();
    
    let node_w = node_list
        .iter()
        .map(|x| overlap_coefficient(data.select(Axis(1), [*x].as_slice()).view()))
        .collect::<Vec<_>>();
    
    println!("{} {} {}", &edge_list.len(), &node_list.len(), &edge_list.len()*&node_list.len());
    
    Hypergraph {
        incidence_matrix: inc_mat, 
        edge_weights: compute_edge_weights(&data, &edge_list),
        node_weights: node_w,
        edge_list: edge_list, 
        node_list: node_list, 
    }
}

fn compute_edge_weights(data: &Array2<u8>, edge_list: &Vec<Vec<usize>>) -> Vec<f32> {
    
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

fn overlap_coefficient(data: ArrayView2<u8>) -> f32 {
    
    // Calculates the overlap coefficient for an edge, or the prevalence for 
    // data on only one disease. 
    
    match data.ncols() {
        
        1 => {
            data
                .iter()
                .sum::<u8>() as f32 / 
            data.nrows() as f32
        }
    
        
        _ => {
            let denom = data
                .axis_iter(Axis(1))
                .map(|x| x.sum())
                .min()
                .unwrap() as f32;
                
            if denom < 0.5 { 
            // NOTE(jim): this is an integer count cast to a f32, so if its less than 
            // 1.0 - eps its zero and the code should panic.
                panic!("overlap_coefficient: denominator is zero.");
            }
            
            data
                .axis_iter(Axis(0))
                .filter(|data_row| usize::from(data_row.sum()) == data_row.len())
                .count() as f32 / denom
            // NOTE(jim): .fold may be faster than .filter.count
        }
    
    }
       
}

fn reduced_powerset(data_row: ArrayView1<u8>) -> HashSet<Vec<usize>> {

    // Returns the "reduced" powerset of a single set of diseases (ie the powerset
    // without the empty or singleton sets. 
    
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

fn construct_edge_list(data: &Array2<u8>) -> Vec<Vec<usize>> {
    
    // Constructs the full edge list from the original data array.
    
    data
        .axis_iter(Axis(0))
        .map(|x| reduced_powerset(x))
        .flatten()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
    
}

/*
fn adjacency_matrix_times_vector(
    incidence_matrix: &ArrayView2<f32>,
    weight: &[f32],
    vector: &[f32],
) -> Vec<f32> {

    // Serial
    if let [outer, inner] = &incidence_matrix.shape() {
        
        let mut weighted_incidence: Array2<f32> = Array::zeros((*outer, *inner));
        for i in 0..*outer * *inner {
                weighted_incidence[[i / inner, i % inner]] += incidence_matrix[[i / inner, i % inner]] * weight[i / inner];
        }
        
        let mut intermediate = vec![0.0; *outer];

        for i in 0..*outer * *inner {
            intermediate[i / inner] += weighted_incidence[[i / inner, i % inner]] * vector[i % inner];
        }        
        
        let mut term_1 = vec![0.0; vector.len()];

        
        for i in 0..*outer * *inner {
            term_1[i % inner] += incidence_matrix[[i / inner, i % inner]] * intermediate[i / inner];
        }
        
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
*/

fn adjacency_matrix_times_vector(
    incidence_matrix: &ArrayView2<f32>,
    weight: &[f32],
    vector: &[f32],
) -> Vec<f32> {

    // ndarray-linalg / sprs.
    if let [outer, inner] = &incidence_matrix.shape() {
        
        let w: CsMat<f32> = CsMat::new(
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


/*
fn adjacency_matrix_times_vector(
    incidence_matrix: &ArrayView2<f32>,
    weight: &[f32],
    vector: &[f32],
) -> Vec<f32> {

    // Manual pool
    if let [outer, inner] = &incidence_matrix.shape() {
        
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
    
    
        let weighted_incidence: Array2<f32> = pool.install( ||
                Array::from_shape_vec((*outer, *inner), 
                    (0..outer*inner)
                        .into_par_iter()
                        .map(|i| incidence_matrix[[i / inner, i % inner]] * weight[i / inner])
                        .collect()
                ).unwrap()
            );
            
            
        let intermediate  = pool.install( ||
            Array::from_shape_vec((*outer, *inner), 
                (0..outer * inner)
                    .into_par_iter()
                    .map(|i| weighted_incidence[[i / inner, i % inner]] * vector[i % inner])
                    .collect()
                ).unwrap()
                .sum_axis(Axis(1))
        );

            
        let term_1 = pool.install( ||
            Array::from_shape_vec((*outer, *inner), 
                (0..outer * inner)
                    .into_par_iter()
                    .map(|i| incidence_matrix[[i / inner, i % inner]] * intermediate[i / inner])
                    .collect()
                ).unwrap()
                .sum_axis(Axis(0))
        );
                

        let subt = pool.install( ||
            Array::from_shape_vec((*outer, *inner), 
                (0..outer * inner)
                    .into_par_iter()
                    .map(|i| incidence_matrix[[i / inner, i % inner]] * weighted_incidence[[i / inner, i % inner]] * vector[i % inner])
                    .collect()
                ).unwrap()
                .sum_axis(Axis(0))
        );
            
        pool.install( ||
            (0..vector.len())
                .into_par_iter()
                .map(|i| term_1[i] - subt[i])
                .collect()
        )
        
    } else {
        panic!("The incidence matrix has the wrong shape.");
    }
}

*/


fn evc_iteration(
    inc_mat: ArrayView2<f32>, 
    weight: &Vec<f32>, 
    eigenvector: Vec<f32>,
    tolerance: f32,
    iter_no: u32,
    max_iterations: u32,
) -> Vec<f32> {
    
    // 1) perform an iteration
    
    let mut eigenvector_new = adjacency_matrix_times_vector(
        &inc_mat,
        weight,
        &eigenvector,
    );
    
    let evnew_norm = eigenvector_new
        .iter()
        .map(|x| x.powf(2.0))
        .sum::<f32>()
        .sqrt();
    
    eigenvector_new = eigenvector_new
        .iter() 
        .map(|x| x / evnew_norm)
        .collect::<Vec<_>>();
    
    let err_estimate = eigenvector_new
        .iter()
        .zip(&eigenvector)
        .map(|(&x, &y)| (x - y).powf(2.0))
        .sum::<f32>()
        .sqrt();
    
  
    if (err_estimate < tolerance) | (iter_no > max_iterations) {
        //println!("Converged in {} iterations", iter_no);
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

pub enum Representation {
    Standard,
    Dual,
}

#[derive(Debug)]
pub struct Hypergraph {
    incidence_matrix: Array2<u8>, 
    edge_weights: Vec<f32>,
    node_weights: Vec<f32>,
    edge_list: Vec<Vec<usize>>, 
    node_list: Vec<usize>, 
}

fn normalised_vector_init(len: usize) -> Vec<f32> {
    
    let mut rng = rand::thread_rng();
    
    let vector: Vec<f32> = (0..len)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();
    let norm = vector.iter().fold(0., |sum, &num| sum + num.powf(2.0)).sqrt();
    vector.iter().map(|&b| b / norm).collect()
}

impl Hypergraph {
    pub fn eigenvector_centrality(
        &self, 
        max_iterations: u32,
        tolerance: f32,
        rep: Representation,
    ) -> Vec<f32> {
        
        
        let tmp_inc_mat = &self.incidence_matrix;
        let im_dims = tmp_inc_mat.shape();

        
        
        let (eigenvector, inc_mat, weights) = match rep {
            
            Representation::Standard => (
                normalised_vector_init(im_dims[1]),
                
                self.incidence_matrix
                    .mapv(|x| f32::from(x))
                    .dot(&Array::from_diag(&arr1(
                        &self.node_weights
                            .iter()
                            .map(|x| x.sqrt())
                            .collect::<Vec<_>>()
                        )
                    )
                ),
                
                &self.edge_weights,
            ),
            
            Representation::Dual => (
                normalised_vector_init(im_dims[0]),
                
                self.incidence_matrix.t()
                    .mapv(|x| f32::from(x))
                    .dot(&Array::from_diag(&arr1(
                        &self.edge_weights
                            .iter()
                            .map(|x| x.sqrt())
                            .collect::<Vec<_>>()
                        )
                    )
                ),
                
                &self.node_weights,
            ),
        };
    
        evc_iteration(
            inc_mat.view(),
            weights,
            eigenvector,
            tolerance,
            0,
            max_iterations,
        )
    }
    
}


// Idiomatic rust is apparently to have tests and code in the same file
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
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
            reduced_powerset(data.row(3)),
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
            reduced_powerset(data.row(0)),
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
            reduced_powerset(data.row(1)),
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
        
        expected.push(vec![0,1,2,3]);
        
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
            [1,1,1,1],
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
            (vec![0, 1, 2, 3], 1./2.), // TODO(jim): decide if we need the "all diseases" edge.
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
            //(vec![0, 1], 0.0),
            (vec![0, 2], 1./1.),
            (vec![1, 2], 2./3.),
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
        let expected: Vec<f32> = adj.dot(&Array::from_vec(vector.clone()))
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
        let weight: Vec<f32> = (0..n_rows)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        
        let vector: Vec<f32> = (0..n_cols)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        
        let big_mess = inc_mat.t()
                .dot(&Array::from_diag(&arr1(&weight)))
                .dot(&inc_mat);
      
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());
        let expected: Vec<f32> = adj.dot(&Array::from_vec(vector.clone()))
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
        
        let h = Hypergraph {
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


        let inc_mat = h.incidence_matrix.mapv(|x| f32::from(x));

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
        
        let centrality = h.eigenvector_centrality(
            max_iterations, 
            tol, 
            Representation::Standard
        );
        
        println!("{:?}", expected);
        println!("{:?}", centrality);
        
        let rms_error = expected.iter()
            .zip(&centrality)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f32>()
            .sqrt() / expected.len() as f32;
            
        println!("{:?}", rms_error);
        
        println!("{:?}", expected.iter() 
            .zip(centrality)
            .map(|(x, y)| (x - y))
            .collect::<Vec<_>>()
        );
        
        assert!(rms_error < tol);
    }
    
    // TODO - Tests needed
    // 1 - eigenvector centrality with a random initial dataset - DONE
    // 2 - eigenvector centrality of the dual representation - DONE
    // 3 - eigenvector centrality of the bipartite representation
    // 4 - POSSIBLY: a single pass through the recursive function evc_iteration
    
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
            .dot(&h.incidence_matrix.mapv(|x| f32::from(x)).t())
            .dot(&Array::from_diag(&arr1(&h.edge_weights)))
            .dot(&h.incidence_matrix.mapv(|x| f32::from(x)))
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
        
        let res = h.eigenvector_centrality(
            max_iterations, 
            tol,
            Representation::Standard
        );
        
        println!("{:?}", expected);
        println!("{:?}", res);
        
        let rms_error = expected.iter()
            .zip(&res)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f32>()
            .sqrt() / expected.len() as f32;
            
        println!("{:?}", rms_error);
        
        println!("{:?}", expected.iter() 
            .zip(res)
            .map(|(x, y)| (x - y))
            .collect::<Vec<_>>()
        );
        
        assert!(rms_error < tol);
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
            .dot(&h.incidence_matrix.mapv(|x| f32::from(x)))
            .dot(&Array::from_diag(&arr1(&h.node_weights)))
            .dot(&h.incidence_matrix.t().mapv(|x| f32::from(x)))
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
        
        let res = h.eigenvector_centrality(
            max_iterations, 
            tol,
            Representation::Dual,
        );
        
        let rms_error = expected.iter()
            .zip(&res)
            .map(|(x, y)| (x - y).powf(2.0))
            .sum::<f32>()
            .sqrt() / expected.len() as f32;
            
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
}
