
// directives to be used during development only. 
#[allow(dead_code)]
#[allow(unused_imports)]
// done

use ndarray::{
    array, 
    s, 
    Array,
    Array2, 
    ArrayView1, 
    ArrayView2, 
    Axis,
    arr1
};
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

fn standard_deviation(data: Vec<f32>, m: f32) -> f32 {

    let variance = data.iter().map(|value| {
        let diff = m - (value);

        diff * diff
    }).sum::<f32>() / data.len() as f32;


    variance.sqrt()

}


fn iterate_vector(
    incidence_matrix: &ArrayView2<f32>,
    weight: &Vec<f32>,
    eigenvector: &Vec<f32>,
) -> Vec<f32> {

    //let result: Vec<f32>;
    if let [outer, inner] = &incidence_matrix.shape() {
        
        // Happy with this
        let weighted_incidence: Array2<f32> = Array::from_shape_vec((*outer, *inner), 
            (0..outer*inner)
                .into_par_iter()
                .map(|i| (incidence_matrix[[i / inner, i % inner]] as f32) * weight[i % inner])
                .collect()
        ).unwrap();

        
        // Happy with this
        let intermediate: Vec<f32> = (0..*inner)
            .into_par_iter()
            .map(|k| {
                (0..eigenvector.len())
                    .map(|j| weighted_incidence[[j, k]] * eigenvector[j])
                    .sum::<f32>()
            })
            .collect();
        
        //happy with this 
        let term_1: Vec<f32> = (0..eigenvector.len())
            .into_par_iter()
            .map(|i| {
                (0..*inner)
                    .map(|k| weighted_incidence[[i, k]] * intermediate[k])
                    .sum()
            })
            .collect();
            
        // happy with this 
        let subt: Vec<f32> = (0..eigenvector.len())
            .map(|i| {
                (0..*inner)
                    .into_par_iter()
                    .map(|k| (incidence_matrix[[i, k]] as f32) * weighted_incidence[[i, k]] * eigenvector[i])
                    .sum()
            })
            .collect();
            
        (0..eigenvector.len())
            .into_par_iter()
            .map(|i| term_1[i] - subt[i])
            .collect()

    } else {
        panic!("The incidence matrix has the wrong shape.");
    }       
    //result

}

#[derive(Debug)]
pub struct Hypergraph {
    incidence_matrix: Array2<u8>, 
    edge_weights: Vec<f32>,
    node_weights: Vec<f32>,
    edge_list: Vec<Vec<usize>>, 
    node_list: Vec<usize>, 
}

impl Hypergraph {
    pub fn eigenvector_centrality(
        &self, 
        max_iterations: u32,
        tolerance: f32,
    ) -> Vec<f32> {
        
        let tmp_inc_mat = &self.incidence_matrix;
        let im_dims = tmp_inc_mat.shape();

        // generate a random initial estimate for the eigenvector.
        // Possibly slow but only has to be done once.
        let mut rng = rand::thread_rng();
        let mut eigenvector_old: Vec<f32> = vec![0.0; im_dims[0]];
        for ii in 0..eigenvector_old.len() {
            eigenvector_old[ii] = rng.gen_range(0.0..1.0);
        }

        let norm = eigenvector_old.iter().fold(0., |sum, &num| sum + num*num).sqrt();
        eigenvector_old = eigenvector_old.iter().map(|&b| b / norm).collect();

        let mut eigenvector_new = eigenvector_old.to_owned();
        let mut eigenvalue: f32 = 0.0;

        for iteration in 0..max_iterations {

            eigenvector_new = iterate_vector(
                &self.incidence_matrix.mapv(|x| f32::from(x)).view(),
                &self.edge_weights,
                &eigenvector_old,
            );


            // calculate the eigenvalue estimate:
            // 1) find all values of eigenvector_new > 0

            let mut iter_eigenvalue: Vec<f32> = eigenvector_new.iter().zip(
                eigenvector_old.iter()
            ).map(|(&a, &b)| {
                    let mut out = 0.0;
                    if b > 1e-6 {
                        out = a / b;
                    } 
                    out
                }
            ).collect();
            iter_eigenvalue.retain(|&b| b > 0.0);

            //println!("{:?}", iter_eigenvalue);

            // 2) estimate = mean( #1 above )
            eigenvalue = iter_eigenvalue.iter().sum::<f32>() / (iter_eigenvalue.len() as f32);

            // 3) error estimate = std( #1 above)    
            let err_estimate = standard_deviation(iter_eigenvalue, eigenvalue); 


            // 4) termination condition: if error_estimate / eigenvector_estimate < tolerance
                     
            if err_estimate / eigenvalue < tolerance {
                println!("Converged after {} iterations", iteration + 1);
                break;
            } 


            let norm = eigenvector_new.iter().fold(0., |sum, &num| sum + num*num).sqrt();
            eigenvector_old = eigenvector_new.iter().map(|&b| b / norm).collect();


        }


        //(eigenvalue, PyArray::from_vec(py, eigenvector_new))
        eigenvector_new
    
    }
    
}

// Idiomatic rust is apparently to have tests and code in the same file
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
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
    fn iterate_vector_t() {
        
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
        let w_n = vec![0.4, 0.8, 0.6, 0.6];
        
        
        let vector = vec![0.5; 4];
        
        
        
        let sqrt_w_n: Vec<f32> = w_n
            .iter()
            .map(|x: &f32| (*x).sqrt())
            .collect();
            
         
        let big_mess = Array::from_diag(&arr1(&sqrt_w_n))
                .dot(&inc_mat.t())
                .dot(&Array::from_diag(&arr1(&w_e)))
                .dot(&inc_mat)
                .dot(&Array::from_diag(&arr1(&sqrt_w_n)));
      
        let adj = &big_mess -  Array::from_diag(&big_mess.diag());
        let expected: Vec<f32> = adj.dot(&Array::from_vec(vector.clone()))
            .into_iter()
            .map(|x| x)
            .collect();
        
        
        let res = iterate_vector(&inc_mat.view(), &w_e, &vector);
        
        assert_eq!(
            res, 
            expected
        );
        
    }
    
    #[test]
    fn eigenvector_centrality_t () {
        
        // test the computation of the eigenvector centrality
        // NOTE(jim): Only calculating the weighted resultant: sqrt(w_n) * M^T * w_e * M * sqrt(w_n)
        
        let data = array![
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ];
        
        // Note(jim): the following python code was used to calculate the exoected value:
        /* `python 
        
        import numpy as np 
        inc_mat = np.array([[1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 0]]
        )
        w_e = np.array([0.5, 0.6666667, 0.5, 0.5, 1.0, 0.5, 0.6666667, 0.5, 0.6666667, 0.5, 1.0])
        w_n = np.array([0.4, 0.8, 0.6, 0.6])
        
        a = np.sqrt(np.diag(w_n)).dot(inc_mat.T).dot(np.diag(w_e)).dot(inc_mat).dot(np.diag(w_n))
        np.fill_diagonal(a, 0.0)
        e_vals, e_vecs = np.linalg.eig(a)
        np.abs(e_vecs[:, np.where((e_vals == e_vals.max()))[0][0]])
        
        output: array([0.43576871, 0.52442996, 0.51250412, 0.52193714])
        
        
        NOT weighted resultant
        
        import numpy as np 
        inc_mat = np.array([[1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 0]]
        )
        w_e = np.array([0.5, 0.6666667, 0.5, 0.5, 1.0, 0.5, 0.6666667, 0.5, 0.6666667, 0.5, 1.0])
        w_n = np.array([0.4, 0.8, 0.6, 0.6])
        
        a = np.sqrt(inc_mat.T.dot(np.diag(w_e)).dot(inc_mat))
        np.fill_diagonal(a, 0.0)
        e_vals, e_vecs = np.linalg.eig(a)
        np.abs(e_vecs[:, np.where((e_vals == e_vals.max()))[0][0]])
        
        output: array([0.48837738, 0.50224377, 0.50694174, 0.50224377])
        
        */
        //let expected = vec![0.43576871, 0.52442996, 0.51250412, 0.52193714];
        let expected = vec![0.48837738, 0.50224377, 0.50694174, 0.50224377];
        let e_norm = (expected.iter().fold(0.0, |acc, num| acc + num * num) as f32).sqrt();
        
        let tol = 0.001;
        let h = compute_hypergraph(&data);
        let centrality = h.eigenvector_centrality(10, tol);
        let c_norm = (centrality.iter().fold(0.0, |acc, num| acc + num * num) as f32).sqrt();
        
        println!("{:?}", expected.iter().map(|x| x / e_norm).collect::<Vec<_>>());
        println!("{:?}", centrality.iter().map(|x| x / c_norm).collect::<Vec<_>>());
        
        
        assert!(
            expected.iter().map(|x| x / e_norm)  //.collect::<Vec<_>>(), 
                .zip(centrality.iter().map(|x| x / c_norm).collect::<Vec<_>>())
                .map(|(x, y)| (x - y).abs() < tol)
                .all(|x| x)
        );
    }
}
