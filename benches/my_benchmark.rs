use criterion::{black_box, criterion_group, criterion_main, Criterion};
use multimorbidity_hypergraphs::undirected_hypergraphs::*;
use ndarray::prelude::*;
use rand::Rng;

/*
fn build_hypergraph(data: &Array2<u8>) {

    let h = compute_hypergraph(&data);
    
}

fn criterion_benchmark_build_hypergraph(c: &mut Criterion) {
    
    let n_patients = 5000;
    let n_diseases = 10;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    c.bench_function("build_hypergraph_5k_10", |b| b.iter(|| build_hypergraph(&data)));
}

fn criterion_benchmark_eigenvector_centrality(c: &mut Criterion) {
    
    let n_patients = 5000;
    let n_diseases = 10;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_5k_10", 
        |b| b.iter(|| h.eigenvector_centrality(100, 1e-6, Representation::Standard))
    );
}

fn criterion_benchmark_eigenvector_centrality_dual(c: &mut Criterion) {
    
    let n_patients = 5000;
    let n_diseases = 10;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_5k_10_dual", 
        |b| b.iter(|| h.eigenvector_centrality(100, 1e-6, Representation::Dual))
    );
}
*/

fn criterion_benchmark_eigenvector_centrality_b(c: &mut Criterion) {
    
    let n_patients = 100000;
    let n_diseases = 10;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_100k_10", 
        |b| b.iter(|| eigenvector_centrality(&h, 100, 1e-6, Representation::Standard))
    );
}

/* 
fn criterion_benchmark_eigenvector_centrality_dual_b(c: &mut Criterion) {
    
    let n_patients = 100000;
    let n_diseases = 10;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_100k_10_dual", 
        |b| b.iter(|| h.eigenvector_centrality(100, 1e-6, Representation::Dual))
    );
}

fn criterion_benchmark_eigenvector_centrality_vb(c: &mut Criterion) {
    
    let n_patients = 100000;
    let n_diseases = 15;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_100k_15", 
        |b| b.iter(|| h.eigenvector_centrality(100, 1e-6, Representation::Standard))
    );
}

fn criterion_benchmark_eigenvector_centrality_dual_vb(c: &mut Criterion) {
    
    let n_patients = 100000;
    let n_diseases = 15;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_100k_15_dual", 
        |b| b.iter(|| h.eigenvector_centrality(100, 1e-6, Representation::Dual))
    );
}
*/

fn criterion_benchmark_eigenvector_centrality_b1(c: &mut Criterion) {
    
    let n_patients = 100000;
    let n_diseases = 11;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_100k_11", 
        |b| b.iter(|| eigenvector_centrality(&h, 100, 1e-6, Representation::Standard))
    );
}
fn criterion_benchmark_eigenvector_centrality_b2(c: &mut Criterion) {
    
    let n_patients = 100000;
    let n_diseases = 12;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_100k_12", 
        |b| b.iter(|| eigenvector_centrality(&h, 100, 1e-6, Representation::Standard))
    );
}

fn criterion_benchmark_eigenvector_centrality_b3(c: &mut Criterion) {
    
    let n_patients = 100000;
    let n_diseases = 13;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_100k_13", 
        |b| b.iter(|| eigenvector_centrality(&h, 100, 1e-6, Representation::Standard))
    );
}

fn criterion_benchmark_eigenvector_centrality_b4(c: &mut Criterion) {
    
    let n_patients = 100000;
    let n_diseases = 14;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_100k_14", 
        |b| b.iter(|| eigenvector_centrality(&h, 100, 1e-6, Representation::Standard))
    );
}
fn criterion_benchmark_eigenvector_centrality_b5(c: &mut Criterion) {
    
    let n_patients = 100000;
    let n_diseases = 15;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    let h = compute_hypergraph(&data);
    c.bench_function(
        "eigenvector_centrality_100k_15", 
        |b| b.iter(|| eigenvector_centrality(&h, 100, 1e-6, Representation::Standard))
    );
}




criterion_group!(
    benches, 
    //criterion_benchmark_build_hypergraph, 
    //criterion_benchmark_eigenvector_centrality,
    //criterion_benchmark_eigenvector_centrality_dual,
    criterion_benchmark_eigenvector_centrality_b,
    criterion_benchmark_eigenvector_centrality_b1,
    criterion_benchmark_eigenvector_centrality_b2,
    criterion_benchmark_eigenvector_centrality_b3,
    criterion_benchmark_eigenvector_centrality_b4,
    criterion_benchmark_eigenvector_centrality_b5,
    //criterion_benchmark_eigenvector_centrality_dual_b,
    //criterion_benchmark_eigenvector_centrality_vb,
    //criterion_benchmark_eigenvector_centrality_dual_vb
);

criterion_main!(benches);