use criterion::{black_box, criterion_group, criterion_main, Criterion};
use multimorbidity_hypergraphs_rs::*;
use ndarray::prelude::*;
use rand::Rng;

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
    
    c.bench_function("build_hypergraph_5_10k", |b| b.iter(|| build_hypergraph(&data)));
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
        "eigenvector_centrality_5_10k", 
        |b| b.iter(|| h.eigenvector_centrality(100, 1e-6, Representation::Standard))
    );
}

/*
fn criterion_benchmark_2(c: &mut Criterion) {
    
    let n_diseases = 10;
    let n_patients = 5000;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    c.bench_function("build_hypergraph_10_5k", |b| b.iter(|| build_hypergraph(&data)));
}

fn criterion_benchmark_3(c: &mut Criterion) {
    
    let n_diseases = 15;
    let n_patients = 100000;
    
    let mut rng = rand::thread_rng();
    let data = Array::from_shape_fn(
        (n_patients, n_diseases),
        |_| {
            (rng.gen_range(0.0..1.0) > 0.8).then(|| 1).unwrap_or(0)
        }
    );
    
    c.bench_function("build_hypergraph_15_100k", |b| b.iter(|| build_hypergraph(&data)));
}
*/


criterion_group!(
    benches, 
    criterion_benchmark_build_hypergraph, 
    criterion_benchmark_eigenvector_centrality
);

criterion_main!(benches);