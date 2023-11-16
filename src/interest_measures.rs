

use ndarray::{
    ArrayView2, 
    Axis,
};


pub fn overlap_coefficient(data: ArrayView2<u8>) -> f64 {
    
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


#[cfg(test)]
mod tests {
    use super::*;
    
    use ndarray::{array, s};

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
}