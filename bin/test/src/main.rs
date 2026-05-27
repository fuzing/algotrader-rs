

const ROWS: usize = 5;
const COLS: usize = 10;


type MultiDimensional = [[f64; COLS]; ROWS];


fn main() {
    println!("Hello, world!");

    let mut m: MultiDimensional = [[0.0; COLS]; ROWS];
    for i in 0..ROWS {
        for j in 0..COLS {
            m[i][j] = j as f64;
        }
    }
    println!("{:?}", m);
}


