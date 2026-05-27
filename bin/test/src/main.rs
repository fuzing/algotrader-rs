

const PATCH_TEMPORAL_WINDOW_SIZE: usize = 5;
const LOB_LEVELS: usize = 10;


type PatchData = [[f64; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];


fn main() {
    println!("Hello, world!");

    let mut m: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
    for i in 0..PATCH_TEMPORAL_WINDOW_SIZE {
        for j in 0..LOB_LEVELS {
            m[i][j] = j as f64;
        }
    }
    println!("{:?}", m);
}


