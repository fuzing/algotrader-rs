

const PATCH_TEMPORAL_WINDOW_SIZE: usize = 5;
const PATCH_TEMPORAL_STRIDE: usize = PATCH_TEMPORAL_WINDOW_SIZE;
const LOB_LEVELS: usize = 1;

const PREDICTION_TEMPORAL_WINDOW_SIZE: usize = 10;


const NUMBER_OF_SNAPSHOTS: usize = 10;


type PatchData = [[f64; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];


struct Level {
    price: f64,
    volume: u32,
}

struct Snapshot {
    bids: Vec<Level>,
    asks: Vec<Level>,
}


fn main() {
    println!("Hello, world!");

    let mut n: Option<Box<PatchData>> = None;

    {
        let mut m: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
        for i in 0..PATCH_TEMPORAL_WINDOW_SIZE {
            for j in 0..LOB_LEVELS {
                m[i][j] = j as f64;
            }
        }
        // println!("{:?}", m);
        n = Some(Box::new(m));
    }

    println!("{:?}", n);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Snapshot windows
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    let mut snapshots: Vec<Snapshot> = Vec::new();
    for _ in 0..NUMBER_OF_SNAPSHOTS {
        snapshots.push(Snapshot{
            bids: (0..LOB_LEVELS).map(|i| Level{price: i as f64, volume: i as u32}).collect(),
            asks: (0..LOB_LEVELS).map(|i| Level{price: i as f64, volume: i as u32}).collect(),
        });
    }

    // just some bogus numbers
    let price_mean: f64 = 10.0;
    let price_std_dev: f64 = 2.0;
    let volume_mean: f64 = 10.0;
    let volume_std_dev: f64 = 2.0;

    let mut patches: Vec<PatchData> = Vec::new();

    for i in 0..(snapshots.len() - PREDICTION_TEMPORAL_WINDOW_SIZE) {
        for j in (0..(PREDICTION_TEMPORAL_WINDOW_SIZE - PATCH_TEMPORAL_WINDOW_SIZE)).step_by(PATCH_TEMPORAL_STRIDE) {
            let mut bid_price_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
            let mut ask_price_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
            let mut bid_volume_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
            let mut ask_volume_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];

            for k in (0..PATCH_TEMPORAL_WINDOW_SIZE) {
                for l in 0..LOB_LEVELS {
                    bid_price_patch[k][l] = (snapshots[i + j + k].bids[l].price - price_mean) / price_std_dev;
                    bid_volume_patch[k][l] = (snapshots[i + j + k].bids[l].volume as f64 - volume_mean) / volume_std_dev;
                    ask_price_patch[k][l] = (snapshots[i + j + k].asks[l].price - price_mean) / price_std_dev;
                    ask_volume_patch[k][l] = (snapshots[i + j + k].asks[l].volume as f64 - volume_mean) / volume_std_dev;
                }
            }

            patches.push(bid_price_patch);
            patches.push(bid_volume_patch);
            patches.push(ask_price_patch);
            patches.push(ask_volume_patch);
        }
    }

    println!("Total number of snapshots: {}", snapshots.len());
    println!("Total number of patches: {}", patches.len());


}


