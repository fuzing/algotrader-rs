

use burn::{
    prelude::*,
    nn::{PositionalEncodingConfig, PositionalEncoding},
    tensor::{
        Device,
        DeviceConfig,
        Element,
        Tensor,
        TensorData,
        Shape
    },
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::{
            Dataset,
            transform::PartialDataset
        },
    },
};

use std::{
    path::PathBuf,
    sync::Arc,
};

use ai_models::price_gain::data::{
    batcher::{PriceGainBatcher, PriceGainTrainingBatch},
    dataset::{PriceGainDataset, PriceGainItem},
};


// type Elem = f32;
type Elem = burn::tensor::f16;


#[allow(unreachable_code)]
fn select_device() -> Device {
    // #[cfg(feature = "flex")]
    return Device::flex();

    #[cfg(all(feature = "tch-gpu", not(target_os = "macos")))]
    return Device::libtorch_cuda(burn::tensor::DeviceIndex::Default);

    #[cfg(all(feature = "tch-gpu", target_os = "macos"))]
    return Device::libtorch_mps();

    #[cfg(feature = "tch-cpu")]
    return Device::libtorch();

    #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);

    #[cfg(feature = "cuda")]
    return Device::cuda(burn::tensor::DeviceIndex::Default);

    #[cfg(feature = "rocm")]
    return Device::rocm(burn::tensor::DeviceIndex::Default);

    unreachable!("At least one backend will be selected.")
}

fn tensor_ops() {
    let mut device = select_device();
    device
        .configure(DeviceConfig::default().float_dtype(Elem::dtype()))
        .unwrap();

    // let mut device = Device::
    let d_model = 4;
    let n_tokens = 5;
    let pe = PositionalEncodingConfig::new(d_model)
        .with_max_sequence_size(n_tokens)
        .with_max_timescale(1_000_000)
        .init(&device);

    const BATCH_SIZE: usize = 5;
    let t = Tensor::<3, Float>::zeros(Shape::new([BATCH_SIZE, n_tokens, d_model]), &device);
    // let a = Tensor::<1>::from_floats([1.0], &device);
    // let t2 = t.clone().add(a);
    let t = t.clone().add_scalar(1.0);
    println!("Tensor {}", t);
    println!("Tensor Shape {:?}", t.shape());
    let x = pe.forward(t.clone());
    println!("Tensor x {}", x);
    let y = pe.forward(t);
    println!("Tensor y {}", y);

    // pull the value at 1,2,1
    let v = y.clone().slice([1,2,1]);
    let v: f32 = v.into_scalar();
    println!("Tensor v 1,2,1 {}", v);


    // getting all float data out of a tensor
    let w = y.to_data();
    println!("Tensor Data {}", w);
    for x in w.iter::<f64>() {
        print!("{},", x);
    }
    println!("");

    // create a tensor from float data
    // let mut a: Vec<Vec<f32>> = Vec::new();
    // a.push(vec![1.0, 2.0, 3.0]);
    // a.push(vec![4.0, 5.0, 6.0]);
    // let mut a: Vec<f32> = Vec::new();
    // a.push(1.0);
    let mut a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let y = Tensor::<2, Float>::from_floats(
        // TensorData::new(a, vec![2,3]),
        TensorData::new(a, Shape::new([2,3])),
        &device
    );

    println!("Tensor Data {}", y);
}

fn create_splits<D, I>(dataset: D, train_test_validation_shares: (usize, usize, usize)) -> (PartialDataset<D, I>, PartialDataset<D, I>, PartialDataset<D, I>)
where
    D: Dataset<I> + Clone,
    I: Clone + Send + Sync,
{
    let total_len = dataset.len();

    let (train_share, test_share, validation_share) = train_test_validation_shares;
    let total_shares = train_share + test_share + validation_share;

    let test_size = (test_share as f64 / total_shares as f64 * total_len as f64).floor() as usize;
    let validation_size = (validation_share as f64 / total_shares as f64 * total_len as f64).floor() as usize;
    let train_size = total_len - test_size - validation_size;

    let train_dataset = PartialDataset::new(dataset.clone(), 0, train_size);
    let test_dataset = PartialDataset::new(dataset.clone(), train_size, train_size + test_size);
    let validation_dataset = PartialDataset::new(dataset, train_size + test_size, total_len);

    (train_dataset, test_dataset, validation_dataset)
}


fn stream_data() {


    /// Demonstrates how to use the **streaming CSV dataset**.
    ///
    /// - The CSV file is memory-mapped (no RAM copy)
    /// - Rows are read lazily and parsed only when needed
    /// - Dataloader workers fetch rows in parallel
    /// - Suitable for multi-GB datasets


    let mut device = select_device();
    device
        .configure(DeviceConfig::default().float_dtype(Elem::dtype()))
        .unwrap();


    let spec_filename = PathBuf::from("/run/media/peter/genetics/algotrader/data/KHC-2024.json");
    let dataset_filename = PathBuf::from("/run/media/peter/genetics/algotrader/data/KHC-2024.csv");

    // ---- Create dataset (streaming, no loading) ----
    println!("Indexing CSV into memory-mapped structure...");
    let full_dataset = PriceGainDataset::new(&spec_filename, &dataset_filename);
    let (train_dataset, test_dataset, validation_dataset) =
        create_splits(full_dataset.clone(), (4,1,0));

    println!("Full Set {}.  Train Set {}.  Test Set {}.  Validation Set {}", full_dataset.len(), train_dataset.len(), test_dataset.len(), validation_dataset.len());

    // ---- Build DataLoader ----
    // let batcher = CsvBatcher::<MyBackend>::new();
    let batcher = PriceGainBatcher::new();
    // let dataloader = DataLoaderBuilder::new(batcher)
    let dataloader: Arc<dyn DataLoader<PriceGainTrainingBatch>> = DataLoaderBuilder::new(batcher)
        .batch_size(64)
        .shuffle(42)    // Efficient even for huge datasets (shuffles indices)
        .num_workers(8) // Parallel reading/parsing
        .build(train_dataset);

    println!("Starting streaming batch iteration...");

    for j in 0..1 {
        // ---- Iterate over batches ----
        for (i, batch) in dataloader.iter().enumerate() {
            // if i == 0 {
            //     println!("First batch (tokens): {}", batch.tokens);
            //     println!("First batch (labels): {}", batch.labels);
            // }
            let vec = batch.labels.to_data().iter::<i32>().collect::<Vec<_>>();
            println!("labels: {:?}", vec);

            if i % 100 == 0 {
                println!("Processed batch {}", i);
            }
        }
    }
}


fn main() {

    stream_data();
    // tensor_ops();
    return ();
}


