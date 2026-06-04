
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Train our AI Model
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


// use anyhow::anyhow;
use clap::Parser as ClapParser;
use std::{
    env,
    io::{ Read, stderr, stdin, stdout },
    path::PathBuf,
    process::exit,
    time::Duration,
};
use tracing_subscriber::{EnvFilter, fmt};
use tracing::{debug, info, warn, error, Instrument};
use tokio;
use std::error::Error;
use dotenv::dotenv;

use burn::{
    data::{
        dataloader::{DataLoaderBuilder, DataLoader},
        dataset::transform::SamplerDataset
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    nn::{attention::SeqLengthOption, transformer::TransformerEncoderConfig},
    optim::{
        AdamConfig,
        decay::{
            WeightDecayConfig,
        }
    },
    prelude::*,
    record::{CompactRecorder, Recorder},
    train::{
        ExecutionStrategy,
        Learner,
        SupervisedTraining,
        metric::{
            AccuracyMetric, CudaMetric, IterationSpeedMetric, LearningRateMetric, LossMetric,
        }
    },
    tensor::{
        Tensor,
        DType,
        DeviceConfig,
        Element,
    },
};
use std::sync::Arc;
use burn::data::dataloader::Dataset;
use burn::data::dataset::transform::PartialDataset;
use burn::nn::activation::ActivationConfig;
use burn::nn::SwiGluConfig;
use ai_models::price_gain::{
    data::{
        batcher::{PriceGainBatcher, PriceGainTrainingBatch},
        dataset::{PriceGainDataset, PriceGainItem},
    },
    model::{
        PriceGainModelConfig,
        PriceGainModel,
    },
    training::ExperimentConfig,
};

// #[cfg(not(any(feature = "f16", feature = "flex32")))]
// #[allow(unused)]
// type ElemType = f32;
// #[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;
#[cfg(feature = "flex32")]
type ElemType = burn::tensor::flex32;


fn create_artifact_dir(artifact_dir: &PathBuf) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}


#[cfg(all(feature = "cuda", not(feature = "ddp")))]
pub fn launch_multi() {
    let mut devices = Device::enumerate(burn::tensor::DeviceType::Cuda);

    devices.iter_mut().for_each(|d| {
        d.configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
            .unwrap()
    });

    launch(ExecutionStrategy::MultiDevice(
        devices,
        burn::train::MultiDeviceOptim::OptimSharded,
    ))
}

#[cfg(all(feature = "cuda", feature = "ddp"))]
pub fn launch_multi<B: AutodiffBackend + DistributedBackend>() {
    let mut devices = Device::enumerate(burn::tensor::DeviceType::Cuda);

    devices.iter_mut().for_each(|d| {
        d.configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
            .unwrap()
    });

    launch(ExecutionStrategy::ddp(
        devices,
        DistributedConfig {
            all_reduce_op: ReduceOperation::Mean,
        },
    ))
}


#[allow(unreachable_code)]
fn select_device() -> Device {
    // #[cfg(feature = "flex")]
    // return Device::flex();

    // #[cfg(all(feature = "tch-gpu", not(target_os = "macos")))]
    return Device::libtorch_cuda(burn::tensor::DeviceIndex::Default);

    #[cfg(all(feature = "tch-gpu", target_os = "macos"))]
    return Device::libtorch_mps();

    #[cfg(feature = "tch-cpu")]
    return Device::libtorch();

    #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);

    // #[cfg(feature = "cuda")]
    // return Device::cuda(burn::tensor::DeviceIndex::Default);

    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);

    #[cfg(feature = "rocm")]
    return Device::rocm(burn::tensor::DeviceIndex::Default);

    unreachable!("At least one backend will be selected.")
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

async fn train(
    spec_path: &PathBuf,
    dataset_path: &PathBuf,
    artifact_path: &PathBuf,
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    let full_dataset = PriceGainDataset::new(spec_path, dataset_path);
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(full_dataset.spec.token_size, args.feed_forward_size, args.transformer_heads, args.transformer_layers)
            .with_norm_first(true)
            .with_quiet_softmax(true)
            .with_dropout(args.dropout),
            /*.with_activation(ActivationConfig::SwiGlu(SwiGluConfig::new())),*/
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
        args.batch_size,         // batch size
        args.shuffle_seed,         // shuffle seed
        args.num_epochs,          // number of epochs
    );

    create_artifact_dir(artifact_path);

    let mut device = select_device();
    device
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))?;

    println!("{:?}", device);
    // return Ok(());

    let strategy = ExecutionStrategy::SingleDevice(device);

    let (dataset_train, dataset_test, _dataset_validation) =
        create_splits(full_dataset, (4,1,0));


    // ---- Build DataLoader ----
    let batcher = PriceGainBatcher::new();

    let dataloader_train: Arc<dyn DataLoader<PriceGainTrainingBatch>> = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.shuffle_seed)    // Efficient even for huge datasets (shuffles indices)
        .num_workers(8) // Parallel reading/parsing
        .build(dataset_train);

    let dataloader_test: Arc<dyn DataLoader<PriceGainTrainingBatch>> = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.shuffle_seed)    // Efficient even for huge datasets (shuffles indices)
        .num_workers(2) // Parallel reading/parsing
        .build(dataset_test);

    // Initialize model
    let model = PriceGainModelConfig::new(
        config.transformer.clone(),
        PriceGainDataset::num_classes(),
    )
        .init(&strategy.main_device().clone().autodiff());

    // Initialize optimizer
    let optim = config.optimizer.init();

    // Initialize learning rate scheduler
    let lr_scheduler = NoamLrSchedulerConfig::new(1e-2)
        .with_warmup_steps(1_000)
        .with_model_size(config.transformer.d_model)
        .init()?;

    // Initialize learner
    let training = SupervisedTraining::new(artifact_path, dataloader_train, dataloader_test)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train(IterationSpeedMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .with_training_strategy(strategy.into())
        .num_epochs(config.num_epochs)
        .summary();

    // Train the model
    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    // Save the configuration and the trained model
    config.save(format!("{}/config.json", artifact_path.to_string_lossy()))?;
    CompactRecorder::new()
        .record(
            result.model.into_record(),
            format!("{}/model", artifact_path.to_string_lossy()).into(),
        )?;


    Ok(())
}




#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>
{
    // get .env variables into environment
    dotenv().ok();

    // tracing format
    fmt()
        // .with_level(true)
        // .without_time()
        // .with_file(false)
        // .with_line_number(false)
        // .with_thread_ids(true)
        // .with_thread_names(true)
        // .pretty()
        .with_ansi(false)   // turns off display characters that change color etc.
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    info!("Building order book");

    // Parse the command line arguments
    let args = Args::parse();
    // info!("Run with arguments: {args:#?}");
    let root_folder = env::var("ROOT_FOLDER").expect("no ROOT_FOLDER found in environment");

    let spec_path: PathBuf = PathBuf::from(std::format!("{}/data/{}", root_folder, &args.spec_file));
    let dataset_path: PathBuf = PathBuf::from(std::format!("{}/data/{}", root_folder, &args.dataset_file));
    let artifacts_path: PathBuf = PathBuf::from(&args.artifacts_folder);

    train(&spec_path, &dataset_path, &artifacts_path, &args).await?;

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    // /// Write additional debut output in the output directory.
    // #[arg(short, long)]
    // enable_debug_output: bool,

    #[arg(long)]
    spec_file: String,

    #[arg(long)]
    dataset_file: String,

    #[arg(long)]
    dropout: f64,

    #[arg(long)]
    batch_size: usize,

    #[arg(long)]
    num_epochs: usize,

    #[arg(long)]
    shuffle_seed: u64,

    #[arg(long)]
    transformer_heads: usize,

    #[arg(long)]
    transformer_layers: usize,

    #[arg(long)]
    feed_forward_size: usize,

    #[arg(long)]
    artifacts_folder: String,



}

