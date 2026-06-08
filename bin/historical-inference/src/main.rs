

use extractors::{
    extractor::Extractor,
    interval_extractor::{
        IntervalExtractor,
        IntervalExtraction,
        IntervalExtractionWithGain,
        ExtractedDataFile,
    },
};

use burn::{
    prelude::*,
    nn::{PositionalEncodingConfig, PositionalEncoding},
    record::{CompactRecorder, Recorder},
    tensor::{
        Device,
        DeviceConfig,
        Element,
        Tensor,
        TensorData,
        Shape
    },
};

use statrs::statistics::Statistics;

use serde::{Deserialize, Serialize};
use anyhow::anyhow;

use ai_models::price_gain::{
    data::{
        batcher::{
            PriceGainBatcher,
            PriceGainInferenceBatch,
        },
        data::{
            PriceGainPatchType,
            PriceGainPatchSide,
        },
        dataset::{
            PriceGainItem,
            PriceGainDataset,
        },
        data_spec::{PriceGainDataSpec, PriceGainDataSpecBuilder}
    },
    model::{
        PriceGainModel,
        PriceGainModelConfig,
    },
    training::ExperimentConfig,
};

use clap::Parser as ClapParser;
use std::{
    collections::VecDeque,
    env,
    io::{ BufWriter, Read, stderr, stdin, stdout },
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    process::exit,
    time::Duration,
};
use tracing_subscriber::{EnvFilter, fmt};
use tracing::{debug, info, warn, error, Instrument};
use tokio;
use std::error::Error;
use std::fmt::Display;
use std::io::Write;
use burn::data::dataloader::batcher::Batcher;
use dotenv::dotenv;

use databento::{
    dbn::{
        MboMsg,
        decode::{AsyncDbnDecoder},
    },
};
use utilities::date_time::{nanos_to_offset_date_time_with_tz, str_to_offset_date_time};


#[cfg(not(any(feature = "f16", feature = "flex32")))]
#[allow(unused)]
type ElemType = f32;
// #[cfg(feature = "f16")]
// type ElemType = burn::tensor::f16;
#[cfg(feature = "flex32")]
type ElemType = burn::tensor::flex32;


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

    #[cfg(feature = "cuda")]
    return Device::cuda(burn::tensor::DeviceIndex::Default);

    #[cfg(feature = "rocm")]
    return Device::rocm(burn::tensor::DeviceIndex::Default);

    unreachable!("At least one backend will be selected.")
}

fn initialize_model(
    args: &Args,
    spec: &PriceGainDataSpec,
) -> Result<(PriceGainModel, Arc<PriceGainBatcher>, PositionalEncoding), Box<dyn Error>> {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{}/config.json", args.artifacts_folder.to_string_lossy()).as_str())
        .expect("Config file present");

    // Get number of classes from dataset
    let n_classes = PriceGainDataset::num_classes();

    // Initialize batcher for batching samples
    let batcher = Arc::new(PriceGainBatcher::new());

    let mut device = select_device();
    device
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
        .unwrap();
    
    // Load pre-trained model weights
    println!("Loading weights ...");
    let record = CompactRecorder::new()
        .load(format!("{}/model", args.artifacts_folder.to_string_lossy()).into(), &device)
        .expect("Trained model weights tb");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = PriceGainModelConfig::new(
        spec.sequence_length,
        spec.token_size,
        config.transformer.clone().with_dropout(0.0),           // override dropout for inference
        n_classes,
    )
        .init(&device)
        .load_record(record); // Initialize model with loaded weights

    let d_model = spec.token_size;
    let n_tokens = spec.sequence_length;
    let positional_max_timescale = spec.positional_max_timescale;
    let positional_encoder = PositionalEncodingConfig::new(d_model)
        .with_max_sequence_size(n_tokens)
        .with_max_timescale(positional_max_timescale)
        .init(&device);
    
    Ok((model, batcher, positional_encoder))
}


fn prepare_sample(
    positional_encoder: Option<&PositionalEncoding>,
    queue: &VecDeque<IntervalExtraction>,
    spec: &PriceGainDataSpec,
) -> Result<PriceGainItem, Box<dyn Error>> {
    assert_eq!(spec.prediction_intervals, queue.len());

    let mut bid_price_patches: Vec<Vec<f64>> = Vec::new();
    let mut bid_volume_patches: Vec<Vec<f64>> = Vec::new();
    let mut ask_price_patches: Vec<Vec<f64>> = Vec::new();
    let mut ask_volume_patches: Vec<Vec<f64>> = Vec::new();

    for j in (0..queue.len()).step_by(spec.patch_stride) {
        // create each patch - starting with each patch header value pair
        let mut bid_price_patch: Vec<f64> = vec![PriceGainPatchType::Price.value(), PriceGainPatchSide::Bid.value()];
        let mut bid_volume_patch: Vec<f64> = vec![PriceGainPatchType::Volume.value(), PriceGainPatchSide::Bid.value()];
        let mut ask_price_patch: Vec<f64> = vec![PriceGainPatchType::Price.value(), PriceGainPatchSide::Ask.value()];
        let mut ask_volume_patch: Vec<f64> = vec![PriceGainPatchType::Volume.value(), PriceGainPatchSide::Ask.value()];

        for k in 0..spec.patch_intervals {
            for l in 0..spec.lob_levels {
                bid_price_patch.push((queue[j + k].bids[l].price - spec.price_mean) / spec.price_std_dev);
                bid_volume_patch.push((queue[j + k].bids[l].volume as f64 - spec.volume_mean) / spec.volume_std_dev);
                ask_price_patch.push((queue[j + k].asks[l].price - spec.price_mean) / spec.price_std_dev);
                ask_volume_patch.push((queue[j + k].asks[l].volume as f64 - spec.volume_mean) / spec.volume_std_dev);
            }
        }

        assert_eq!(bid_price_patch.len(), spec.patch_size);
        assert_eq!(bid_volume_patch.len(), spec.patch_size);
        assert_eq!(ask_price_patch.len(), spec.patch_size);
        assert_eq!(ask_volume_patch.len(), spec.patch_size);

        // add patches
        bid_price_patches.push(bid_price_patch);
        bid_volume_patches.push(bid_volume_patch);
        ask_price_patches.push(ask_price_patch);
        ask_volume_patches.push(ask_volume_patch);
    }

    assert_eq!(bid_price_patches.len(), spec.sequence_length);

    let mut tokens: Vec<Vec<f64>> = Vec::with_capacity(spec.sequence_length);
    for i in 0..spec.sequence_length {
        let token = [
            bid_price_patches[i].clone(),
            bid_volume_patches[i].clone(),
            ask_price_patches[i].clone(),
            ask_volume_patches[i].clone(),
        ].concat();

        assert_eq!(token.len(), spec.token_size);
        tokens.push(token);
    }

    let final_tokens = if let Some(positional_encoder) = positional_encoder {
        // build a tensor of [batch_size, n_tokens, d_model] with batch size 1 and then add
        // positional encodings
        let flat = tokens.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flat.len(), 1 * spec.sequence_length * spec.token_size);

        let device = positional_encoder.devices()[0].clone();
        let tensor = Tensor::<3, Float>::from_floats(
            TensorData::new(flat, Shape::new([1, spec.sequence_length, spec.token_size])),
            &device
        );

        // add positional encodings and divide by 2.0 to normalize
        let tensor_with_positions = positional_encoder.forward(tensor).div_scalar(2.0);
        let vec_with_positions = tensor_with_positions.to_data().iter::<f64>().collect::<Vec<_>>();
        assert_eq!(vec_with_positions.len(), 1 * spec.sequence_length * spec.token_size);
        let final_vector = vec_with_positions;

        let nested_vec: Vec<Vec<f64>> = final_vector
            .chunks(spec.token_size) // Group into chunks of size 'm'
            .map(|chunk| chunk.to_vec())
            .collect();

        assert_eq!(nested_vec.len(), spec.sequence_length);
        nested_vec
    }
    else {
        tokens
    };


    //
    // // println!("Nested shape [{}, {}]", nested_vec.len(), nested_vec[0].len());
    //
    // let sample = PriceGainItem::new(
    //     nested_vec,
    //     0.0
    // );

    let sample = PriceGainItem::new(
        final_tokens,
        0.0
    );

    Ok(sample)
}


async fn inference(
    model: &PriceGainModel,
    batcher: &Arc<PriceGainBatcher>,
    positional_encoder: Option<&PositionalEncoding>,
    spec: &PriceGainDataSpec,
    queue: &VecDeque<IntervalExtraction>
) -> Result<bool, Box<dyn Error>> {
    let device = model.devices()[0].clone();

    let mut samples: Vec<PriceGainItem> = Vec::new();
    samples.push(prepare_sample(
        positional_encoder,
        queue,
        spec
    )?);

    // Run inference on the given samples
    let batch: PriceGainInferenceBatch = batcher.batch(samples, &device); // Batch samples using the batcher

    let predictions = model.infer(batch); // Get model predictions

    let prediction = predictions.clone().slice(0..1);
    // let logits = prediction.to_data();
    let class_index: i32 = prediction.argmax(1).squeeze_dim::<1>(1).into_scalar();
    // let class_name = PriceGainDataset::class_name(class_index as usize);
    // println!("Class: {}", class_name);

    if class_index == 2 {
        Ok(true)
    }
    else {
        Ok(false)
    }

    // // Print out predictions for each sample
    // for (i, text) in samples.into_iter().enumerate() {
    //     #[allow(clippy::single_range_in_vec_init)]
    //     let prediction = predictions.clone().slice([i..i + 1]); // Get prediction for current sample
    //     let logits = prediction.to_data(); // Convert prediction tensor to data
    //     let class_index: i32 = prediction.argmax(1).squeeze_dim::<1>(1).into_scalar(); // Get class index with the highest value
    //     let class = PriceGainDataset::class_name(class_index as usize); // Get class name
    //
    //     // Print sample text, predicted logits and predicted class
    //     println!(
    //         "\n=== Item {i} ===\n- Text: {text}\n- Logits: {logits}\n- Prediction: \
    //          {class}\n================"
    //     );
    // }
    // Ok(true)
}

async fn decode_data(
    model: &PriceGainModel,
    batcher: &Arc<PriceGainBatcher>,
    positional_encoder: Option<&PositionalEncoding>,
    path: &PathBuf,
    extractor: &mut impl Extractor<IntervalExtraction>,
    spec: &PriceGainDataSpec,
    holding_time_intervals: usize,
    start_date_nanos: u64,
    end_date_nanos: u64,
) -> Result<(), Box<dyn Error>> {
    let mut decoder = AsyncDbnDecoder::from_zstd_file(path).await?;

    let mut queue: VecDeque<IntervalExtraction> = VecDeque::new();

    let mut total_profit = 0.0;

    let mut holding_intervals = 0;
    let mut holding_purchase_price = 0.0;

    while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {
        if mbo.ts_recv >= start_date_nanos && mbo.ts_recv <= end_date_nanos {
            let results = extractor.push(mbo).await?;

            if !results.is_empty() {
                for result in results {
                    queue.push_back(result);

                    if queue.len() > spec.prediction_intervals {
                        queue.pop_front();
                    }

                    if holding_intervals > 0 {
                        holding_intervals -= 1;
                        if holding_intervals == 0 {
                            let sale_result = queue.iter().last().unwrap();
                            let sale_price = (sale_result.bids[0].price + sale_result.asks[0].price) / 2.0;
                            let share_block = 100.0;
                            let profit = (sale_price - holding_purchase_price) * share_block;
                            println!("Holding period over - bought for {}, sold for {}, Profit ({})", holding_purchase_price, sale_price, profit);
                            total_profit += profit;
                        }
                    }
                }

                if holding_intervals == 0 && queue.len() == spec.prediction_intervals {
                    let r = inference(
                        model,
                        batcher,
                        positional_encoder,
                        spec,
                        &queue,
                    ).await?;

                    if r {
                        let purchase_result = queue.iter().last().unwrap();
                        holding_purchase_price = (purchase_result.bids[0].price + purchase_result.asks[0].price) / 2.0;
                        holding_intervals = holding_time_intervals;
                    }
                }
            }
        }
    }

    println!("Total Profit was {}", total_profit);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>
{
    // get .env variables into environment
    dotenv().ok();
    let root_folder = env::var("ROOT_FOLDER").expect("no ROOT_FOLDER found in environment");

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

    // Canonicalize all input files, to ensure that the files exists and that
    // the path is valid. Store it in a vector for further processing.
    let inputs = args
        .inputs.clone()
        .into_iter()
        .map(|p| Path::new(&root_folder).join("data").join(p).canonicalize())
        .collect::<Result<Vec<_>, _>>().map_err(|e| anyhow!(e))?;

    println!("inputs: {:?}", inputs);


    // read in the spec file
    let spec = PriceGainDataSpec::from_file(&args.spec_file)?;

    // number of intervals that we're presuming holding for
    let holding_time_intervals: usize = (spec.holding_time_seconds as u64 * 1_000_000_000 / &spec.extraction_interval_nanos) as usize;

    let start_date_nanos = str_to_offset_date_time(&format!("{} 00:00:00 UTC", &args.start_date)).expect("Invalid start date").unix_timestamp_nanos() as u64;
    let end_date_nanos = str_to_offset_date_time(&format!("{} 23:59:59 UTC", &args.end_date)).expect("Invalid end date").unix_timestamp_nanos() as u64;

    // let mut all_data: Vec<IntervalExtractionWithGain> = Vec::new();
    
    let (model, batcher, positional_encoder) = initialize_model(&args, &spec)?;
    let positional_encoder = if args.with_positional_encoding {
        Some(&positional_encoder)
    }
    else {
        None
    };

    for input in inputs {
        let mut extractor = IntervalExtractor::builder()
            .nbr_lob_levels(&spec.lob_levels)
            .extraction_interval_nanos(&spec.extraction_interval_nanos)
            .build();

        decode_data(
            &model,
            &batcher,
            positional_encoder,
            &input,
            &mut extractor,
            &spec,
            holding_time_intervals,
            start_date_nanos,
            end_date_nanos,
        ).await?;
        // all_data.append(&mut data);

        println!("Stats: {}", extractor.stats());
    }

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    // start/end dates to extract from/to
    #[arg(long)]
    start_date: String,

    #[arg(long)]
    end_date: String,

    #[arg(long)]
    spec_file: PathBuf,

    #[arg(long)]
    artifacts_folder: PathBuf,

    #[arg(long)]
    with_positional_encoding: bool,


    #[arg()]
    inputs: Vec<PathBuf>,
}

