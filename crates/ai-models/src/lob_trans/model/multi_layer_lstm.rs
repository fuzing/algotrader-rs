use burn::{
    nn::lstm::{Lstm, LstmConfig, LstmState},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct MultiLayerLstm {
    // Stacking multiple LSTMs in Burn is done by mapping sequentially or wrapping 
    // them manually. Here we map multiple layers to mimic PyTorch's num_layers > 1.
    layers: Vec<Lstm>,
    d_hidden: usize,
}

impl MultiLayerLstm {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        device: &Device,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            // Layer 1 takes the original input size, subsequent layers take hidden_size
            let current_input_size = if l == 0 { input_size } else { hidden_size };

            // TODO - false for no bias - should we do this
            let config = LstmConfig::new(current_input_size, hidden_size, false)
                .with_batch_first(true);

            layers.push(config.init(device));
        }

        Self {
            layers,
            d_hidden: hidden_size,
        }
    }

    pub fn forward(
        &self,
        input: Tensor<3>, // Shape: [batch_size, seq_length, input_size]
        state: Option<Vec<LstmState<2>>>,
    ) -> (Tensor<3>, Vec<LstmState<2>>) {
        let mut current_input = input;
        let mut next_states = Vec::new();

        let [batch_size, sequence_length, _] = current_input.dims();

        // PyTorch style initialization if no state is provided
        let empty_state = LstmState::new(
            Tensor::zeros([batch_size, self.d_hidden], &current_input.device()),
            Tensor::zeros([batch_size, self.d_hidden], &current_input.device()),
        );

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_state = state.as_ref()
                .and_then(|s| s.get(i).clone())
                .unwrap_or_else(|| empty_state.clone());

            // Run Burn's LSTM forward pass
            let (output, final_state) = layer.forward(current_input, Some(layer_state));

            current_input = output;
            next_states.push(final_state);
        }

        // Return the final output after all layers and the states for all layers
        (current_input, next_states)
    }
}

//use burn::{
//     nn::*,
//     tensor::{backend::Backend, Tensor},
// };
//
// /// Multi-layer LSTM configuration identical to PyTorch's `nn.LSTM`.
// #[derive(Config, Debug)]
// pub struct MultiLstmConfig {
//     pub input_size: usize,
//     pub hidden_size: usize,
//     #[config(default = "1")]
//     pub num_layers: usize,
//     #[config(default = "false")]
//     pub batch_first: bool,
//     #[config(default = "false")]
//     pub dropout: f64,
// }
//
// #[derive(Module, Debug)]
// pub struct MultiLstm<B: Backend> {
//     layers: Vec<Lstm<B>>,
//     num_layers: usize,
//     batch_first: bool,
//     dropout: f64,
//     hidden_size: usize,
// }
//
// impl<B: Backend> MultiLstm<B> {
//     pub fn new(config: &MultiLstmConfig) -> Self {
//         let mut layers = Vec::with_capacity(config.num_layers);
//
//         for i in 0..config.num_layers {
//             // PyTorch's first layer takes input_size, subsequent layers take hidden_size
//             let current_input_size = if i == 0 { config.input_size } else { config.hidden_size };
//
//             let layer_config = LstmConfig::new(current_input_size, config.hidden_size)
//                 .with_batch_first(config.batch_first);
//
//             layers.push(layer_config.init(&Default::default()));
//         }
//
//         Self {
//             layers,
//             num_layers: config.num_layers,
//             batch_first: config.batch_first,
//             dropout: config.dropout,
//             hidden_size: config.hidden_size,
//         }
//     }
//
//     /// Forward pass mimicking PyTorch's stacked LSTM behavior
//     pub fn forward(
//         &self,
//         input: Tensor<B, 3>,
//         initial_states: Option<Vec<(Tensor<B, 2>, Tensor<B, 2>)>>,
//     ) -> (Tensor<B, 3>, Vec<(Tensor<B, 2>, Tensor<B, 2>)>) {
//         let mut states = Vec::with_capacity(self.num_layers);
//         let seq_length = if self.batch_first { input.dims()[1] } else { input.dims()[0] };
//
//         // Loop over layers
//         let mut current_input = input;
//         for i in 0..self.num_layers {
//             let initial_state = initial_states.as_ref().map(|s| {
//                 let (h, c) = s[i].clone();
//                 LstmState::new(h, c)
//             });
//
//             // PyTorch LSTM returns the full sequence or final states
//             let (layer_output, final_state) = self.layers[i].forward(current_input.clone(), initial_state);
//
//             // Apply dropout between layers (if dropout is set and we're not on the last layer)
//             if self.dropout > 0.0 && i < self.num_layers - 1 {
//                 current_input = layer_output.dropout(self.dropout);
//             } else {
//                 current_input = layer_output;
//             }
//
//             states.push((final_state.hidden, final_state.cell));
//         }
//
//         (current_input, states)
//     }
// }




//use burn::nn::rnn::{Lstm, LstmConfig, LstmState};
// use burn::tensor::backend::Backend;
// use burn::tensor::{Bool, Shape, Tensor};
// use burn::module::Module;
// use burn::config::Config;
//
// #[derive(Config, Debug)]
// pub struct MultiLstmConfig {
//     pub input_size: usize,
//     pub hidden_size: usize,
//     pub num_layers: usize,
//     #[config(default = "true")]
//     pub batch_first: bool,
// }
//
// #[derive(Module, Debug)]
// pub struct MultiLstm<B: Backend> {
//     layers: Vec<Lstm<B>>,
//     hidden_size: usize,
//     num_layers: usize,
// }
//
// impl<B: Backend> MultiLstm<B> {
//     pub fn new(config: &MultiLstmConfig) -> Self {
//         let mut layers = Vec::with_capacity(config.num_layers);
//
//         for i in 0..config.num_layers {
//             // First layer takes original input_size, subsequent layers take hidden_size
//             let input_dim = if i == 0 { config.input_size } else { config.hidden_size };
//
//             let layer_config = LstmConfig::new(input_dim, config.hidden_size)
//                 .with_batch_first(config.batch_first);
//
//             layers.push(layer_config.init(&Default::default()));
//         }
//
//         Self {
//             layers,
//             hidden_size: config.hidden_size,
//             num_layers: config.num_layers,
//         }
//     }
//
//     pub fn forward(
//         &self,
//         input: Tensor<B, 3>,
//         state: Option<Vec<LstmState<B, 2>>>,
//     ) -> (Tensor<B, 3>, Vec<LstmState<B, 2>>) {
//         let batch_size = input.dims()[0];
//         let seq_length = input.dims()[1];
//         let device = input.device();
//
//         let mut current_input = input;
//         let mut next_states = Vec::with_capacity(self.num_layers);
//
//         // Match PyTorch's default: create zero states if none are provided
//         let mut states = state.unwrap_or_else(|| {
//             (0..self.num_layers)
//                 .map(|_| {
//                     let h = Tensor::zeros([batch_size, self.hidden_size], &device);
//                     let c = Tensor::zeros([batch_size, self.hidden_size], &device);
//                     LstmState::new(h, c)
//                 })
//                 .collect()
//         });
//
//         for i in 0..self.num_layers {
//             // Forward pass for the current layer
//             let (output, layer_state) = self.layers[i].forward(current_input, Some(states[i].clone()));
//
//             // Output of this layer becomes input for the next
//             current_input = output.clone();
//             next_states.push(layer_state);
//         }
//
//         (current_input, next_states)
//     }
// }

