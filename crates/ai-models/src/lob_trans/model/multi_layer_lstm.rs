use burn::{
    nn::lstm::{Lstm, LstmConfig, LstmState},
    prelude::*,
};
use crate::lob_trans::model::mlp::{MLPConfig, MLP};

#[derive(Config, Debug)]
pub struct MultiLayerLstmConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub bias: Option<bool>,
    pub dropout: Option<f64>,
}


#[derive(Module, Debug)]
pub struct MultiLayerLstm {
    // Stacking multiple LSTMs in Burn is done by mapping sequentially or wrapping 
    // them manually. Here we map multiple layers to mimic PyTorch's num_layers > 1.
    layers: Vec<Lstm>,
    dropout: f64,
    d_input: usize,
    d_hidden: usize,
}


impl MultiLayerLstmConfig {
    pub fn init(&self, device: &Device) -> MultiLayerLstm {
        MultiLayerLstm::new(
            device,
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            self.bias.unwrap_or(true),
            self.dropout.unwrap_or(0.0),
        )
    }
}



impl MultiLayerLstm {
    pub fn new(
        device: &Device,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        with_bias: bool,
        dropout: f64,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            // Layer 1 takes the original input size, subsequent layers take hidden_size
            let current_input_size = if l == 0 { input_size } else { hidden_size };

            // Pytorch has bias by default, so we'll do the same
            let config = LstmConfig::new(current_input_size, hidden_size, with_bias)
                .with_batch_first(true);

            layers.push(config.init(device));
        }

        Self {
            layers,
            d_input: input_size,
            d_hidden: hidden_size,
            dropout,
        }
    }

    pub fn forward(
        &self,
        input: Tensor<3>, // Shape: [batch_size, seq_length, input_size]
        state: Option<Vec<LstmState<2>>>,
    ) -> (Tensor<3>, Vec<LstmState<2>>) {
        // let mut current_input = input;
        // let mut next_states = Vec::new();
        //
        // let [batch_size, sequence_length, _] = current_input.dims();
        //
        // // PyTorch style initialization if no state is provided
        // let empty_state: LstmState<2> = LstmState::new(
        //     Tensor::zeros([batch_size, self.d_hidden], &current_input.device()),
        //     Tensor::zeros([batch_size, self.d_hidden], &current_input.device()),
        // );
        //
        // for (i, layer) in self.layers.iter().enumerate() {
        //     let layer_state = state.as_ref()
        //         .and_then(|s| s.get(i).clone())
        //         .unwrap_or_else(|| &empty_state);
        //
        //     // Run Burn's LSTM forward pass
        //     let (output, final_state) = layer.forward(current_input, Some(
        //         // layer_state
        //         LstmState::new(
        //             layer_state.cell.clone(),
        //             layer_state.hidden.clone(),
        //         )
        //     ));
        //
        //     // Apply dropout between layers (if dropout is set and we're not on the last layer)
        //     if self.dropout > 0.0 && i < self.layers.len() - 1 {
        //         // current_input = output.dropout(self.dropout);
        //         current_input = output;
        //     } else {
        //         current_input = output;
        //     }
        //
        //     next_states.push(final_state);
        // }
        //
        // // Return the final output after all layers and the states for all layers
        // (current_input, next_states)

        let [batch_size, seq_length, _] = input.dims();
        let device = input.device();

        // eprintln!("Num LSTM layers: {} - ({}, {})", self.layers.len(), self.d_input, self.d_hidden);
        let num_layers = self.layers.len();
        let mut current_input = input;
        let mut next_states = Vec::with_capacity(num_layers);

        // Match PyTorch's default: create zero states if none are provided
        let mut states = state.unwrap_or_else(|| {
            (0..num_layers)
                .map(|_| {
                    let h = Tensor::zeros([batch_size, self.d_hidden], &device);
                    let c = Tensor::zeros([batch_size, self.d_hidden], &device);
                    LstmState::new(c, h)
                })
                .collect()
        });

        for i in 0..num_layers {
            // Forward pass for the current layer
            let (output, layer_state) =
                self.layers[i].forward(current_input, Some(
                    // states[i].clone()
                    LstmState::new(
                        states[i].cell.clone(),
                        states[i].hidden.clone(),
                    )
                ));

            // TODO - handle dropout???
            // Output of this layer becomes input for the next
            current_input = output;
            next_states.push(layer_state);
        }

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




//use burn::{
//     nn::rnn::{Lstm, LstmConfig, RnnState},
//     tensor::{backend::Backend, Tensor},
//     module::Module,
// };
//
// #[derive(Module, Debug, Clone)]
// pub struct MultiLayerLstm<B: Backend> {
//     layers: Vec<Lstm<B>>,
//     num_layers: usize,
//     hidden_size: usize,
// }
//
// impl<B: Backend> MultiLayerLstm<B> {
//     pub fn new(
//         input_size: usize,
//         hidden_size: usize,
//         num_layers: usize,
//         dropout: f64,
//         device: &B::Device,
//     ) -> Self {
//         let mut layers = Vec::with_capacity(num_layers);
//
//         for i in 0..num_layers {
//             let current_input_size = if i == 0 { input_size } else { hidden_size };
//             let config = LstmConfig::new(current_input_size, hidden_size)
//                 .with_dropout(if i == num_layers - 1 { 0.0 } else { dropout });
//
//             layers.push(config.init(device));
//         }
//
//         Self {
//             layers,
//             num_layers,
//             hidden_size,
//         }
//     }
//
//     /// Forward pass matching PyTorch's `(output, (h_n, c_n))` signature
//     pub fn forward(
//         &self,
//         input: Tensor<B, 3>,
//         state: Option<RnnState<B>>,
//     ) -> (Tensor<B, 3>, RnnState<B>) {
//         let mut current_input = input;
//         let mut next_states = Vec::with_capacity(self.num_layers);
//
//         // Unpack existing states or initialize empty ones
//         let mut states = match state {
//             Some(s) => s.states,
//             None => vec![(None, None); self.num_layers],
//         };
//
//         for i in 0..self.num_layers {
//             let (h_state, c_state) = states[i].clone();
//
//             // Lstm returns: (whole_sequence_output, final_layer_state)
//             let (output, layer_state) = self.layers[i].forward(current_input.clone(), h_state, c_state);
//
//             next_states.push((layer_state.hidden.clone(), layer_state.cell.clone()));
//             current_input = output; // Output of current layer becomes input to the next layer
//         }
//
//         // Return the final sequence output and the combined states for the whole multi-layer network
//         (current_input, RnnState::new(next_states))
//     }
// }

