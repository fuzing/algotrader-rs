use burn::{
    nn::lstm::{Lstm, LstmConfig, LstmState},
    prelude::*,
};

#[derive(Module, Debug, Clone)]
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

            // false for no bias
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

        let batch_size = current_input.dims()[0];
        let seq_length = current_input.dims()[1];

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

