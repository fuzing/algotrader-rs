

//
// Embedding Layer
//

use burn::{
    module::{
        Module, ModuleDisplay, Content, DisplaySettings, Initializer, Param,
    },
    nn::{LinearConfig, Linear},
    config::Config,
    prelude::*,
    tensor::{Distribution},
};

#[derive(Config, Debug)]
pub struct EmbedderConfig {
    pub sequence_length: usize,
    pub token_size: usize,
    pub d_model: usize,                 // size of model after linear stage
}


#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Embedder {
    pub sequence_length: usize,
    pub token_size: usize,
    pub d_model: usize,

    // input linear layer
    linear: Linear,

    // [batch_size, class_token]
    class_tokens: Param<Tensor<3>>,

    // [batch_size, seq_length, d_model]
    positional_embeddings: Param<Tensor<3>>,
}


impl EmbedderConfig {
    pub fn init(&self, device: &Device) -> Embedder {
        Embedder::new(
            device,
            self.sequence_length,
            self.token_size,
            self.d_model,
        )
    }
}


impl Embedder {
    pub fn new(
        device: &Device,
        sequence_length: usize,     // the total number of patch-pairs
        token_size: usize,          // each token should be the size of 2 patches (i.e. price/volume)
        d_model: usize,             // size of model after linear stage
    ) -> Self {

        let linear = LinearConfig::new(token_size, d_model).init(device);

        // only 1 class token (will be expanded/duplicated in model)
        let class_tokens = Param::from_tensor(
            Tensor::random([1, 1, d_model], Distribution::default(), device)
        );

        // learnable position encodings
        let positional_embeddings = Param::from_tensor(
            Tensor::random([1, sequence_length + 1, d_model], Distribution::default(), device)
        );

        Self {
            sequence_length,
            token_size,
            d_model,
            linear,
            class_tokens,
            positional_embeddings,
        }
    }

    pub fn forward(&self, tokens: Tensor<3>) -> Tensor<3> {
        let [batch_size, sequence_length, token_size] = tokens.dims();

        // projection
        let x = self.linear.forward(tokens);

        // expand the class tokens
        let class_tokens = self.class_tokens.val().expand([batch_size as i32, -1, -1]);

        // prepend the class tokens
        let x = Tensor::cat(vec![class_tokens, x], 1);

        // add the positional encoding
        let x = x.add(self.positional_embeddings.val());

        // divide by 2 to normalize
        let x = x.div_scalar(2.0);

        // and return that
        x
    }
}


impl ModuleDisplay for Embedder {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        // let [input_dim, hidden_dim] = self.hidden_layer.weight().shape().dims();
        content
            // .add("input_dim", &self.input_dim)
            // .add("hidden_dim", &self.hidden_dim)
            // .add("output_dim", &self.output_dim)
            .optional()
    }
}
