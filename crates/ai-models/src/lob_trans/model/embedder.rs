

//
// Embedding Layer
//

use burn::{
    module::{
        Module, ModuleDisplay, Content, DisplaySettings, Initializer, Param,
    },
    nn::{
        conv::{
            Conv2dConfig, Conv2d,
        },
        PaddingConfig2d,
    },
    config::Config,
    prelude::*,
    tensor::{Distribution},
};

#[derive(Config, Debug)]
pub struct EmbedderConfig {
    pub sequence_length: usize,
    pub token_size: usize,
}


#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Embedder {
    pub sequence_length: usize,
    pub token_size: usize,

    // convolution
    conv: Conv2d,

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
        )
    }
}


impl Embedder {
    pub fn new(
        device: &Device,
        sequence_length: usize,
        token_size: usize,
    ) -> Self {

        // TODO - get this in here somehow (hard coded at 4 channels * 15 values *
        let patch_size = 60;

        let conv = Conv2dConfig::new(
            [1,1],
            [patch_size, patch_size],
        )
            .with_stride([patch_size, patch_size])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);


        // only 1 class token (will be expanded/duplicated in model)
        let class_tokens = Param::from_tensor(
            Tensor::random([1, 1, token_size], Distribution::default(), device)
        );

        // learnable position encodings
        let positional_embeddings = Param::from_tensor(
            Tensor::random([1, sequence_length + 1, token_size], Distribution::default(), device)
        );

        Self {
            conv,
            sequence_length,
            token_size,
            class_tokens,
            positional_embeddings,
        }
    }

    pub fn forward(&self, tokens: Tensor<1>) -> Tensor<3> {
        // TODO - get this in here somehow
        let batch_size = 64;
        // let [batch_size, sequence_length, token_size] = tokens.dims();

        // expand the class tokens
        let class_tokens = self.class_tokens.val().expand([batch_size as i32, -1, -1]);

        // perform convolution
        let patch_embeddings = self.conv.forward(tokens);


        // prepend the class tokens
        let tokens_with_class = Tensor::cat(vec![class_tokens, tokens], 1);

        // add the positional encoding
        let tokens_with_class_and_positional_encoding = tokens_with_class.add(self.positional_embeddings.val());
        // TODO - PMB - should we divide by 2 or not?

        // and return that
        tokens_with_class_and_positional_encoding
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
