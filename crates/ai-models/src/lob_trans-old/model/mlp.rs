

//
// Multi-layer perceptron for output
//

use burn::{
    module::{Module, ModuleDisplay, Content, DisplaySettings, Initializer},
    config::Config,
    prelude::*,
    nn::{
        LayerNorm,
        LayerNormConfig,
        Linear,
        LinearConfig,
        activation::{
            Relu,
            Gelu,
        }
    }
};

#[derive(Config, Debug)]
pub struct MLPConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}


#[derive(Module, Debug)]
#[module(custom_display)]
pub struct MLP {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,

    pub layer_norm: LayerNorm,
    pub hidden_layer: Linear,
    pub activation: Relu,
    pub output_layer: Linear,
}


impl MLPConfig {
    pub fn init(&self, device: &Device) -> MLP {
        MLP::new(
            device,
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
        )
    }
}


impl MLP {
    pub fn new(
        device: &Device,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Self {
        let layer_norm = LayerNormConfig::new(input_dim)
            // .with_bias(true)
            // .with_epsilon(0.01)
            .init(&device);
        let hidden_layer = LinearConfig::new(input_dim, hidden_dim).init(device);
        let activation = Relu::new();
        // let output_layer = LinearConfig::new(hidden_dim, output_dim).init(device);
        let output_layer = LinearConfig::new(input_dim, output_dim).init(device);

        Self {
            input_dim,
            hidden_dim,
            output_dim,

            layer_norm,
            hidden_layer,
            activation,
            output_layer,
        }
    }

    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let x = self.layer_norm.forward(input);
        // let x = self.hidden_layer.forward(x);
        // let x = self.activation.forward(x);
        let x = self.output_layer.forward(x);
        x
    }
}


impl ModuleDisplay for MLP {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        // let [input_dim, hidden_dim] = self.hidden_layer.weight().shape().dims();
        content
            .add("input_dim", &self.input_dim)
            .add("hidden_dim", &self.hidden_dim)
            .add("output_dim", &self.output_dim)
            .optional()
    }
}
