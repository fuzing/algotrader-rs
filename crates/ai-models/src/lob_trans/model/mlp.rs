

//
// Multi-layer perceptron for output
//

use burn::{
    module::{Module, ModuleDisplay, Content, DisplaySettings, Initializer},
    config::Config,
    prelude::*,
    nn::{
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
        let hidden_layer = LinearConfig::new(input_dim, hidden_dim).init(device);
        let activation = Relu::new();
        let output_layer = LinearConfig::new(hidden_dim, output_dim).init(device);

        Self {
            input_dim,
            hidden_dim,
            output_dim,

            hidden_layer,
            activation,
            output_layer,
        }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        let x = self.hidden_layer.forward(input);
        let x = self.activation.forward(x);
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
