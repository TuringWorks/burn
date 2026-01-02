//! ONNX RMSNormalization node import implementation.
//!
//! RMS Normalization: Y = X / sqrt(mean(X^2) + epsilon) * scale
//!
//! Maps to burn::nn::RmsNorm

use burn_store::TensorSnapshot;

use super::prelude::*;

impl NodeCodegen for onnx_ir::node::rms_norm::RmsNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let d_model = self.config.d_model.to_tokens();
        let epsilon = self.config.epsilon;

        Some(Field::new(
            self.name.clone(),
            quote! {
                RmsNorm<B>
            },
            quote! {
                let #name = RmsNormConfig::new(#d_model)
                    .with_epsilon(#epsilon)
                    .init(device);
            },
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        // Gamma (scale) tensor at input index 1
        if let Some(scale_input) = self.inputs.get(1) {
            let gamma_path = format!("{}.gamma", field_name);
            if let Some(snapshot) = create_lazy_snapshot(scale_input, &gamma_path, "RmsNorm") {
                snapshots.push(snapshot);
            }
        }

        snapshots
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        if self.config.full_precision {
            // When stash_type == 1, compute in full precision
            quote! {
                let #output = {
                    let dtype = #input.dtype();
                    self.#field.forward(#input.cast(burn::tensor::DType::F32)).cast(dtype)
                };
            }
        } else {
            quote! {
                let #output = self.#field.forward(#input);
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::RmsNorm");
        imports.register("burn::nn::RmsNormConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::rms_norm::{RmsNormConfig, RmsNormalizationNode, RmsNormalizationNodeBuilder};

    fn create_rms_norm_node(name: &str, full_precision: bool) -> RmsNormalizationNode {
        let config = RmsNormConfig::new(512, 1e-5, full_precision);

        RmsNormalizationNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_rms_norm_forward() {
        let node = create_rms_norm_node("rms_norm1", true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let dtype = input.dtype();
                self.rms_norm1.forward(input.cast(burn::tensor::DType::F32)).cast(dtype)
            };
            output
        }
        ");
    }

    #[test]
    fn test_rms_norm_forward_no_full_precision() {
        let node = create_rms_norm_node("rms_norm1", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = self.rms_norm1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_rms_norm_field_init() {
        let node = create_rms_norm_node("rms_norm1", true);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @"let rms_norm1 = RmsNormConfig::new(512).with_epsilon(0.00001f64).init(device);");
    }
}
