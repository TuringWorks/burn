//! ONNX Swish node import implementation.
//!
//! Swish(x) = x * sigmoid(alpha * x)
//!
//! When alpha = 1.0 (default), this is equivalent to SiLU.

use super::prelude::*;

impl NodeCodegen for onnx_ir::swish::SwishNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // When alpha = 1.0, use burn's silu function
        // Otherwise, compute x * sigmoid(alpha * x)
        if (self.config.alpha - 1.0).abs() < f32::EPSILON {
            quote! {
                let #output = burn::tensor::activation::silu(#input);
            }
        } else {
            let alpha = self.config.alpha;
            quote! {
                let #output = {
                    let scaled = #input.clone().mul_scalar(#alpha);
                    #input.mul(burn::tensor::activation::sigmoid(scaled))
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::swish::{SwishConfig, SwishNodeBuilder};

    #[test]
    fn test_swish_default_alpha() {
        let node = SwishNodeBuilder::new("swish1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(SwishConfig::new(1.0))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::silu(input);
            output
        }
        ");
    }

    #[test]
    fn test_swish_custom_alpha() {
        let node = SwishNodeBuilder::new("swish1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(SwishConfig::new(0.5))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let scaled = input.clone().mul_scalar(0.5f32);
                input.mul(burn::tensor::activation::sigmoid(scaled))
            };
            output
        }
        ");
    }
}
