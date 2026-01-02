//! ONNX Shrink node import implementation.
//!
//! Shrink applies soft-thresholding:
//! - If x < -lambd: output = x + bias
//! - If x > lambd: output = x - bias
//! - Otherwise: output = 0
//!
//! Maps to burn operations using mask_where

use super::prelude::*;

impl NodeCodegen for onnx_ir::node::shrink::ShrinkNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lambd = self.config.lambd;
        let bias = self.config.bias;
        let neg_lambd = -lambd;

        // Shrink formula:
        // - If x < -lambd: x + bias
        // - If x > lambd: x - bias
        // - Otherwise: 0
        //
        // Using mask_where:
        // 1. Start with zeros
        // 2. Where x < -lambd, use x + bias
        // 3. Where x > lambd, use x - bias

        quote! {
            let #output = {
                let neg_mask = #input.clone().lower_elem(#neg_lambd);
                let pos_mask = #input.clone().greater_elem(#lambd);
                let neg_values = #input.clone().add_scalar(#bias);
                let pos_values = #input.clone().sub_scalar(#bias);
                let zeros = #input.zeros_like();
                zeros.mask_where(neg_mask, neg_values).mask_where(pos_mask, pos_values)
            };
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // All methods used are on Tensor, no additional imports needed
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::shrink::{ShrinkConfig, ShrinkNode, ShrinkNodeBuilder};

    fn create_shrink_node(name: &str, lambd: f64, bias: f64, rank: usize) -> ShrinkNode {
        let config = ShrinkConfig::new(lambd, bias);

        ShrinkNodeBuilder::new(name)
            .input_tensor("input", rank, DType::F32)
            .output_tensor("output", rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_shrink_default() {
        let node = create_shrink_node("shrink1", 0.5, 0.0, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let neg_mask = input.clone().lower_elem(-0.5f64);
                let pos_mask = input.clone().greater_elem(0.5f64);
                let neg_values = input.clone().add_scalar(0f64);
                let pos_values = input.clone().sub_scalar(0f64);
                let zeros = input.zeros_like();
                zeros.mask_where(neg_mask, neg_values).mask_where(pos_mask, pos_values)
            };
            output
        }
        "#);
    }

    #[test]
    fn test_shrink_with_bias() {
        let node = create_shrink_node("shrink1", 1.5, 0.5, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let neg_mask = input.clone().lower_elem(-1.5f64);
                let pos_mask = input.clone().greater_elem(1.5f64);
                let neg_values = input.clone().add_scalar(0.5f64);
                let pos_values = input.clone().sub_scalar(0.5f64);
                let zeros = input.zeros_like();
                zeros.mask_where(neg_mask, neg_values).mask_where(pos_mask, pos_values)
            };
            output
        }
        "#);
    }

    #[test]
    fn test_shrink_3d() {
        let node = create_shrink_node("shrink1", 2.0, 1.0, 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let neg_mask = input.clone().lower_elem(-2f64);
                let pos_mask = input.clone().greater_elem(2f64);
                let neg_values = input.clone().add_scalar(1f64);
                let pos_values = input.clone().sub_scalar(1f64);
                let zeros = input.zeros_like();
                zeros.mask_where(neg_mask, neg_values).mask_where(pos_mask, pos_values)
            };
            output
        }
        "#);
    }

    #[test]
    fn test_shrink_with_clone() {
        let node = create_shrink_node("shrink1", 0.5, 0.0, 2);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let neg_mask = input.clone().clone().lower_elem(-0.5f64);
                let pos_mask = input.clone().clone().greater_elem(0.5f64);
                let neg_values = input.clone().clone().add_scalar(0f64);
                let pos_values = input.clone().clone().sub_scalar(0f64);
                let zeros = input.clone().zeros_like();
                zeros.mask_where(neg_mask, neg_values).mask_where(pos_mask, pos_values)
            };
            output
        }
        "#);
    }
}
