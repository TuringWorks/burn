//! ONNX LpNormalization node import implementation.
//!
//! LpNormalization applies Lp-norm normalization along a specified axis.
//! For p=1 (L1): output = input / sum(|input|, axis)
//! For p=2 (L2): output = input / sqrt(sum(input^2, axis))
//!
//! Maps to burn::tensor::linalg::vector_normalize

use super::prelude::*;

impl NodeCodegen for onnx_ir::node::lp_norm::LpNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Get input rank to handle negative axis
        let rank = match &self.inputs.first().unwrap().ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for LpNormalization"),
        };

        // Convert axis (handle negative axis)
        let axis = if self.config.axis < 0 {
            (rank as i64 + self.config.axis) as usize
        } else {
            self.config.axis as usize
        };

        // Generate norm type based on p value
        let norm = match self.config.p {
            1 => quote! { burn::tensor::linalg::Norm::L1 },
            2 => quote! { burn::tensor::linalg::Norm::L2 },
            _ => panic!("LpNormalization: p must be 1 or 2, got {}", self.config.p),
        };

        // Use vector_normalize with a small epsilon for numerical stability
        let eps = 1e-10f64;

        quote! {
            let #output = burn::tensor::linalg::vector_normalize(#input, #norm, #axis, #eps);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::linalg::vector_normalize");
        imports.register("burn::tensor::linalg::Norm");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::lp_norm::{LpNormConfig, LpNormalizationNode, LpNormalizationNodeBuilder};

    fn create_lp_norm_node(name: &str, axis: i64, p: i64, rank: usize) -> LpNormalizationNode {
        let config = LpNormConfig::new(axis, p);

        LpNormalizationNodeBuilder::new(name)
            .input_tensor("input", rank, DType::F32)
            .output_tensor("output", rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_lp_norm_l2_default() {
        let node = create_lp_norm_node("lpnorm1", -1, 2, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::linalg::vector_normalize(
                input,
                burn::tensor::linalg::Norm::L2,
                1usize,
                0.0000000001f64,
            );
            output
        }
        ");
    }

    #[test]
    fn test_lp_norm_l1() {
        let node = create_lp_norm_node("lpnorm1", 0, 1, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::linalg::vector_normalize(
                input,
                burn::tensor::linalg::Norm::L1,
                0usize,
                0.0000000001f64,
            );
            output
        }
        ");
    }

    #[test]
    fn test_lp_norm_3d() {
        let node = create_lp_norm_node("lpnorm1", 1, 2, 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = burn::tensor::linalg::vector_normalize(
                input,
                burn::tensor::linalg::Norm::L2,
                1usize,
                0.0000000001f64,
            );
            output
        }
        ");
    }

    #[test]
    fn test_lp_norm_with_clone() {
        let node = create_lp_norm_node("lpnorm1", -1, 2, 2);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::linalg::vector_normalize(
                input.clone(),
                burn::tensor::linalg::Norm::L2,
                1usize,
                0.0000000001f64,
            );
            output
        }
        ");
    }
}
