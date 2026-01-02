use super::prelude::*;

impl NodeCodegen for onnx_ir::hardmax::HardmaxNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let axis = self.config.axis.to_tokens();

        // Hardmax: argmax along axis, then one_hot encode
        // one_hot_fill(num_classes, on_value=1.0, off_value=0.0, axis)
        quote! {
            let #output = {
                let x = #input;
                let num_classes = x.dims()[#axis];
                let indices = x.argmax(#axis);
                indices.one_hot_fill(num_classes, 1.0f32, 0.0f32, #axis as i64).float()
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::hardmax::{HardmaxConfig, HardmaxNode, HardmaxNodeBuilder};

    fn create_hardmax_node(name: &str, axis: usize) -> HardmaxNode {
        let config = HardmaxConfig::new(axis);

        HardmaxNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_hardmax_forward_last_axis() {
        let node = create_hardmax_node("hardmax1", 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let x = input;
                let num_classes = x.dims()[2];
                let indices = x.argmax(2);
                indices.one_hot_fill(num_classes, 1.0f32, 0.0f32, 2 as i64).float()
            };
            output
        }
        ");
    }

    #[test]
    fn test_hardmax_forward_axis_0() {
        let node = create_hardmax_node("hardmax1", 0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let x = input;
                let num_classes = x.dims()[0];
                let indices = x.argmax(0);
                indices.one_hot_fill(num_classes, 1.0f32, 0.0f32, 0 as i64).float()
            };
            output
        }
        ");
    }

    #[test]
    fn test_hardmax_forward_axis_1() {
        let node = create_hardmax_node("hardmax1", 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let x = input;
                let num_classes = x.dims()[1];
                let indices = x.argmax(1);
                indices.one_hot_fill(num_classes, 1.0f32, 0.0f32, 1 as i64).float()
            };
            output
        }
        ");
    }
}
