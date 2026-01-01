use super::prelude::*;

impl NodeCodegen for onnx_ir::node::max::MaxNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Handle variadic inputs: chain max_pair calls for all inputs
        let mut inputs_iter = self.inputs.iter();
        let first_input = scope.arg(inputs_iter.next().unwrap());

        // Collect remaining inputs
        let remaining: Vec<_> = inputs_iter.map(|arg| scope.arg(arg)).collect();

        if remaining.is_empty() {
            // Single input case: output equals input
            quote! {
                let #output = #first_input;
            }
        } else {
            // Chain max_pair calls for all remaining inputs
            let chain = remaining.iter().fold(
                quote! { #first_input },
                |acc, input| quote! { #acc.max_pair(#input) },
            );
            quote! {
                let #output = #chain;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::max::MaxNodeBuilder;

    #[test]
    fn test_max() {
        let node = MaxNodeBuilder::new("max1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.max_pair(b);
            output
        }
        ");
    }

    #[test]
    fn test_max_variadic_3_inputs() {
        let node = MaxNodeBuilder::new("max1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .input_tensor("c", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2>,
            b: Tensor<B, 2>,
            c: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = a.max_pair(b).max_pair(c);
            output
        }
        ");
    }

    #[test]
    fn test_max_variadic_4_inputs() {
        let node = MaxNodeBuilder::new("max1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .input_tensor("c", 2, DType::F32)
            .input_tensor("d", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2>,
            b: Tensor<B, 2>,
            c: Tensor<B, 2>,
            d: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = a.max_pair(b).max_pair(c).max_pair(d);
            output
        }
        ");
    }
}
