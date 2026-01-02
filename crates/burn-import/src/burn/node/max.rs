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
        let first_arg = inputs_iter.next().unwrap();
        let first_input = scope.arg(first_arg);
        let first_rank = match &first_arg.ty {
            ArgType::Tensor(t) => t.rank,
            _ => 0,
        };

        // Collect remaining inputs with their ranks
        let remaining: Vec<_> = inputs_iter
            .map(|arg| {
                let input = scope.arg(arg);
                let rank = match &arg.ty {
                    ArgType::Tensor(t) => t.rank,
                    _ => 0,
                };
                (input, rank)
            })
            .collect();

        if remaining.is_empty() {
            // Single input case: output equals input
            quote! {
                let #output = #first_input;
            }
        } else {
            // Chain max_pair calls for all remaining inputs with broadcasting support
            let output_rank = self
                .outputs
                .first()
                .and_then(|o| match &o.ty {
                    ArgType::Tensor(t) => Some(t.rank),
                    _ => None,
                })
                .unwrap_or(first_rank);

            // Broadcast first input if needed
            let first_broadcast = if first_rank < output_rank {
                let num_dims = output_rank - first_rank;
                let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                quote! { #first_input.unsqueeze_dims(&[#(#dims),*]) }
            } else {
                quote! { #first_input }
            };

            let chain = remaining.iter().fold(first_broadcast, |acc, (input, rank)| {
                if *rank < output_rank {
                    let num_dims = output_rank - rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #acc.max_pair(#input.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    quote! { #acc.max_pair(#input) }
                }
            });
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

    #[test]
    fn test_max_broadcast() {
        let node = MaxNodeBuilder::new("max1")
            .input_tensor("a", 3, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = a.max_pair(b.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    #[test]
    fn test_max_broadcast_first_input() {
        let node = MaxNodeBuilder::new("max1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = a.unsqueeze_dims(&[0isize]).max_pair(b);
            output
        }
        ");
    }
}
