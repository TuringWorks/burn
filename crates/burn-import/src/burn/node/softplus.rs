use super::prelude::*;

impl NodeCodegen for onnx_ir::elementwise::ElementwiseUnaryNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        match self.node_type {
            onnx_ir::ir::NodeType::Softplus => {
                // ONNX Softplus: log(1 + exp(x)), equivalent to burn's softplus with beta=1.0
                quote! {
                    let #output = burn::tensor::activation::softplus(#input, 1.0);
                }
            }
            onnx_ir::ir::NodeType::Softsign => {
                // Softsign: x / (1 + |x|)
                quote! {
                    let #output = #input.clone() / (#input.abs() + 1);
                }
            }
            onnx_ir::ir::NodeType::Elu => {
                // ELU with default alpha=1.0: x if x > 0, else alpha * (exp(x) - 1)
                quote! {
                    let #output = {
                        let x = #input;
                        let alpha = 1.0f64;
                        let positive = x.clone().clamp_min(0.0);
                        let negative = (x.clone().clamp_max(0.0).exp() - 1).mul_scalar(alpha);
                        positive + negative
                    };
                }
            }
            onnx_ir::ir::NodeType::Selu => {
                // SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
                // Default: alpha = 1.6732632423543772, scale = 1.0507009873554805
                quote! {
                    let #output = {
                        let x = #input;
                        let alpha = 1.6732632423543772f64;
                        let scale = 1.0507009873554805f64;
                        let positive = x.clone().clamp_min(0.0);
                        let negative = (x.clone().clamp_max(0.0).exp() - 1).mul_scalar(alpha);
                        (positive + negative).mul_scalar(scale)
                    };
                }
            }
            onnx_ir::ir::NodeType::Mish => {
                // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
                quote! {
                    let #output = burn::tensor::activation::mish(#input);
                }
            }
            onnx_ir::ir::NodeType::Celu => {
                // CELU with default alpha=1.0: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
                quote! {
                    let #output = {
                        let x = #input;
                        let alpha = 1.0f64;
                        let positive = x.clone().clamp_min(0.0);
                        let negative = (x.clone().div_scalar(alpha).exp() - 1).mul_scalar(alpha).clamp_max(0.0);
                        positive + negative
                    };
                }
            }
            onnx_ir::ir::NodeType::ThresholdedRelu => {
                // ThresholdedRelu with default alpha=1.0: x if x > alpha, else 0
                quote! {
                    let #output = {
                        let x = #input;
                        let alpha = 1.0f64;
                        x.clone().mask_where(x.clone().lower_equal_elem(alpha), x.zeros_like())
                    };
                }
            }
            _ => panic!(
                "Unsupported node type for ElementwiseUnaryNode codegen: {:?}",
                self.node_type
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::elementwise::ElementwiseUnaryNode;
    use onnx_ir::ir::{ArgType, Argument, NodeType, TensorType};

    fn create_unary_node(name: &str, node_type: NodeType) -> ElementwiseUnaryNode {
        let input_ty = ArgType::Tensor(TensorType::new(DType::F32, 2, None));
        let output_ty = ArgType::Tensor(TensorType::new(DType::F32, 2, None));

        ElementwiseUnaryNode {
            name: name.to_string(),
            inputs: vec![Argument::new("input", input_ty)],
            outputs: vec![Argument::new("output", output_ty)],
            node_type,
        }
    }

    #[test]
    fn test_softplus_forward() {
        let node = create_unary_node("softplus1", NodeType::Softplus);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::softplus(input, 1.0);
            output
        }
        ");
    }

    #[test]
    fn test_softsign_forward() {
        let node = create_unary_node("softsign1", NodeType::Softsign);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.clone() / (input.abs() + 1);
            output
        }
        ");
    }

    #[test]
    fn test_elu_forward() {
        let node = create_unary_node("elu1", NodeType::Elu);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let x = input;
                let alpha = 1.0f64;
                let positive = x.clone().clamp_min(0.0);
                let negative = (x.clone().clamp_max(0.0).exp() - 1).mul_scalar(alpha);
                positive + negative
            };
            output
        }
        ");
    }

    #[test]
    fn test_selu_forward() {
        let node = create_unary_node("selu1", NodeType::Selu);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let x = input;
                let alpha = 1.6732632423543772f64;
                let scale = 1.0507009873554805f64;
                let positive = x.clone().clamp_min(0.0);
                let negative = (x.clone().clamp_max(0.0).exp() - 1).mul_scalar(alpha);
                (positive + negative).mul_scalar(scale)
            };
            output
        }
        ");
    }

    #[test]
    fn test_mish_forward() {
        let node = create_unary_node("mish1", NodeType::Mish);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::mish(input);
            output
        }
        ");
    }

    #[test]
    fn test_celu_forward() {
        let node = create_unary_node("celu1", NodeType::Celu);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let x = input;
                let alpha = 1.0f64;
                let positive = x.clone().clamp_min(0.0);
                let negative = (x.clone().div_scalar(alpha).exp() - 1)
                    .mul_scalar(alpha)
                    .clamp_max(0.0);
                positive + negative
            };
            output
        }
        ");
    }

    #[test]
    fn test_thresholded_relu_forward() {
        let node = create_unary_node("thresholded_relu1", NodeType::ThresholdedRelu);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let x = input;
                let alpha = 1.0f64;
                x.clone().mask_where(x.clone().lower_equal_elem(alpha), x.zeros_like())
            };
            output
        }
        ");
    }
}
