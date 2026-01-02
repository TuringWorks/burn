//! # Swish
//!
//! Applies the Swish activation function element-wise.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Swish.html>
//!
//! ## Formula
//! ```text
//! y = x * sigmoid(alpha * x)
//! ```
//!
//! When alpha = 1, this is equivalent to SiLU (Sigmoid Linear Unit).
//!
//! ## Type Constraints
//! - `T`: float16, float32, float64, bfloat16
//!
//! ## Opset Versions
//! - **Opset 24**: Initial version

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for Swish operation
#[derive(Debug, Clone, new)]
pub struct SwishConfig {
    /// Coefficient to multiply with input before sigmoid (default: 1.0)
    pub alpha: f32,
}

/// Node representation for Swish operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SwishNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SwishConfig,
}

pub(crate) struct SwishProcessor;

impl NodeProcessor for SwishProcessor {
    type Config = SwishConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 24,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Output type is same as input
        crate::processor::same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let alpha = node
            .attrs
            .get("alpha")
            .map(|v| v.clone().into_f32())
            .unwrap_or(1.0);

        Ok(SwishConfig::new(alpha))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Swish(SwishNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;
    use burn_tensor::DType;

    fn create_test_node() -> RawNode {
        TestNodeBuilder::new(NodeType::Swish, "test_swish")
            .input_tensor_f32("X", 4, Some(vec![1, 3, 224, 224]))
            .output_tensor_f32("Y", 0, None)
            .build()
    }

    #[test]
    fn test_swish_type_inference() {
        let mut node = create_test_node();
        let processor = SwishProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 24, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
                assert_eq!(tensor.static_shape, Some(vec![1, 3, 224, 224]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_swish_default_alpha() {
        let node = create_test_node();
        let processor = SwishProcessor;
        let config = processor.extract_config(&node, 24).unwrap();
        assert_eq!(config.alpha, 1.0);
    }

    #[test]
    fn test_swish_custom_alpha() {
        let mut node = create_test_node();
        node.attrs.insert(
            "alpha".to_string(),
            crate::ir::AttributeValue::Float32(0.5),
        );
        let processor = SwishProcessor;
        let config = processor.extract_config(&node, 24).unwrap();
        assert_eq!(config.alpha, 0.5);
    }
}
