//! # Shrink
//!
//! Shrink operator applies soft-thresholding:
//! - If x < -lambd: output = x + bias
//! - If x > lambd: output = x - bias
//! - Otherwise: output = 0
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Shrink.html>
//!
//! ## Type Constraints
//!
//! - T: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64),
//!   tensor(int8), tensor(int16), tensor(int32), tensor(int64),
//!   tensor(float16), tensor(float), tensor(double)
//!
//! ## Opset Versions
//!
//! - **Opset 9+**: Initial version

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
};

/// Configuration for Shrink operation
#[derive(Debug, Clone, new)]
pub struct ShrinkConfig {
    /// The threshold value (default: 0.5)
    pub lambd: f64,
    /// The bias value (default: 0.0)
    pub bias: f64,
}

/// Node representation for Shrink operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ShrinkNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ShrinkConfig,
}

pub(crate) struct ShrinkProcessor;

impl NodeProcessor for ShrinkProcessor {
    type Config = ShrinkConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9,
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
        same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut lambd: f64 = 0.5; // default
        let mut bias: f64 = 0.0; // default

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "lambd" => lambd = value.clone().into_f32() as f64,
                "bias" => bias = value.clone().into_f32() as f64,
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Shrink: {key}"),
                    });
                }
            }
        }

        Ok(ShrinkConfig::new(lambd, bias))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Shrink(ShrinkNode {
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
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node_default() -> RawNode {
        TestNodeBuilder::new(NodeType::Shrink, "test_shrink")
            .input_tensor_f32("X", 2, None)
            .output_tensor_f32("Y", 2, None)
            .build()
    }

    fn create_test_node_with_params(lambd: f32, bias: f32) -> RawNode {
        TestNodeBuilder::new(NodeType::Shrink, "test_shrink")
            .input_tensor_f32("X", 2, None)
            .output_tensor_f32("Y", 2, None)
            .attr_float("lambd", lambd)
            .attr_float("bias", bias)
            .build()
    }

    #[test]
    fn test_shrink_default_config() {
        let node = create_test_node_default();
        let processor = ShrinkProcessor;

        let config = processor.extract_config(&node, 9).unwrap();

        // Default values
        assert!((config.lambd - 0.5).abs() < 1e-6);
        assert!((config.bias - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_shrink_custom_config() {
        let node = create_test_node_with_params(1.5, 0.5);
        let processor = ShrinkProcessor;

        let config = processor.extract_config(&node, 9).unwrap();

        assert!((config.lambd - 1.5).abs() < 1e-6);
        assert!((config.bias - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_shrink_type_inference() {
        let mut node = create_test_node_default();
        let processor = ShrinkProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 9, &prefs).unwrap();

        // Output should have same type as input
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }

    #[test]
    fn test_shrink_build_node() {
        let node = create_test_node_with_params(2.0, 1.0);
        let processor = ShrinkProcessor;

        let result = processor.build_node(node, 9);

        match result {
            Node::Shrink(shrink_node) => {
                assert_eq!(shrink_node.name, "test_shrink");
                assert!((shrink_node.config.lambd - 2.0).abs() < 1e-6);
                assert!((shrink_node.config.bias - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected Shrink node"),
        }
    }
}
