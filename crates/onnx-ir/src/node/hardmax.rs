//! # Hardmax
//!
//! Computes hardmax values for each element of the input tensor along the specified axis.
//!
//! The output tensor has the same shape as the input tensor. The element at the
//! argmax position along the specified axis is set to 1, all others are set to 0.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Hardmax.html>
//!
//! ## Type Constraints
//! - T: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with axis=1 default.
//! - **Opset 11**: Changed default axis to -1 (last dimension).
//! - **Opset 13**: Axis is now required to be in the range [-rank, rank-1].
//!
//! **Implementation Note**: This implementation requires opset 13+ and uses the modern behavior.

use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for Hardmax operations
#[derive(Debug, Clone, new)]
pub struct HardmaxConfig {
    /// Axis along which to apply hardmax
    pub axis: usize,
}

/// Node representation for Hardmax operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct HardmaxNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: HardmaxConfig,
}

pub(crate) struct HardmaxProcessor;

impl NodeProcessor for HardmaxProcessor {
    type Config = HardmaxConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 13,
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
        // Output has same shape as input
        crate::processor::same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract the shape of the input tensor
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Extract the axis attribute (default: -1 per ONNX spec opset 11+)
        let mut axis: i64 = -1;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = value.clone().into_i64()
            }
        }

        // if axis is negative, it is counted from the end
        if axis < 0 {
            axis += tensor.rank as i64;
        }

        let config = HardmaxConfig {
            axis: axis as usize,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Hardmax(HardmaxNode {
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

    fn create_test_node(axis: i64, input_rank: usize) -> RawNode {
        TestNodeBuilder::new(NodeType::Hardmax, "test_hardmax")
            .input_tensor_f32("data", input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_hardmax_config_basic() {
        let node = create_test_node(-1, 3);
        let mut node = node;
        let processor = HardmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.axis, 2); // -1 + 3 = 2 (last dimension)
    }

    #[test]
    fn test_hardmax_config_explicit_axis() {
        let node = create_test_node(1, 3);
        let mut node = node;
        let processor = HardmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.axis, 1);
    }
}
