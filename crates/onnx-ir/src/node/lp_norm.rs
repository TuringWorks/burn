//! # LpNormalization
//!
//! Applies Lp-norm normalization to a tensor along a specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__LpNormalization.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
//! - **Opset 22**: Added bfloat16 support
//!
//! ## Attributes
//! - axis: Axis along which to normalize (default: -1)
//! - p: The order of normalization, only 1 or 2 are supported (default: 2)
//!
//! ## Formula
//! For p=2 (L2 norm): output = input / sqrt(sum(input^2, axis))
//! For p=1 (L1 norm): output = input / sum(|input|, axis)

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for LpNormalization operations
#[derive(Debug, Clone, new)]
pub struct LpNormConfig {
    /// Axis along which to normalize
    pub axis: i64,
    /// The order of normalization (1 or 2)
    pub p: i64,
}

impl LpNormConfig {
    /// Set the axis value
    pub fn with_axis(mut self, axis: i64) -> Self {
        self.axis = axis;
        self
    }

    /// Set the p value
    pub fn with_p(mut self, p: i64) -> Self {
        self.p = p;
        self
    }
}

/// Node representation for LpNormalization operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct LpNormalizationNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: LpNormConfig,
}

pub(crate) struct LpNormProcessor;

impl NodeProcessor for LpNormProcessor {
    type Config = LpNormConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
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
        // Output has the same type as input
        crate::processor::same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut axis: i64 = -1; // default
        let mut p: i64 = 2; // default

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                "p" => p = value.clone().into_i64(),
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for LpNormalization: {key}"),
                    });
                }
            }
        }

        // Validate p value - only 1 and 2 are supported
        if p != 1 && p != 2 {
            return Err(ProcessError::InvalidAttribute {
                name: "p".to_string(),
                reason: format!("LpNormalization: p must be 1 or 2, got {p}"),
            });
        }

        Ok(LpNormConfig::new(axis, p))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::LpNormalization(LpNormalizationNode {
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
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(axis: i64, p: i64) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::LpNormalization, "test_lpnorm")
            .input_tensor_f32("input", 2, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", axis)
            .attr_int("p", p)
    }

    #[test]
    fn test_lp_norm_config_l2_default() {
        let mut node = create_test_node(-1, 2).build();
        let processor = LpNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 1).unwrap();
        processor.infer_types(&mut node, 1, &prefs).unwrap();

        assert_eq!(config.axis, -1);
        assert_eq!(config.p, 2);
    }

    #[test]
    fn test_lp_norm_config_l1() {
        let mut node = create_test_node(0, 1).build();
        let processor = LpNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 1).unwrap();
        processor.infer_types(&mut node, 1, &prefs).unwrap();

        assert_eq!(config.axis, 0);
        assert_eq!(config.p, 1);
    }

    #[test]
    fn test_lp_norm_config_custom_axis() {
        let mut node = create_test_node(1, 2).build();
        let processor = LpNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 1).unwrap();
        processor.infer_types(&mut node, 1, &prefs).unwrap();

        assert_eq!(config.axis, 1);
        assert_eq!(config.p, 2);
    }

    #[test]
    fn test_lp_norm_config_invalid_p() {
        let node = create_test_node(-1, 3).build(); // p=3 is invalid
        let processor = LpNormProcessor;

        let result = processor.extract_config(&node, 1);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_lp_norm_type_inference() {
        let mut node = create_test_node(-1, 2).build();
        let processor = LpNormProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 1, &prefs).unwrap();

        // Output should have same type as input
        assert_eq!(node.outputs[0].ty.elem_type(), DType::F32);
    }
}
