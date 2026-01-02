//! # RMSNormalization
//!
//! RMS (Root Mean Square) Normalization operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__RMSNormalization.html>
//!
//! ## Opset Versions
//! - **Opset 23**: Initial version introducing RMSNormalization operator.
//!   Supports `axis`, `epsilon`, and `stash_type` attributes.
//!
//! **Implementation Note**: This implementation validates opset 23+.
//! The current implementation only supports normalization on the last axis (axis=-1),
//! which is the most common use case.
//!
//! ## Formula
//! `Y = X / sqrt(mean(X^2) + epsilon) * scale`
//!
//! Where:
//! - `X` is the input tensor
//! - `Y` is the output tensor
//! - `scale` is the learnable weight (gamma)
//! - `mean` computes the mean along the normalized axis
//! - `epsilon` is a small value for numerical stability

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for RMSNorm operations
#[derive(Debug, Clone, new)]
pub struct RmsNormConfig {
    /// Number of features/model dimension (size of the last axis)
    pub d_model: usize,
    /// Small constant added for numerical stability (default: 1e-5)
    pub epsilon: f64,
    /// Whether to use full precision for intermediate calculations (stash_type == 1)
    pub full_precision: bool,
}

impl RmsNormConfig {
    /// Set the epsilon value
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the full_precision value
    pub fn with_full_precision(mut self, full_precision: bool) -> Self {
        self.full_precision = full_precision;
        self
    }
}

/// Node representation for RMSNormalization operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct RmsNormalizationNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: RmsNormConfig,
}

pub(crate) struct RmsNormProcessor;

impl NodeProcessor for RmsNormProcessor {
    type Config = RmsNormConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 23,
            max_opset: None,
            inputs: InputSpec::Exact(2), // X, scale
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift scale (input 1) to static
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get scale tensor shape
        let scale_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("RMSNorm: scale tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        // Validate axis attribute
        let mut axis = -1i64;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                "epsilon" | "stash_type" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for RMSNorm: {key}"),
                    });
                }
            }
        }

        // Currently only support axis=-1 (last dimension)
        if axis != -1 && axis != scale_shape.len() as i64 - 1 {
            return Err(ProcessError::Custom(
                "RMSNorm: normalization is only supported on the last axis right now".to_string(),
            ));
        }

        // Output type is same as input
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let scale_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("RMSNorm: scale tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        // d_model is the size of the scale tensor (which matches the last dimension of input)
        let d_model = scale_shape[0];
        let mut epsilon = 1e-5f32;
        let mut stash_type = 1i64; // Default value is 1 (full precision)

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => {} // Already validated in infer_types
                "epsilon" => epsilon = value.clone().into_f32(),
                "stash_type" => stash_type = value.clone().into_i64(),
                _ => {}
            }
        }

        let full_precision = stash_type == 1;
        let config = RmsNormConfig::new(d_model, epsilon as f64, full_precision);
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::RMSNormalization(RmsNormalizationNode {
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

    fn create_test_node(epsilon: f32, axis: i64, stash_type: i64, d_model: usize) -> TestNodeBuilder {
        let scale_data = vec![1.0; d_model]; // Not important for the test

        TestNodeBuilder::new(NodeType::RMSNormalization, "test_rmsnorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", scale_data, vec![d_model])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", epsilon)
            .attr_int("axis", axis)
            .attr_int("stash_type", stash_type)
    }

    #[test]
    fn test_rms_norm_config_basic() {
        let mut node = create_test_node(1e-5, -1, 1, 64).build_with_graph_data(23);
        let processor = RmsNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 23).unwrap();
        processor.infer_types(&mut node, 23, &prefs).unwrap();

        assert_eq!(config.d_model, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(config.full_precision); // stash_type == 1
    }

    #[test]
    fn test_rms_norm_config_custom_epsilon() {
        let mut node = create_test_node(1e-6, -1, 1, 128).build_with_graph_data(23);
        let processor = RmsNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 23).unwrap();
        processor.infer_types(&mut node, 23, &prefs).unwrap();

        assert_eq!(config.d_model, 128);
        assert!(f64::abs(config.epsilon - 1e-6) < 1e-7);
        assert!(config.full_precision);
    }

    #[test]
    fn test_rms_norm_config_no_stash_type() {
        let mut node = create_test_node(1e-5, -1, 0, 32).build_with_graph_data(23);
        let processor = RmsNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 23).unwrap();
        processor.infer_types(&mut node, 23, &prefs).unwrap();

        assert_eq!(config.d_model, 32);
        assert!(!config.full_precision); // stash_type == 0
    }

    #[test]
    fn test_rms_norm_config_invalid_axis() {
        // Create a custom node with a 2D scale tensor to test invalid axis
        let scale_data = vec![1.0; 32 * 64]; // 2D scale tensor

        let node = TestNodeBuilder::new(NodeType::RMSNormalization, "test_rmsnorm_invalid")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", scale_data, vec![32, 64]) // 2D shape
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", 1e-5)
            .attr_int("axis", 0) // axis=0 is NOT the last dimension for 2D scale
            .attr_int("stash_type", 1)
            .build_with_graph_data(23);

        // Now axis=0 should trigger an error since it's not the last dimension (1)
        let mut node = node;
        let processor = RmsNormProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 23, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
