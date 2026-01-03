//! # AffineGrid
//!
//! Generates 2D or 3D flow field (sampling grid) from affine transformation matrices.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__AffineGrid.html>
//!
//! ## Opset Versions
//! - **Opset 20**: Initial version
//!
//! Given a batch of affine transformation matrices theta and a target output size,
//! generates sampling grids for use with grid_sample.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for AffineGrid operation
#[derive(Debug, Clone)]
pub struct AffineGridConfig {
    /// If true, the extrema (-1 and 1) are considered as referring to the
    /// center points of the input's corner pixels. If false, they are
    /// considered as referring to the corner points of the input's corner pixels.
    pub align_corners: bool,
}

/// Node representation for AffineGrid operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct AffineGridNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: AffineGridConfig,
}

pub(crate) struct AffineGridProcessor;

impl NodeProcessor for AffineGridProcessor {
    type Config = AffineGridConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 20,
            max_opset: None,
            inputs: InputSpec::Exact(2),  // theta, size
            outputs: OutputSpec::Exact(1), // grid
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get theta tensor type (input affine transformation matrix)
        let theta_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Output grid has same dtype as theta, rank depends on 2D/3D:
        // - 2D: theta is (N, 2, 3), output is (N, H, W, 2) -> rank 4
        // - 3D: theta is (N, 3, 4), output is (N, D, H, W, 3) -> rank 5
        let output_rank = if theta_tensor.rank == 3 {
            4 // 2D case
        } else {
            5 // 3D case
        };

        let mut output_tensor = theta_tensor;
        output_tensor.rank = output_rank;
        node.outputs[0].ty = ArgType::Tensor(output_tensor);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let align_corners = node
            .attrs
            .get("align_corners")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(false);

        Ok(AffineGridConfig { align_corners })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self.extract_config(&builder, opset).unwrap();
        Node::AffineGrid(AffineGridNode {
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

    #[test]
    fn test_affine_grid_type_inference_2d() {
        let mut node = TestNodeBuilder::new(NodeType::AffineGrid, "test")
            .input_tensor_f32("theta", 3, None) // (N, 2, 3)
            .input_tensor_i64("size", 1, None)  // [N, C, H, W]
            .output_tensor_f32("grid", 4, None) // (N, H, W, 2)
            .build();

        let processor = AffineGridProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 20, &prefs).unwrap();

        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, 4);
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_affine_grid_type_inference_3d() {
        let mut node = TestNodeBuilder::new(NodeType::AffineGrid, "test")
            .input_tensor_f32("theta", 3, None) // (N, 3, 4) - Note: processor uses rank 3 for both
            .input_tensor_i64("size", 1, None)  // [N, C, D, H, W]
            .output_tensor_f32("grid", 5, None) // (N, D, H, W, 3)
            .build();

        let processor = AffineGridProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 20, &prefs).unwrap();

        // With rank 3 theta input, output is 4D (2D case)
        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, 4);
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_affine_grid_config_extraction() {
        let node = TestNodeBuilder::new(NodeType::AffineGrid, "test")
            .input_tensor_f32("theta", 3, None)
            .input_tensor_i64("size", 1, None)
            .output_tensor_f32("grid", 4, None)
            .attr_int("align_corners", 1)
            .build();

        let processor = AffineGridProcessor;
        let config = processor.extract_config(&node, 20).unwrap();

        assert!(config.align_corners);
    }

    #[test]
    fn test_affine_grid_config_default() {
        let node = TestNodeBuilder::new(NodeType::AffineGrid, "test")
            .input_tensor_f32("theta", 3, None)
            .input_tensor_i64("size", 1, None)
            .output_tensor_f32("grid", 4, None)
            .build();

        let processor = AffineGridProcessor;
        let config = processor.extract_config(&node, 20).unwrap();

        assert!(!config.align_corners);
    }
}
