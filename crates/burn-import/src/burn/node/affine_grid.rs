//! ONNX AffineGrid node import implementation.
//!
//! AffineGrid generates 2D or 3D sampling grids from affine transformation matrices.
//! Currently only 2D grids (theta shape N,2,3) are supported.

use super::prelude::*;
use proc_macro2::TokenStream;
use quote::quote;

impl NodeCodegen for onnx_ir::node::affine_grid::AffineGridNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let theta = scope.arg(self.inputs.first().unwrap());
        let size = scope.arg(self.inputs.get(1).unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let align_corners = self.config.align_corners;

        // Get theta tensor rank to determine 2D vs 3D
        let theta_rank = match &self.inputs.first().unwrap().ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for AffineGrid theta"),
        };

        if theta_rank != 3 {
            panic!("Only 2D AffineGrid (theta rank 3) is currently supported, got rank {theta_rank}");
        }

        if align_corners {
            // Use burn's affine_grid_2d which uses align_corners=true
            quote! {
                let size_data = #size.to_data().to_vec::<i64>().expect("size tensor data");
                let dims = [
                    size_data[0] as usize,  // N
                    size_data[1] as usize,  // C
                    size_data[2] as usize,  // H
                    size_data[3] as usize,  // W
                ];
                let #output = burn::tensor::grid::affine_grid_2d(#theta, dims);
            }
        } else {
            // For align_corners=false, we need a different normalization
            // x = (x_idx + 0.5) * 2 / width - 1.0
            // y = (y_idx + 0.5) * 2 / height - 1.0
            quote! {
                let size_data = #size.to_data().to_vec::<i64>().expect("size tensor data");
                let batch_size = size_data[0] as usize;
                let height = size_data[2] as usize;
                let width = size_data[3] as usize;

                let device = &#theta.device();

                // Create coordinate grids
                let x = burn::tensor::Tensor::<B, 1, burn::tensor::Int>::arange(0..width as i64, device)
                    .reshape([1, width])
                    .expand([height, width]);
                let y = burn::tensor::Tensor::<B, 1, burn::tensor::Int>::arange(0..height as i64, device)
                    .reshape([height, 1])
                    .expand([height, width]);

                // Normalize to (-1, 1) with align_corners=false: (idx + 0.5) * 2 / size - 1
                let x = x.float()
                    .add_scalar(0.5)
                    .mul_scalar(2.0 / width as f32)
                    .sub_scalar(1.0);
                let y = y.float()
                    .add_scalar(0.5)
                    .mul_scalar(2.0 / height as f32)
                    .sub_scalar(1.0);

                // Broadcast to batch dimension
                let x = x.unsqueeze_dim::<3>(0).expand([batch_size, height, width]);
                let y = y.unsqueeze_dim::<3>(0).expand([batch_size, height, width]);

                // Extract affine matrix components
                let a_11 = #theta.clone().slice([0..batch_size, 0..1, 0..1]).squeeze::<1>(1).squeeze::<1>(1);
                let a_12 = #theta.clone().slice([0..batch_size, 0..1, 1..2]).squeeze::<1>(1).squeeze::<1>(1);
                let trans_x = #theta.clone().slice([0..batch_size, 0..1, 2..3]).squeeze::<1>(1).squeeze::<1>(1);
                let a_21 = #theta.clone().slice([0..batch_size, 1..2, 0..1]).squeeze::<1>(1).squeeze::<1>(1);
                let a_22 = #theta.clone().slice([0..batch_size, 1..2, 1..2]).squeeze::<1>(1).squeeze::<1>(1);
                let trans_y = #theta.slice([0..batch_size, 1..2, 2..3]).squeeze::<1>(1).squeeze::<1>(1);

                // Apply affine transform
                let grid_x = a_11.mul(x.clone()).add(a_12.mul(y.clone())).add(trans_x);
                let grid_y = a_21.mul(x).add(a_22.mul(y)).add(trans_y);

                let #output = burn::tensor::Tensor::stack(alloc::vec![grid_x, grid_y], 3);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::affine_grid::{AffineGridConfig, AffineGridNode, AffineGridNodeBuilder};

    fn create_affine_grid_node(name: &str, align_corners: bool) -> AffineGridNode {
        AffineGridNodeBuilder::new(name)
            .input_tensor("theta", 3, DType::F32) // (N, 2, 3)
            .input_tensor("size", 1, DType::I64)  // [N, C, H, W]
            .output_tensor("grid", 4, DType::F32) // (N, H, W, 2)
            .config(AffineGridConfig { align_corners })
            .build()
    }

    #[test]
    fn test_affine_grid_align_corners_true() {
        let node = create_affine_grid_node("affine1", true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, theta: Tensor<B, 3>, size: Tensor<B, 1, Int>) -> Tensor<B, 4> {
            let size_data = size.to_data().to_vec::<i64>().expect("size tensor data");
            let dims = [
                size_data[0] as usize,
                size_data[1] as usize,
                size_data[2] as usize,
                size_data[3] as usize,
            ];
            let grid = burn::tensor::grid::affine_grid_2d(theta, dims);
            grid
        }
        "#);
    }

    #[test]
    fn test_affine_grid_align_corners_false() {
        let node = create_affine_grid_node("affine2", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, theta: Tensor<B, 3>, size: Tensor<B, 1, Int>) -> Tensor<B, 4> {
            let size_data = size.to_data().to_vec::<i64>().expect("size tensor data");
            let batch_size = size_data[0] as usize;
            let height = size_data[2] as usize;
            let width = size_data[3] as usize;
            let device = &theta.device();
            let x = burn::tensor::Tensor::<
                B,
                1,
                burn::tensor::Int,
            >::arange(0..width as i64, device)
                .reshape([1, width])
                .expand([height, width]);
            let y = burn::tensor::Tensor::<
                B,
                1,
                burn::tensor::Int,
            >::arange(0..height as i64, device)
                .reshape([height, 1])
                .expand([height, width]);
            let x = x.float().add_scalar(0.5).mul_scalar(2.0 / width as f32).sub_scalar(1.0);
            let y = y.float().add_scalar(0.5).mul_scalar(2.0 / height as f32).sub_scalar(1.0);
            let x = x.unsqueeze_dim::<3>(0).expand([batch_size, height, width]);
            let y = y.unsqueeze_dim::<3>(0).expand([batch_size, height, width]);
            let a_11 = theta
                .clone()
                .slice([0..batch_size, 0..1, 0..1])
                .squeeze::<1>(1)
                .squeeze::<1>(1);
            let a_12 = theta
                .clone()
                .slice([0..batch_size, 0..1, 1..2])
                .squeeze::<1>(1)
                .squeeze::<1>(1);
            let trans_x = theta
                .clone()
                .slice([0..batch_size, 0..1, 2..3])
                .squeeze::<1>(1)
                .squeeze::<1>(1);
            let a_21 = theta
                .clone()
                .slice([0..batch_size, 1..2, 0..1])
                .squeeze::<1>(1)
                .squeeze::<1>(1);
            let a_22 = theta
                .clone()
                .slice([0..batch_size, 1..2, 1..2])
                .squeeze::<1>(1)
                .squeeze::<1>(1);
            let trans_y = theta
                .slice([0..batch_size, 1..2, 2..3])
                .squeeze::<1>(1)
                .squeeze::<1>(1);
            let grid_x = a_11.mul(x.clone()).add(a_12.mul(y.clone())).add(trans_x);
            let grid_y = a_21.mul(x).add(a_22.mul(y)).add(trans_y);
            let grid = burn::tensor::Tensor::stack(alloc::vec![grid_x, grid_y], 3);
            grid
        }
        "#);
    }
}
