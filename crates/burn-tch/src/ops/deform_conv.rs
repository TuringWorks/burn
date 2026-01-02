//! Deformable convolution implementation for the tch backend.
//!
//! This module implements deformable convolution using pure tensor operations,
//! following the algorithm from "Deformable Convolutional Networks" (DCNv1) and
//! "Deformable ConvNets v2: More Deformable, Better Results" (DCNv2).

use burn_backend::ops::{conv::calculate_conv_output_size, DeformConvOptions};
use burn_backend::TensorMetadata;

use crate::TchTensor;

/// Perform 2D deformable convolution.
///
/// # Arguments
/// * `input` - Input tensor of shape [batch_size, in_channels, height, width]
/// * `offset` - Offset tensor of shape [batch_size, 2 * offset_groups * kernel_h * kernel_w, out_h, out_w]
/// * `weight` - Weight tensor of shape [out_channels, in_channels / groups, kernel_h, kernel_w]
/// * `mask` - Optional mask tensor of shape [batch_size, offset_groups * kernel_h * kernel_w, out_h, out_w]
/// * `bias` - Optional bias tensor of shape [out_channels]
/// * `options` - Convolution options (stride, padding, dilation, groups)
pub fn deform_conv2d(
    input: TchTensor,
    offset: TchTensor,
    weight: TchTensor,
    mask: Option<TchTensor>,
    bias: Option<TchTensor>,
    options: DeformConvOptions<2>,
) -> TchTensor {
    let input_shape = input.shape();
    let weight_shape = weight.shape();

    let batch_size = input_shape.dims[0];
    let in_channels = input_shape.dims[1];
    let in_height = input_shape.dims[2];
    let in_width = input_shape.dims[3];

    let out_channels = weight_shape.dims[0];
    let kernel_h = weight_shape.dims[2];
    let kernel_w = weight_shape.dims[3];

    let groups = options.weight_groups;

    let out_h = calculate_conv_output_size(
        kernel_h,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_height,
    );
    let out_w = calculate_conv_output_size(
        kernel_w,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_width,
    );

    // Perform deformable im2col
    let columns = deform_im2col(
        &input.tensor,
        &offset.tensor,
        mask.as_ref().map(|m| &m.tensor),
        &options,
        (batch_size, in_channels, in_height, in_width),
        (kernel_h, kernel_w),
        (out_h, out_w),
    );

    // columns shape: [in_channels * kernel_h * kernel_w, batch_size * out_h * out_w]
    let col_size_0 = in_channels * kernel_h * kernel_w / groups;
    let col_size_1 = batch_size * out_h * out_w;
    let out_c_per_group = out_channels / groups;

    // Reshape weight and columns for grouped matmul
    let weight_reshaped = weight
        .tensor
        .reshape([groups as i64, out_c_per_group as i64, col_size_0 as i64]);
    let columns_reshaped =
        columns.reshape([groups as i64, col_size_0 as i64, col_size_1 as i64]);

    // Batched matrix multiplication: [groups, out_c_per_group, col_size_0] x [groups, col_size_0, col_size_1]
    // Result: [groups, out_c_per_group, col_size_1]
    let out = weight_reshaped.bmm(&columns_reshaped);

    // Reshape to [out_channels, batch_size, out_h, out_w] then transpose to [batch_size, out_channels, out_h, out_w]
    let mut out = out.reshape([
        out_channels as i64,
        batch_size as i64,
        out_h as i64,
        out_w as i64,
    ]);
    out = out.permute([1, 0, 2, 3]);

    // Add bias if present
    if let Some(bias) = bias {
        let bias_reshaped = bias.tensor.reshape([1, out_channels as i64, 1, 1]);
        out += bias_reshaped;
    }

    TchTensor::new(out)
}

/// Perform deformable im2col transformation.
fn deform_im2col(
    input: &tch::Tensor,
    offset: &tch::Tensor,
    mask: Option<&tch::Tensor>,
    options: &DeformConvOptions<2>,
    input_dims: (usize, usize, usize, usize),
    kernel_dims: (usize, usize),
    out_dims: (usize, usize),
) -> tch::Tensor {
    let (batch_size, in_channels, height, width) = input_dims;
    let (kernel_h, kernel_w) = kernel_dims;
    let (out_h, out_w) = out_dims;
    let offset_groups = options.offset_groups;
    let channels_per_offset_group = in_channels / offset_groups;

    let device = input.device();
    let kind = input.kind();

    // Create output columns tensor
    // Shape: [in_channels, kernel_h, kernel_w, batch_size * out_h * out_w]
    let num_positions = batch_size * out_h * out_w;

    // Create coordinate grids for output positions
    let out_y_coords = tch::Tensor::arange(out_h as i64, (kind, device))
        .reshape([1, out_h as i64, 1])
        .expand([batch_size as i64, out_h as i64, out_w as i64], true);
    let out_x_coords = tch::Tensor::arange(out_w as i64, (kind, device))
        .reshape([1, 1, out_w as i64])
        .expand([batch_size as i64, out_h as i64, out_w as i64], true);

    // Reshape offset: [batch, 2 * offset_groups * kh * kw, out_h, out_w]
    // -> [batch, offset_groups, kh, kw, 2, out_h, out_w]
    let offset_reshaped = offset.reshape([
        batch_size as i64,
        offset_groups as i64,
        kernel_h as i64,
        kernel_w as i64,
        2,
        out_h as i64,
        out_w as i64,
    ]);

    // Reshape mask if present
    let mask_reshaped = mask.map(|m| {
        m.reshape([
            batch_size as i64,
            offset_groups as i64,
            kernel_h as i64,
            kernel_w as i64,
            out_h as i64,
            out_w as i64,
        ])
    });

    // Build columns by iterating over kernel positions
    let mut column_list = Vec::with_capacity(in_channels * kernel_h * kernel_w);

    for in_c in 0..in_channels {
        let group_idx = in_c / channels_per_offset_group;

        for ky in 0..kernel_h {
            for kx in 0..kernel_w {
                // Base position for this kernel element
                let base_y = &out_y_coords * (options.stride[0] as i64)
                    + (ky * options.dilation[0]) as i64
                    - options.padding[0] as i64;
                let base_x = &out_x_coords * (options.stride[1] as i64)
                    + (kx * options.dilation[1]) as i64
                    - options.padding[1] as i64;

                // Add offset
                let offset_y = offset_reshaped.select(1, group_idx as i64).select(1, ky as i64).select(1, kx as i64).select(1, 0);
                let offset_x = offset_reshaped.select(1, group_idx as i64).select(1, ky as i64).select(1, kx as i64).select(1, 1);

                let y = base_y.to_kind(kind) + &offset_y;
                let x = base_x.to_kind(kind) + &offset_x;

                // Bilinear interpolation
                let interpolated =
                    bilinear_interpolate(input, in_c, &y, &x, height, width);

                // Apply mask if present
                let col_value = if let Some(ref mask_r) = mask_reshaped {
                    let mask_val = mask_r.select(1, group_idx as i64).select(1, ky as i64).select(1, kx as i64);
                    interpolated * mask_val
                } else {
                    interpolated
                };

                // Flatten to [batch * out_h * out_w]
                column_list.push(col_value.reshape([num_positions as i64]));
            }
        }
    }

    // Stack all columns: [in_channels * kernel_h * kernel_w, batch * out_h * out_w]
    tch::Tensor::stack(&column_list, 0)
}

/// Perform bilinear interpolation at fractional positions.
fn bilinear_interpolate(
    input: &tch::Tensor,
    channel: usize,
    y: &tch::Tensor,
    x: &tch::Tensor,
    height: usize,
    width: usize,
) -> tch::Tensor {
    let device = input.device();
    let kind = input.kind();

    // Get the channel slice: [batch, height, width]
    let input_c = input.select(1, channel as i64);

    // Compute floor coordinates
    let y_low = y.floor();
    let x_low = x.floor();
    let y_high = &y_low + 1.0;
    let x_high = &x_low + 1.0;

    // Compute interpolation weights
    let ly = y - &y_low;
    let lx = x - &x_low;
    let hy = 1.0 - &ly;
    let hx = 1.0 - &lx;

    let w1 = &hy * &hx;
    let w2 = &hy * &lx;
    let w3 = &ly * &hx;
    let w4 = &ly * &lx;

    // Clamp coordinates for safe indexing
    let height_t = height as i64;
    let width_t = width as i64;

    let y_low_safe = y_low.clamp(0, height_t - 1).to_kind(tch::Kind::Int64);
    let x_low_safe = x_low.clamp(0, width_t - 1).to_kind(tch::Kind::Int64);
    let y_high_safe = y_high.clamp(0, height_t - 1).to_kind(tch::Kind::Int64);
    let x_high_safe = x_high.clamp(0, width_t - 1).to_kind(tch::Kind::Int64);

    // Create validity masks
    let valid_y_low = y_low.ge(0.0).logical_and(&y_low.lt(height as f64));
    let valid_y_high = y_high.ge(0.0).logical_and(&y_high.lt(height as f64));
    let valid_x_low = x_low.ge(0.0).logical_and(&x_low.lt(width as f64));
    let valid_x_high = x_high.ge(0.0).logical_and(&x_high.lt(width as f64));

    let valid_ll = valid_y_low.logical_and(&valid_x_low);
    let valid_lh = valid_y_low.logical_and(&valid_x_high);
    let valid_hl = valid_y_high.logical_and(&valid_x_low);
    let valid_hh = valid_y_high.logical_and(&valid_x_high);

    // Gather values using advanced indexing
    // input_c shape: [batch, height, width]
    // We need to index with [batch_idx, y_idx, x_idx]
    let batch_size = input_c.size()[0];
    let out_shape = y.size();

    // Create batch indices
    let batch_idx = tch::Tensor::arange(batch_size, (tch::Kind::Int64, device))
        .reshape([batch_size, 1, 1])
        .expand(&out_shape, true)
        .reshape([-1]);

    let y_low_flat = y_low_safe.reshape([-1]);
    let x_low_flat = x_low_safe.reshape([-1]);
    let y_high_flat = y_high_safe.reshape([-1]);
    let x_high_flat = x_high_safe.reshape([-1]);

    // Compute linear indices
    let idx_ll = &batch_idx * (height_t * width_t) + &y_low_flat * width_t + &x_low_flat;
    let idx_lh = &batch_idx * (height_t * width_t) + &y_low_flat * width_t + &x_high_flat;
    let idx_hl = &batch_idx * (height_t * width_t) + &y_high_flat * width_t + &x_low_flat;
    let idx_hh = &batch_idx * (height_t * width_t) + &y_high_flat * width_t + &x_high_flat;

    // Flatten input and gather
    let input_flat = input_c.reshape([-1]);

    let v1 = input_flat.index_select(0, &idx_ll).reshape(&out_shape);
    let v2 = input_flat.index_select(0, &idx_lh).reshape(&out_shape);
    let v3 = input_flat.index_select(0, &idx_hl).reshape(&out_shape);
    let v4 = input_flat.index_select(0, &idx_hh).reshape(&out_shape);

    // Apply validity masks
    let zero = tch::Tensor::zeros(&out_shape, (kind, device));
    let v1 = v1.where_self(&valid_ll.to_kind(tch::Kind::Bool), &zero);
    let v2 = v2.where_self(&valid_lh.to_kind(tch::Kind::Bool), &zero);
    let v3 = v3.where_self(&valid_hl.to_kind(tch::Kind::Bool), &zero);
    let v4 = v4.where_self(&valid_hh.to_kind(tch::Kind::Bool), &zero);

    // Weighted sum
    w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
}

/// Backward pass for deformable convolution.
pub mod backward {
    use super::*;
    use burn_backend::ops::DeformConv2dBackward;
    use burn_backend::TensorMetadata;

    /// Calculate gradients for deformable convolution.
    pub fn deform_conv2d_backward<E>(
        input: TchTensor,
        offset: TchTensor,
        weight: TchTensor,
        mask: Option<TchTensor>,
        bias: Option<TchTensor>,
        out_grad: TchTensor,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<crate::LibTorch<E>>
    where
        E: crate::TchElement,
    {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let out_grad_shape = out_grad.shape();

        let batch_size = input_shape.dims[0];
        let in_channels = input_shape.dims[1];
        let in_height = input_shape.dims[2];
        let in_width = input_shape.dims[3];

        let out_channels = weight_shape.dims[0];
        let _in_c_per_group = weight_shape.dims[1];
        let kernel_h = weight_shape.dims[2];
        let kernel_w = weight_shape.dims[3];

        let out_h = out_grad_shape.dims[2];
        let out_w = out_grad_shape.dims[3];

        let groups = options.weight_groups;
        let out_c_per_group = out_channels / groups;
        let col_shape_1 = batch_size * out_h * out_w;

        // Bias gradient: sum over batch, height, width
        let bias_grad = bias.map(|_| {
            let grad = out_grad.tensor.sum_dim_intlist([0i64, 2, 3].as_slice(), false, out_grad.tensor.kind());
            TchTensor::new(grad)
        });

        // Reshape out_grad: [batch, out_c, out_h, out_w] -> [out_c, batch, out_h, out_w]
        // -> [groups, out_c_per_group, batch * out_h * out_w]
        let out_grad_perm = out_grad.tensor.permute([1, 0, 2, 3]);
        let out_grad_reshaped = out_grad_perm.reshape([
            groups as i64,
            out_c_per_group as i64,
            col_shape_1 as i64,
        ]);

        // Compute weight gradient
        let weight_grad = compute_weight_grad(
            &input.tensor,
            &offset.tensor,
            mask.as_ref().map(|m| &m.tensor),
            &out_grad_reshaped,
            &options,
            (batch_size, in_channels, in_height, in_width),
            (kernel_h, kernel_w),
            (out_h, out_w),
        );

        // Compute input and offset gradients
        let (input_grad, offset_grad, mask_grad) = compute_input_offset_grad(
            &input.tensor,
            &weight.tensor,
            &offset.tensor,
            mask.as_ref().map(|m| &m.tensor),
            &out_grad_reshaped,
            &options,
            (batch_size, in_channels, in_height, in_width),
            (kernel_h, kernel_w),
            (out_h, out_w),
        );

        DeformConv2dBackward {
            x_grad: TchTensor::new(input_grad),
            offset_grad: TchTensor::new(offset_grad),
            weight_grad: TchTensor::new(weight_grad),
            mask_grad: mask_grad.map(TchTensor::new),
            bias_grad,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_weight_grad(
        input: &tch::Tensor,
        offset: &tch::Tensor,
        mask: Option<&tch::Tensor>,
        out_grad: &tch::Tensor,
        options: &DeformConvOptions<2>,
        input_dims: (usize, usize, usize, usize),
        kernel_dims: (usize, usize),
        out_dims: (usize, usize),
    ) -> tch::Tensor {
        let (batch_size, in_channels, _height, _width) = input_dims;
        let (kernel_h, kernel_w) = kernel_dims;
        let (out_h, out_w) = out_dims;
        let groups = options.weight_groups;

        // Get columns via im2col
        let columns = super::deform_im2col(
            input,
            offset,
            mask,
            options,
            input_dims,
            kernel_dims,
            out_dims,
        );

        let col_size_0 = in_channels * kernel_h * kernel_w / groups;
        let col_size_1 = batch_size * out_h * out_w;

        // Reshape columns: [in_c * kh * kw, batch * out_h * out_w]
        // -> [groups, col_size_0, col_size_1]
        let columns_reshaped =
            columns.reshape([groups as i64, col_size_0 as i64, col_size_1 as i64]);

        // Transpose columns: [groups, col_size_1, col_size_0]
        let columns_t = columns_reshaped.permute([0, 2, 1]);

        // out_grad: [groups, out_c_per_group, col_size_1]
        // grad_weight = out_grad @ columns_t
        // Result: [groups, out_c_per_group, col_size_0]
        // where col_size_0 = in_c_per_group * kernel_h * kernel_w
        let grad_weight = out_grad.bmm(&columns_t);

        let in_c_per_group = in_channels / groups;
        let out_c_per_group = out_grad.size()[1] as usize;
        let out_channels = groups * out_c_per_group;

        // Reshape from [groups, out_c_per_group, in_c_per_group * kernel_h * kernel_w]
        // to [out_channels, in_c_per_group, kernel_h, kernel_w]
        grad_weight.reshape([
            out_channels as i64,
            in_c_per_group as i64,
            kernel_h as i64,
            kernel_w as i64,
        ])
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_input_offset_grad(
        input: &tch::Tensor,
        weight: &tch::Tensor,
        offset: &tch::Tensor,
        mask: Option<&tch::Tensor>,
        out_grad: &tch::Tensor,
        options: &DeformConvOptions<2>,
        input_dims: (usize, usize, usize, usize),
        kernel_dims: (usize, usize),
        out_dims: (usize, usize),
    ) -> (tch::Tensor, tch::Tensor, Option<tch::Tensor>) {
        let (batch_size, in_channels, _height, _width) = input_dims;
        let (kernel_h, kernel_w) = kernel_dims;
        let (out_h, out_w) = out_dims;
        let groups = options.weight_groups;

        let out_channels = weight.size()[0] as usize;
        let in_c_per_group = in_channels / groups;
        let out_c_per_group = out_channels / groups;
        let col_size_0 = in_c_per_group * kernel_h * kernel_w;

        // Reshape weight: [out_c, in_c_per_group, kh, kw] -> [groups, out_c_per_group, col_size_0]
        let weight_reshaped = weight.reshape([
            groups as i64,
            out_c_per_group as i64,
            col_size_0 as i64,
        ]);

        // Transpose weight: [groups, col_size_0, out_c_per_group]
        let weight_t = weight_reshaped.permute([0, 2, 1]);

        // columns = weight_t @ out_grad
        // [groups, col_size_0, out_c_per_group] @ [groups, out_c_per_group, col_size_1]
        // Result: [groups, col_size_0, col_size_1]
        let columns = weight_t.bmm(out_grad);

        // Reshape columns: [in_channels, kernel_h, kernel_w, batch_size, out_h, out_w]
        let columns = columns.reshape([
            in_channels as i64,
            kernel_h as i64,
            kernel_w as i64,
            batch_size as i64,
            out_h as i64,
            out_w as i64,
        ]);

        // Compute gradients using col2im pattern
        let input_grad = compute_input_grad_col2im(
            &columns, offset, mask, options, input_dims, kernel_dims, out_dims,
        );

        let (offset_grad, mask_grad) = compute_offset_mask_grad(
            input, &columns, offset, mask, options, input_dims, kernel_dims, out_dims,
        );

        (input_grad, offset_grad, mask_grad)
    }

    fn compute_input_grad_col2im(
        columns: &tch::Tensor,
        offset: &tch::Tensor,
        mask: Option<&tch::Tensor>,
        options: &DeformConvOptions<2>,
        input_dims: (usize, usize, usize, usize),
        kernel_dims: (usize, usize),
        out_dims: (usize, usize),
    ) -> tch::Tensor {
        let (batch_size, in_channels, height, width) = input_dims;
        let (kernel_h, kernel_w) = kernel_dims;
        let (out_h, out_w) = out_dims;
        let offset_groups = options.offset_groups;
        let channels_per_offset_group = in_channels / offset_groups;

        let device = columns.device();
        let kind = columns.kind();

        // Initialize input gradient
        let mut input_grad = tch::Tensor::zeros(
            [batch_size as i64, in_channels as i64, height as i64, width as i64],
            (kind, device),
        );

        // Reshape offset
        let offset_reshaped = offset.reshape([
            batch_size as i64,
            offset_groups as i64,
            kernel_h as i64,
            kernel_w as i64,
            2,
            out_h as i64,
            out_w as i64,
        ]);

        let mask_reshaped = mask.map(|m| {
            m.reshape([
                batch_size as i64,
                offset_groups as i64,
                kernel_h as i64,
                kernel_w as i64,
                out_h as i64,
                out_w as i64,
            ])
        });

        // Create coordinate grids
        let out_y_coords = tch::Tensor::arange(out_h as i64, (kind, device))
            .reshape([1, out_h as i64, 1])
            .expand([batch_size as i64, out_h as i64, out_w as i64], true);
        let out_x_coords = tch::Tensor::arange(out_w as i64, (kind, device))
            .reshape([1, 1, out_w as i64])
            .expand([batch_size as i64, out_h as i64, out_w as i64], true);

        // Scatter gradients back to input
        for in_c in 0..in_channels {
            let group_idx = in_c / channels_per_offset_group;

            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let col_val = columns
                        .select(0, in_c as i64)
                        .select(0, ky as i64)
                        .select(0, kx as i64);

                    // Apply mask
                    let col_val = if let Some(ref m) = mask_reshaped {
                        let mask_val = m
                            .select(1, group_idx as i64)
                            .select(1, ky as i64)
                            .select(1, kx as i64);
                        col_val * mask_val
                    } else {
                        col_val
                    };

                    // Compute target positions
                    let base_y = &out_y_coords * (options.stride[0] as i64)
                        + (ky * options.dilation[0]) as i64
                        - options.padding[0] as i64;
                    let base_x = &out_x_coords * (options.stride[1] as i64)
                        + (kx * options.dilation[1]) as i64
                        - options.padding[1] as i64;

                    let offset_y = offset_reshaped
                        .select(1, group_idx as i64)
                        .select(1, ky as i64)
                        .select(1, kx as i64)
                        .select(1, 0);
                    let offset_x = offset_reshaped
                        .select(1, group_idx as i64)
                        .select(1, ky as i64)
                        .select(1, kx as i64)
                        .select(1, 1);

                    let y = base_y.to_kind(kind) + &offset_y;
                    let x = base_x.to_kind(kind) + &offset_x;

                    // Bilinear splatting (reverse of interpolation)
                    bilinear_splat(
                        &mut input_grad,
                        in_c,
                        &y,
                        &x,
                        &col_val,
                        height,
                        width,
                    );
                }
            }
        }

        input_grad
    }

    fn bilinear_splat(
        grad: &mut tch::Tensor,
        channel: usize,
        y: &tch::Tensor,
        x: &tch::Tensor,
        value: &tch::Tensor,
        height: usize,
        width: usize,
    ) {
        let device = grad.device();
        let in_channels = grad.size()[1];

        let y_low = y.floor();
        let x_low = x.floor();
        let y_high = &y_low + 1.0;
        let x_high = &x_low + 1.0;

        let ly = y - &y_low;
        let lx = x - &x_low;
        let hy = 1.0 - &ly;
        let hx = 1.0 - &lx;

        let w1 = &hy * &hx;
        let w2 = &hy * &lx;
        let w3 = &ly * &hx;
        let w4 = &ly * &lx;

        let batch_size = y.size()[0];
        let height_t = height as i64;
        let width_t = width as i64;

        // For each corner, atomically add weighted values
        let y_low_i = y_low.clamp(0, height_t - 1).to_kind(tch::Kind::Int64);
        let x_low_i = x_low.clamp(0, width_t - 1).to_kind(tch::Kind::Int64);
        let y_high_i = y_high.clamp(0, height_t - 1).to_kind(tch::Kind::Int64);
        let x_high_i = x_high.clamp(0, width_t - 1).to_kind(tch::Kind::Int64);

        // Create validity masks
        let valid_y_low = y_low.ge(0.0).logical_and(&y_low.lt(height as f64));
        let valid_y_high = y_high.ge(0.0).logical_and(&y_high.lt(height as f64));
        let valid_x_low = x_low.ge(0.0).logical_and(&x_low.lt(width as f64));
        let valid_x_high = x_high.ge(0.0).logical_and(&x_high.lt(width as f64));

        let v1 = (value * &w1).where_self(
            &valid_y_low.logical_and(&valid_x_low).to_kind(tch::Kind::Bool),
            &tch::Tensor::zeros_like(value),
        );
        let v2 = (value * &w2).where_self(
            &valid_y_low.logical_and(&valid_x_high).to_kind(tch::Kind::Bool),
            &tch::Tensor::zeros_like(value),
        );
        let v3 = (value * &w3).where_self(
            &valid_y_high.logical_and(&valid_x_low).to_kind(tch::Kind::Bool),
            &tch::Tensor::zeros_like(value),
        );
        let v4 = (value * &w4).where_self(
            &valid_y_high.logical_and(&valid_x_high).to_kind(tch::Kind::Bool),
            &tch::Tensor::zeros_like(value),
        );

        // Use scatter_add on the full 4D tensor flattened
        // grad shape: [batch, channels, height, width]
        // We need global indices that include the channel dimension
        let channel_t = channel as i64;
        let spatial_size = height_t * width_t;
        let channel_stride = spatial_size;
        let batch_stride = in_channels * spatial_size;

        let batch_idx = tch::Tensor::arange(batch_size, (tch::Kind::Int64, device))
            .reshape([batch_size, 1, 1])
            .expand(y.size(), true)
            .reshape([-1]);

        let y_low_flat = y_low_i.reshape([-1]);
        let x_low_flat = x_low_i.reshape([-1]);
        let y_high_flat = y_high_i.reshape([-1]);
        let x_high_flat = x_high_i.reshape([-1]);

        // Global indices include batch, channel, y, x
        let idx_ll = &batch_idx * batch_stride + channel_t * channel_stride + &y_low_flat * width_t + &x_low_flat;
        let idx_lh = &batch_idx * batch_stride + channel_t * channel_stride + &y_low_flat * width_t + &x_high_flat;
        let idx_hl = &batch_idx * batch_stride + channel_t * channel_stride + &y_high_flat * width_t + &x_low_flat;
        let idx_hh = &batch_idx * batch_stride + channel_t * channel_stride + &y_high_flat * width_t + &x_high_flat;

        // Flatten the entire grad tensor and scatter_add
        let mut grad_flat = grad.reshape([-1]);
        let _ = grad_flat.scatter_add_(0, &idx_ll, &v1.reshape([-1]));
        let _ = grad_flat.scatter_add_(0, &idx_lh, &v2.reshape([-1]));
        let _ = grad_flat.scatter_add_(0, &idx_hl, &v3.reshape([-1]));
        let _ = grad_flat.scatter_add_(0, &idx_hh, &v4.reshape([-1]));
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_offset_mask_grad(
        input: &tch::Tensor,
        columns: &tch::Tensor,
        offset: &tch::Tensor,
        mask: Option<&tch::Tensor>,
        options: &DeformConvOptions<2>,
        input_dims: (usize, usize, usize, usize),
        kernel_dims: (usize, usize),
        out_dims: (usize, usize),
    ) -> (tch::Tensor, Option<tch::Tensor>) {
        let (batch_size, in_channels, height, width) = input_dims;
        let (kernel_h, kernel_w) = kernel_dims;
        let (out_h, out_w) = out_dims;
        let offset_groups = options.offset_groups;
        let channels_per_offset_group = in_channels / offset_groups;

        let device = input.device();
        let kind = input.kind();

        // Initialize gradients
        let offset_grad = tch::Tensor::zeros(
            [
                batch_size as i64,
                (offset_groups * kernel_h * kernel_w * 2) as i64,
                out_h as i64,
                out_w as i64,
            ],
            (kind, device),
        );

        let mut mask_grad = mask.map(|_| {
            tch::Tensor::zeros(
                [
                    batch_size as i64,
                    (offset_groups * kernel_h * kernel_w) as i64,
                    out_h as i64,
                    out_w as i64,
                ],
                (kind, device),
            )
        });

        // Reshape offset
        let offset_reshaped = offset.reshape([
            batch_size as i64,
            offset_groups as i64,
            kernel_h as i64,
            kernel_w as i64,
            2,
            out_h as i64,
            out_w as i64,
        ]);

        // Create coordinate grids
        let out_y_coords = tch::Tensor::arange(out_h as i64, (kind, device))
            .reshape([1, out_h as i64, 1])
            .expand([batch_size as i64, out_h as i64, out_w as i64], true);
        let out_x_coords = tch::Tensor::arange(out_w as i64, (kind, device))
            .reshape([1, 1, out_w as i64])
            .expand([batch_size as i64, out_h as i64, out_w as i64], true);

        for in_c in 0..in_channels {
            let group_idx = in_c / channels_per_offset_group;

            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let col_val = columns
                        .select(0, in_c as i64)
                        .select(0, ky as i64)
                        .select(0, kx as i64);

                    let base_y = &out_y_coords * (options.stride[0] as i64)
                        + (ky * options.dilation[0]) as i64
                        - options.padding[0] as i64;
                    let base_x = &out_x_coords * (options.stride[1] as i64)
                        + (kx * options.dilation[1]) as i64
                        - options.padding[1] as i64;

                    let offset_y = offset_reshaped
                        .select(1, group_idx as i64)
                        .select(1, ky as i64)
                        .select(1, kx as i64)
                        .select(1, 0);
                    let offset_x = offset_reshaped
                        .select(1, group_idx as i64)
                        .select(1, ky as i64)
                        .select(1, kx as i64)
                        .select(1, 1);

                    let y = base_y.to_kind(kind) + &offset_y;
                    let x = base_x.to_kind(kind) + &offset_x;

                    // Compute coordinate weight gradients
                    let input_c = input.select(1, in_c as i64);
                    let (dy_grad, dx_grad) =
                        get_coordinate_weight_grad(&input_c, &y, &x, height, width);

                    // Mask value
                    let mask_val = mask.map(|m| {
                        let m_reshaped = m.reshape([
                            batch_size as i64,
                            offset_groups as i64,
                            kernel_h as i64,
                            kernel_w as i64,
                            out_h as i64,
                            out_w as i64,
                        ]);
                        m_reshaped
                            .select(1, group_idx as i64)
                            .select(1, ky as i64)
                            .select(1, kx as i64)
                    });

                    let col_masked = if let Some(ref mv) = mask_val {
                        &col_val * mv
                    } else {
                        col_val.shallow_clone()
                    };

                    // Accumulate offset gradients
                    let offset_y_grad_val = &col_masked * &dy_grad;
                    let offset_x_grad_val = &col_masked * &dx_grad;

                    let offset_idx = (group_idx * kernel_h * kernel_w + ky * kernel_w + kx) * 2;

                    // Use index assignment with += semantics
                    let current_y = offset_grad.select(1, offset_idx as i64);
                    let new_y = &current_y + &offset_y_grad_val;
                    offset_grad.select(1, offset_idx as i64).copy_(&new_y);

                    let current_x = offset_grad.select(1, (offset_idx + 1) as i64);
                    let new_x = &current_x + &offset_x_grad_val;
                    offset_grad.select(1, (offset_idx + 1) as i64).copy_(&new_x);

                    // Mask gradient
                    if let Some(ref mut mg) = mask_grad {
                        let interp = super::bilinear_interpolate(input, in_c, &y, &x, height, width);
                        let mask_idx = group_idx * kernel_h * kernel_w + ky * kernel_w + kx;
                        let current_m = mg.select(1, mask_idx as i64);
                        let new_m = &current_m + &(&col_val * &interp);
                        mg.select(1, mask_idx as i64).copy_(&new_m);
                    }
                }
            }
        }

        (offset_grad, mask_grad)
    }

    fn get_coordinate_weight_grad(
        input: &tch::Tensor,
        y: &tch::Tensor,
        x: &tch::Tensor,
        height: usize,
        width: usize,
    ) -> (tch::Tensor, tch::Tensor) {
        let device = input.device();
        let kind = input.kind();

        let y_low = y.floor();
        let x_low = x.floor();
        let y_high = &y_low + 1.0;
        let x_high = &x_low + 1.0;

        let height_t = height as i64;
        let width_t = width as i64;

        let y_low_safe = y_low.clamp(0, height_t - 1).to_kind(tch::Kind::Int64);
        let x_low_safe = x_low.clamp(0, width_t - 1).to_kind(tch::Kind::Int64);
        let y_high_safe = y_high.clamp(0, height_t - 1).to_kind(tch::Kind::Int64);
        let x_high_safe = x_high.clamp(0, width_t - 1).to_kind(tch::Kind::Int64);

        let valid_y_low = y_low.ge(0.0).logical_and(&y_low.lt(height as f64));
        let valid_y_high = y_high.ge(0.0).logical_and(&y_high.lt(height as f64));
        let valid_x_low = x_low.ge(0.0).logical_and(&x_low.lt(width as f64));
        let valid_x_high = x_high.ge(0.0).logical_and(&x_high.lt(width as f64));

        let batch_size = y.size()[0];
        let out_shape = y.size();

        let batch_idx = tch::Tensor::arange(batch_size, (tch::Kind::Int64, device))
            .reshape([batch_size, 1, 1])
            .expand(&out_shape, true)
            .reshape([-1]);

        let y_low_flat = y_low_safe.reshape([-1]);
        let x_low_flat = x_low_safe.reshape([-1]);
        let y_high_flat = y_high_safe.reshape([-1]);
        let x_high_flat = x_high_safe.reshape([-1]);

        let idx_ll = &batch_idx * (height_t * width_t) + &y_low_flat * width_t + &x_low_flat;
        let idx_lh = &batch_idx * (height_t * width_t) + &y_low_flat * width_t + &x_high_flat;
        let idx_hl = &batch_idx * (height_t * width_t) + &y_high_flat * width_t + &x_low_flat;
        let idx_hh = &batch_idx * (height_t * width_t) + &y_high_flat * width_t + &x_high_flat;

        let input_flat = input.reshape([-1]);

        let zero = tch::Tensor::zeros(&out_shape, (kind, device));

        let v_ll = input_flat
            .index_select(0, &idx_ll)
            .reshape(&out_shape)
            .where_self(&valid_y_low.logical_and(&valid_x_low).to_kind(tch::Kind::Bool), &zero);
        let v_lh = input_flat
            .index_select(0, &idx_lh)
            .reshape(&out_shape)
            .where_self(&valid_y_low.logical_and(&valid_x_high).to_kind(tch::Kind::Bool), &zero);
        let v_hl = input_flat
            .index_select(0, &idx_hl)
            .reshape(&out_shape)
            .where_self(&valid_y_high.logical_and(&valid_x_low).to_kind(tch::Kind::Bool), &zero);
        let v_hh = input_flat
            .index_select(0, &idx_hh)
            .reshape(&out_shape)
            .where_self(&valid_y_high.logical_and(&valid_x_high).to_kind(tch::Kind::Bool), &zero);

        let delta_x = x - &x_low;
        let delta_y = y - &y_low;

        // dy = dx * (v_hh - v_lh) + (1 - dx) * (v_hl - v_ll)
        let dy_grad = &delta_x * (&v_hh - &v_lh) + (1.0 - &delta_x) * (&v_hl - &v_ll);

        // dx = dy * (v_hh - v_hl) + (1 - dy) * (v_lh - v_ll)
        let dx_grad = &delta_y * (&v_hh - &v_hl) + (1.0 - &delta_y) * (&v_lh - &v_ll);

        (dy_grad, dx_grad)
    }
}
