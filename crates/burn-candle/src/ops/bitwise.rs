//! Bitwise operations for the Candle backend.
//!
//! Candle doesn't have native bitwise tensor operations, so we implement them
//! by extracting data, performing operations on raw values, and recreating tensors.

use crate::CandleTensor;
use burn_backend::{Element, TensorMetadata};

/// Apply a binary bitwise operation on two tensors with broadcasting.
fn bitwise_binary_op<F64Op, U32Op, U8Op>(
    lhs: CandleTensor,
    rhs: CandleTensor,
    op_i64: F64Op,
    op_u32: U32Op,
    op_u8: U8Op,
) -> CandleTensor
where
    F64Op: Fn(i64, i64) -> i64,
    U32Op: Fn(u32, u32) -> u32,
    U8Op: Fn(u8, u8) -> u8,
{
    let dtype = lhs.tensor.dtype();

    // Handle broadcasting
    let broadcast_shape = lhs
        .tensor
        .shape()
        .broadcast_shape_binary_op(rhs.tensor.shape(), "bitwise_op")
        .unwrap();

    let lhs_broadcast = if *lhs.tensor.shape() == broadcast_shape {
        lhs.tensor
    } else {
        lhs.tensor.broadcast_as(broadcast_shape.clone()).unwrap()
    };

    let rhs_broadcast = if *rhs.tensor.shape() == broadcast_shape {
        rhs.tensor
    } else {
        rhs.tensor.broadcast_as(broadcast_shape.clone()).unwrap()
    };

    let shape = broadcast_shape.dims();
    let device = lhs_broadcast.device();

    match dtype {
        candle_core::DType::I64 => {
            let lhs_vec: Vec<i64> = lhs_broadcast.flatten_all().unwrap().to_vec1().unwrap();
            let rhs_vec: Vec<i64> = rhs_broadcast.flatten_all().unwrap().to_vec1().unwrap();
            let result: Vec<i64> = lhs_vec
                .iter()
                .zip(rhs_vec.iter())
                .map(|(&l, &r)| op_i64(l, r))
                .collect();
            CandleTensor::new(candle_core::Tensor::from_vec(result, shape, device).unwrap())
        }
        candle_core::DType::U32 => {
            let lhs_vec: Vec<u32> = lhs_broadcast.flatten_all().unwrap().to_vec1().unwrap();
            let rhs_vec: Vec<u32> = rhs_broadcast.flatten_all().unwrap().to_vec1().unwrap();
            let result: Vec<u32> = lhs_vec
                .iter()
                .zip(rhs_vec.iter())
                .map(|(&l, &r)| op_u32(l, r))
                .collect();
            CandleTensor::new(candle_core::Tensor::from_vec(result, shape, device).unwrap())
        }
        candle_core::DType::U8 => {
            let lhs_vec: Vec<u8> = lhs_broadcast.flatten_all().unwrap().to_vec1().unwrap();
            let rhs_vec: Vec<u8> = rhs_broadcast.flatten_all().unwrap().to_vec1().unwrap();
            let result: Vec<u8> = lhs_vec
                .iter()
                .zip(rhs_vec.iter())
                .map(|(&l, &r)| op_u8(l, r))
                .collect();
            CandleTensor::new(candle_core::Tensor::from_vec(result, shape, device).unwrap())
        }
        _ => panic!("Bitwise operations only supported for integer types (I64, U32, U8)"),
    }
}

/// Apply a scalar bitwise operation.
fn bitwise_scalar_op<E, F64Op, U32Op, U8Op>(
    lhs: CandleTensor,
    rhs: E,
    op_i64: F64Op,
    op_u32: U32Op,
    op_u8: U8Op,
) -> CandleTensor
where
    E: Element,
    F64Op: Fn(i64, i64) -> i64,
    U32Op: Fn(u32, u32) -> u32,
    U8Op: Fn(u8, u8) -> u8,
{
    let dtype = lhs.tensor.dtype();
    let shape = lhs.shape();
    let device = lhs.tensor.device();

    match dtype {
        candle_core::DType::I64 => {
            let rhs_val: i64 = rhs.elem();
            let lhs_vec: Vec<i64> = lhs.tensor.flatten_all().unwrap().to_vec1().unwrap();
            let result: Vec<i64> = lhs_vec.iter().map(|&l| op_i64(l, rhs_val)).collect();
            CandleTensor::new(
                candle_core::Tensor::from_vec(result, shape.dims, device).unwrap(),
            )
        }
        candle_core::DType::U32 => {
            let rhs_val: u32 = rhs.elem();
            let lhs_vec: Vec<u32> = lhs.tensor.flatten_all().unwrap().to_vec1().unwrap();
            let result: Vec<u32> = lhs_vec.iter().map(|&l| op_u32(l, rhs_val)).collect();
            CandleTensor::new(
                candle_core::Tensor::from_vec(result, shape.dims, device).unwrap(),
            )
        }
        candle_core::DType::U8 => {
            let rhs_val: u8 = rhs.elem();
            let lhs_vec: Vec<u8> = lhs.tensor.flatten_all().unwrap().to_vec1().unwrap();
            let result: Vec<u8> = lhs_vec.iter().map(|&l| op_u8(l, rhs_val)).collect();
            CandleTensor::new(
                candle_core::Tensor::from_vec(result, shape.dims, device).unwrap(),
            )
        }
        _ => panic!("Bitwise operations only supported for integer types (I64, U32, U8)"),
    }
}

/// Apply a unary bitwise operation.
fn bitwise_unary_op<F64Op, U32Op, U8Op>(
    tensor: CandleTensor,
    op_i64: F64Op,
    op_u32: U32Op,
    op_u8: U8Op,
) -> CandleTensor
where
    F64Op: Fn(i64) -> i64,
    U32Op: Fn(u32) -> u32,
    U8Op: Fn(u8) -> u8,
{
    let dtype = tensor.tensor.dtype();
    let shape = tensor.shape();
    let device = tensor.tensor.device();

    match dtype {
        candle_core::DType::I64 => {
            let vec: Vec<i64> = tensor.tensor.flatten_all().unwrap().to_vec1().unwrap();
            let result: Vec<i64> = vec.iter().map(|&v| op_i64(v)).collect();
            CandleTensor::new(
                candle_core::Tensor::from_vec(result, shape.dims, device).unwrap(),
            )
        }
        candle_core::DType::U32 => {
            let vec: Vec<u32> = tensor.tensor.flatten_all().unwrap().to_vec1().unwrap();
            let result: Vec<u32> = vec.iter().map(|&v| op_u32(v)).collect();
            CandleTensor::new(
                candle_core::Tensor::from_vec(result, shape.dims, device).unwrap(),
            )
        }
        candle_core::DType::U8 => {
            let vec: Vec<u8> = tensor.tensor.flatten_all().unwrap().to_vec1().unwrap();
            let result: Vec<u8> = vec.iter().map(|&v| op_u8(v)).collect();
            CandleTensor::new(
                candle_core::Tensor::from_vec(result, shape.dims, device).unwrap(),
            )
        }
        _ => panic!("Bitwise operations only supported for integer types (I64, U32, U8)"),
    }
}

/// Bitwise AND of two tensors.
pub fn bitwise_and(lhs: CandleTensor, rhs: CandleTensor) -> CandleTensor {
    bitwise_binary_op(lhs, rhs, |a, b| a & b, |a, b| a & b, |a, b| a & b)
}

/// Bitwise AND with a scalar.
pub fn bitwise_and_scalar<E: Element>(lhs: CandleTensor, rhs: E) -> CandleTensor {
    bitwise_scalar_op(lhs, rhs, |a, b| a & b, |a, b| a & b, |a, b| a & b)
}

/// Bitwise OR of two tensors.
pub fn bitwise_or(lhs: CandleTensor, rhs: CandleTensor) -> CandleTensor {
    bitwise_binary_op(lhs, rhs, |a, b| a | b, |a, b| a | b, |a, b| a | b)
}

/// Bitwise OR with a scalar.
pub fn bitwise_or_scalar<E: Element>(lhs: CandleTensor, rhs: E) -> CandleTensor {
    bitwise_scalar_op(lhs, rhs, |a, b| a | b, |a, b| a | b, |a, b| a | b)
}

/// Bitwise XOR of two tensors.
pub fn bitwise_xor(lhs: CandleTensor, rhs: CandleTensor) -> CandleTensor {
    bitwise_binary_op(lhs, rhs, |a, b| a ^ b, |a, b| a ^ b, |a, b| a ^ b)
}

/// Bitwise XOR with a scalar.
pub fn bitwise_xor_scalar<E: Element>(lhs: CandleTensor, rhs: E) -> CandleTensor {
    bitwise_scalar_op(lhs, rhs, |a, b| a ^ b, |a, b| a ^ b, |a, b| a ^ b)
}

/// Bitwise NOT of a tensor.
pub fn bitwise_not(tensor: CandleTensor) -> CandleTensor {
    bitwise_unary_op(tensor, |a| !a, |a| !a, |a| !a)
}

/// Bitwise left shift of two tensors.
pub fn bitwise_left_shift(lhs: CandleTensor, rhs: CandleTensor) -> CandleTensor {
    bitwise_binary_op(
        lhs,
        rhs,
        |a, b| a << (b as u32),
        |a, b| a << b,
        |a, b| a << b,
    )
}

/// Bitwise left shift with a scalar.
pub fn bitwise_left_shift_scalar<E: Element>(lhs: CandleTensor, rhs: E) -> CandleTensor {
    bitwise_scalar_op(
        lhs,
        rhs,
        |a, b| a << (b as u32),
        |a, b| a << b,
        |a, b| a << b,
    )
}

/// Bitwise right shift of two tensors.
pub fn bitwise_right_shift(lhs: CandleTensor, rhs: CandleTensor) -> CandleTensor {
    bitwise_binary_op(
        lhs,
        rhs,
        |a, b| a >> (b as u32),
        |a, b| a >> b,
        |a, b| a >> b,
    )
}

/// Bitwise right shift with a scalar.
pub fn bitwise_right_shift_scalar<E: Element>(lhs: CandleTensor, rhs: E) -> CandleTensor {
    bitwise_scalar_op(
        lhs,
        rhs,
        |a, b| a >> (b as u32),
        |a, b| a >> b,
        |a, b| a >> b,
    )
}
