//! Quantization strategy implementations for the tch backend.
//!
//! This module provides symmetric quantization for per-tensor and per-block quantization schemes.

use num_traits::{Float, PrimInt};

use burn_backend::quantization::{BlockSize, QuantValue};

/// Quantization strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizationStrategy {
    /// Per-tensor symmetric quantization.
    PerTensorSymmetric(SymmetricQuantization<f32>),
    /// Per-block symmetric quantization.
    PerBlockSymmetric(Vec<SymmetricQuantization<f32>>, BlockSize),
}

impl QuantizationStrategy {
    /// Quantize the values to a lower precision data type.
    pub fn quantize(&self, values: &[f32]) -> Vec<i8> {
        match self {
            QuantizationStrategy::PerTensorSymmetric(strategy) => strategy.quantize(values),
            QuantizationStrategy::PerBlockSymmetric(strategy, block_size) => {
                let block_elems = block_size.num_elements();
                let num_blocks = strategy.len();
                let numel = values.len();
                assert_eq!(
                    numel / block_elems,
                    num_blocks,
                    "Invalid per-block quantization with num blocks {num_blocks} and {numel} values"
                );
                values
                    .chunks(block_elems)
                    .enumerate()
                    .flat_map(|(block_id, block)| strategy[block_id].quantize(block))
                    .collect()
            }
        }
    }

    /// Dequantize the values to a higher precision data type.
    pub fn dequantize(&self, values: &[i8]) -> Vec<f32> {
        match self {
            QuantizationStrategy::PerTensorSymmetric(strategy) => strategy.dequantize(values),
            QuantizationStrategy::PerBlockSymmetric(strategy, block_size) => {
                let block_elems = block_size.num_elements();
                let num_blocks = strategy.len();
                let numel = values.len();
                assert_eq!(
                    numel / block_elems,
                    num_blocks,
                    "Invalid per-block quantization with block size {block_elems}, num blocks {num_blocks} and {numel} values"
                );
                values
                    .chunks(block_elems)
                    .enumerate()
                    .flat_map(|(block_id, block)| strategy[block_id].dequantize(block))
                    .collect()
            }
        }
    }
}

/// Quantization scheme to convert elements of a higher precision data type `E` to a lower precision
/// data type `Q` and vice-versa.
pub trait Quantization<E: Float + Send + Sync> {
    /// Returns the quantization range `[a, b]`.
    fn range(&self) -> (E, E);
    /// Convert the values to a lower precision data type.
    fn quantize<Q: PrimInt>(&self, values: &[E]) -> Vec<Q>;
    /// Convert a single value to a lower precision data type.
    fn quantize_one<Q: PrimInt>(&self, value: E) -> Q;
    /// Convert the values back to a higher precision data type.
    fn dequantize<Q: PrimInt>(&self, values: &[Q]) -> Vec<E>;
    /// Convert a single value back to a higher precision data type.
    fn dequantize_one<Q: PrimInt>(&self, value: Q) -> E;
}

fn valid_scale<E: Float>(mut scale: E) -> E {
    // If scale is 0 (most likely due to a tensor full of zeros), we arbitrarily adjust the
    // scale to 0.1 to avoid division by zero.
    if scale.eq(&E::zero()) {
        scale = E::from(0.1).unwrap();
    }
    scale
}

/// Symmetric quantization scheme.
#[derive(Debug, Clone, Copy)]
pub struct SymmetricQuantization<E: Float + Send + Sync> {
    /// The scaling factor.
    pub scale: E,
    /// The quantization value data type.
    value: QuantValue,
}

impl<E: Float + Send + Sync> SymmetricQuantization<E> {
    /// Initialize a symmetric quantization scheme with the given parameters.
    pub fn init(scale: E, value: QuantValue) -> Self {
        Self {
            scale: valid_scale(scale),
            value,
        }
    }
}

impl<E: Float + Send + Sync> Quantization<E> for SymmetricQuantization<E> {
    fn quantize<Q: PrimInt>(&self, values: &[E]) -> Vec<Q> {
        values.iter().map(|x| self.quantize_one(*x)).collect()
    }

    fn dequantize<Q: PrimInt>(&self, values: &[Q]) -> Vec<E> {
        values.iter().map(|x_q| self.dequantize_one(*x_q)).collect()
    }

    fn quantize_one<Q: PrimInt>(&self, value: E) -> Q {
        let (a, b) = self.range();

        // x_q = clamp(round(x / scale), a, b)
        Q::from(value.div(self.scale).round().clamp(a, b)).unwrap()
    }

    fn dequantize_one<Q: PrimInt>(&self, value: Q) -> E {
        // x = scale * x_q
        self.scale * E::from(value).unwrap()
    }

    fn range(&self) -> (E, E) {
        let (a, b) = self.value.range();
        let a = E::from(a).unwrap();
        let b = E::from(b).unwrap();
        (a, b)
    }
}

impl<E: Float + Send + Sync> PartialEq for SymmetricQuantization<E> {
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale
    }
}

impl<E: Float + Send + Sync> Eq for SymmetricQuantization<E> {}
