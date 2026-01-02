//! Quantized tensor operations for the tch backend.

use burn_backend::{
    ExecutionError, Shape, TensorData, TensorMetadata,
    ops::QTensorOps,
    quantization::{
        QParams, QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue,
        QuantizationParametersPrimitive, QuantizedBytes,
    },
    tensor::{FloatTensor, IntTensor, QuantizedTensor},
};

use crate::{LibTorch, LibTorchDevice, TchElement, TchQTensor, TchShape, TchTensor};

use super::quantization::{QuantizationStrategy, SymmetricQuantization};
use super::TchOps;

impl<E: TchElement> QTensorOps<Self> for LibTorch<E> {
    fn q_from_data(data: TensorData, device: &LibTorchDevice) -> QuantizedTensor<Self> {
        match data.dtype {
            burn_backend::DType::QFloat(scheme) => {
                let shape = data.shape.clone();
                let num_elements = data.num_elements();
                let q_bytes = QuantizedBytes {
                    bytes: data.into_bytes(),
                    scheme,
                    num_elements,
                };

                match scheme {
                    QuantScheme {
                        level: QuantLevel::Tensor | QuantLevel::Block(_),
                        mode: QuantMode::Symmetric,
                        value:
                            QuantValue::Q8F
                            | QuantValue::Q8S
                            | QuantValue::Q4F
                            | QuantValue::Q4S
                            | QuantValue::Q2F
                            | QuantValue::Q2S
                            | QuantValue::E4M3
                            | QuantValue::E5M2
                            | QuantValue::E2M1,
                        store: QuantStore::Native | QuantStore::U32,
                        ..
                    } => {
                        let (values, qparams_data) = q_bytes.into_vec_i8();

                        // Create i8 tensor
                        let shape_tch = TchShape::from(shape.as_slice());
                        let tensor = tch::Tensor::from_slice(&values)
                            .reshape(&shape_tch.dims)
                            .to((*device).into());

                        let scheme = scheme.with_store(QuantStore::Native);
                        let qparams = qparams_data
                            .scales
                            .into_iter()
                            .map(|scales| QParams { scales })
                            .collect();

                        TchQTensor {
                            qtensor: TchTensor::new(tensor),
                            scheme,
                            qparams,
                        }
                    }
                }
            }
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let shape = tensor.shape();
        let device = tensor.tensor.device();
        let numel = shape.num_elements();

        // Get float data
        let mut data_f = vec![0f32; numel];
        tensor
            .tensor
            .to_kind(tch::Kind::Float)
            .copy_data(&mut data_f, numel);

        // Get scales
        let scales_shape = qparams.scales.shape();
        let scales_numel = scales_shape.num_elements();
        let mut scales = vec![0f32; scales_numel];
        qparams
            .scales
            .tensor
            .to_kind(tch::Kind::Float)
            .copy_data(&mut scales, scales_numel);

        let (values, qparams_vec) = match scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                value:
                    QuantValue::Q8F
                    | QuantValue::Q8S
                    | QuantValue::Q4F
                    | QuantValue::Q4S
                    | QuantValue::Q2F
                    | QuantValue::Q2S
                    | QuantValue::E4M3
                    | QuantValue::E5M2
                    | QuantValue::E2M1,
                store: QuantStore::Native,
                ..
            } => {
                let scale = scales[0];
                let strategy = QuantizationStrategy::PerTensorSymmetric(
                    SymmetricQuantization::init(scale, scheme.value),
                );
                let values = strategy.quantize(&data_f);
                (values, vec![QParams { scales: scale }])
            }
            QuantScheme {
                level: QuantLevel::Block(block_size),
                mode: QuantMode::Symmetric,
                value:
                    QuantValue::Q8F
                    | QuantValue::Q8S
                    | QuantValue::Q4F
                    | QuantValue::Q4S
                    | QuantValue::Q2F
                    | QuantValue::Q2S
                    | QuantValue::E4M3
                    | QuantValue::E5M2
                    | QuantValue::E2M1,
                store: QuantStore::Native,
                ..
            } => {
                let (strategy_vec, qparams): (Vec<_>, Vec<_>) = scales
                    .iter()
                    .map(|&s| {
                        (
                            SymmetricQuantization::init(s, scheme.value),
                            QParams { scales: s },
                        )
                    })
                    .unzip();
                let strategy = QuantizationStrategy::PerBlockSymmetric(strategy_vec, *block_size);
                let values = strategy.quantize(&data_f);
                (values, qparams)
            }
            _ => unimplemented!("Quantization not supported for scheme {scheme:?}"),
        };

        // Create i8 tensor
        let shape_tch = TchShape::from(shape.as_slice());
        let q_tensor = tch::Tensor::from_slice(&values)
            .reshape(&shape_tch.dims)
            .to(device);

        TchQTensor {
            qtensor: TchTensor::new(q_tensor),
            scheme: *scheme,
            qparams: qparams_vec,
        }
    }

    fn quantize_dynamic(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
    ) -> QuantizedTensor<Self> {
        let device = tensor.tensor.device();

        // Compute min/max for calibration
        let min_val = tensor.tensor.min().double_value(&[]) as f32;
        let max_val = tensor.tensor.max().double_value(&[]) as f32;

        // Compute scale for symmetric quantization
        let (a, b) = scheme.value.range();
        let alpha = min_val.abs().max(max_val.abs());
        let mut scale = (alpha + alpha) / (b - a);

        // Avoid division by zero
        if scale == 0.0 {
            scale = 0.1;
        }

        // Create scale tensor
        let scales_tensor = tch::Tensor::from_slice(&[scale]).to(device);
        let qparams = QuantizationParametersPrimitive {
            scales: TchTensor::new(scales_tensor),
        };

        Self::quantize(tensor, scheme, qparams)
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        let shape = tensor.qtensor.shape();
        let device = tensor.qtensor.tensor.device();
        let numel = shape.num_elements();

        // Get i8 values
        let mut values = vec![0i8; numel];
        tensor.qtensor.tensor.copy_data(&mut values, numel);

        // Build strategy and dequantize
        let strategy = build_strategy(&tensor.scheme, &tensor.qparams);
        let float_values = strategy.dequantize(&values);

        // Create float tensor
        let shape_tch = TchShape::from(shape.as_slice());
        let float_tensor = tch::Tensor::from_slice(&float_values)
            .reshape(&shape_tch.dims)
            .to_kind(E::kind())
            .to(device);

        TchTensor::new(float_tensor)
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> LibTorchDevice {
        tensor.qtensor.tensor.device().into()
    }

    fn q_to_device(
        tensor: QuantizedTensor<Self>,
        device: &LibTorchDevice,
    ) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::to_device(tensor.qtensor, device),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::reshape(tensor.qtensor, shape),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        let shape = tensor.qtensor.shape();
        let numel = shape.num_elements();

        // Get i8 values
        let mut values = vec![0i8; numel];
        tensor.qtensor.tensor.copy_data(&mut values, numel);

        let scales: Vec<f32> = tensor.qparams.iter().map(|q| q.scales).collect();

        Ok(TensorData::quantized(values, shape, tensor.scheme, &scales))
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::swap_dims(tensor.qtensor, dim1, dim2),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::permute(tensor.qtensor, axes),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::flip(tensor.qtensor, axes),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::index_select_dim(tensor.qtensor, dim, indices),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_slice(
        tensor: QuantizedTensor<Self>,
        slices: &[burn_backend::Slice],
    ) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::slice_with_steps(tensor.qtensor, slices),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        TchOps::argmax(tensor.qtensor, dim)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        TchOps::argmin(tensor.qtensor, dim)
    }

    fn q_max_dim_with_indices(
        tensor: QuantizedTensor<Self>,
        dim: usize,
    ) -> (QuantizedTensor<Self>, IntTensor<Self>) {
        let (values, indices) = TchOps::max_dim_with_indices(tensor.qtensor, dim);
        (
            TchQTensor {
                qtensor: values,
                scheme: tensor.scheme,
                qparams: tensor.qparams,
            },
            indices,
        )
    }

    fn q_max_dim(tensor: QuantizedTensor<Self>, dim: usize) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::max_dim(tensor.qtensor, dim),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_min_dim(tensor: QuantizedTensor<Self>, dim: usize) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::min_dim(tensor.qtensor, dim),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_min_dim_with_indices(
        tensor: QuantizedTensor<Self>,
        dim: usize,
    ) -> (QuantizedTensor<Self>, IntTensor<Self>) {
        let (values, indices) = TchOps::min_dim_with_indices(tensor.qtensor, dim);
        (
            TchQTensor {
                qtensor: values,
                scheme: tensor.scheme,
                qparams: tensor.qparams,
            },
            indices,
        )
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::expand(tensor.qtensor, shape),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_sort(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        descending: bool,
    ) -> QuantizedTensor<Self> {
        TchQTensor {
            qtensor: TchOps::sort(tensor.qtensor, dim, descending),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_sort_with_indices(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        descending: bool,
    ) -> (QuantizedTensor<Self>, IntTensor<Self>) {
        let (values, indices) = TchOps::sort_with_indices(tensor.qtensor, dim, descending);
        (
            TchQTensor {
                qtensor: values,
                scheme: tensor.scheme,
                qparams: tensor.qparams,
            },
            indices,
        )
    }

    fn q_argsort(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        descending: bool,
    ) -> IntTensor<Self> {
        TchOps::argsort(tensor.qtensor, dim, descending)
    }
}

/// Build a quantization strategy from scheme and parameters.
fn build_strategy(scheme: &QuantScheme, qparams: &[QParams<f32>]) -> QuantizationStrategy {
    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            ..
        } => QuantizationStrategy::PerTensorSymmetric(SymmetricQuantization::init(
            qparams[0].scales,
            scheme.value,
        )),
        QuantScheme {
            level: QuantLevel::Block(block_size),
            mode: QuantMode::Symmetric,
            ..
        } => QuantizationStrategy::PerBlockSymmetric(
            qparams
                .iter()
                .map(|q| SymmetricQuantization::init(q.scales, scheme.value))
                .collect(),
            *block_size,
        ),
    }
}
