//! ONNX GRU node import implementation.
//!
//! ## Supported ONNX Features
//!
//! - Forward direction only (burn-nn GRU limitation)
//! - Bias support
//! - Initial hidden state
//! - `linear_before_reset` attribute (maps to burn's `reset_after`)
//!
//! ## Unsupported ONNX Features
//!
//! - **Bidirectional/Reverse**: burn-nn GRU only supports forward direction
//! - **Variable sequence lengths**: ONNX input `sequence_lens` not supported
//! - **Custom activations**: burn-nn GRU uses fixed Sigmoid/Tanh activations
//! - **Clip threshold**: burn-nn GRU doesn't support state clipping

use super::prelude::*;
use burn_store::TensorSnapshot;
use onnx_ir::gru::{GruActivationFunction, GruConfig, GruDirection};

/// Collect tensor snapshots for GRU burnpack serialization.
///
/// This function handles the weight transformation from ONNX's packed format
/// to Burn's individual GateController structure.
///
/// ONNX GRU weight layout:
/// - W: `[num_directions, 3*hidden_size, input_size]` - gates ordered as [z, r, h]
/// - R: `[num_directions, 3*hidden_size, hidden_size]` - gates ordered as [z, r, h]
/// - B: `[num_directions, 6*hidden_size]` - Wb[z,r,h] then Rb[z,r,h]
///
/// Burn GRU structure:
/// - update_gate.input_transform: weight `[input_size, hidden_size]`, bias `[hidden_size]`
/// - update_gate.hidden_transform: weight `[hidden_size, hidden_size]`, bias `[hidden_size]`
/// - reset_gate, new_gate: same structure
///
/// Where: z=update_gate, r=reset_gate, h=new_gate
#[allow(clippy::single_range_in_vec_init)]
fn collect_gru_snapshots(
    field_name: &str,
    inputs: &[Argument],
    config: &GruConfig,
) -> Vec<TensorSnapshot> {
    use crate::burn::node_traits::{SerializationBackend, extract_node_data};
    use burn::tensor::Tensor;

    let hidden_size = config.hidden_size;
    let input_size = config.input_size;

    // Extract weight tensors from inputs
    let data_w = extract_node_data(inputs, 1);
    let data_r = extract_node_data(inputs, 2);
    let data_b = extract_node_data(inputs, 3);

    let Some(data_w) = data_w else {
        return vec![];
    };
    let Some(data_r) = data_r else {
        return vec![];
    };

    let dtype = data_w.dtype;
    let device = Default::default();

    // ONNX gate order: z(update), r(reset), h(new/hidden)
    // Burn gate order: update_gate, reset_gate, new_gate
    // So mapping is: z->update(0), r->reset(1), h->new(2) = [0, 1, 2]
    let gate_names = ["update_gate", "reset_gate", "new_gate"];

    let mut snapshots = Vec::new();

    // Create tensors from data
    let w_tensor: Tensor<SerializationBackend, 3> = Tensor::from_data(data_w.clone(), &device);
    let r_tensor: Tensor<SerializationBackend, 3> = Tensor::from_data(data_r.clone(), &device);
    let b_tensor: Option<Tensor<SerializationBackend, 2>> =
        data_b.clone().map(|b| Tensor::from_data(b, &device));

    // For forward-only GRU, dir_idx = 0
    let dir_idx = 0;

    // W shape: [num_directions, 3*hidden_size, input_size]
    let w_dir = w_tensor
        .clone()
        .slice([dir_idx..dir_idx + 1, 0..3 * hidden_size, 0..input_size])
        .squeeze::<2>(); // [3*hidden_size, input_size]

    // R shape: [num_directions, 3*hidden_size, hidden_size]
    let r_dir = r_tensor
        .clone()
        .slice([dir_idx..dir_idx + 1, 0..3 * hidden_size, 0..hidden_size])
        .squeeze::<2>(); // [3*hidden_size, hidden_size]

    // B shape: [num_directions, 6*hidden_size]
    let b_dir = b_tensor.as_ref().map(|b| {
        b.clone()
            .slice([dir_idx..dir_idx + 1, 0..6 * hidden_size])
            .squeeze::<1>() // [6*hidden_size]
    });

    for (gate_idx, gate_name) in gate_names.iter().enumerate() {
        let start = gate_idx * hidden_size;
        let end = start + hidden_size;

        // Input transform weight: slice from W and transpose
        // ONNX: [hidden_size, input_size] -> Burn: [input_size, hidden_size]
        let w_gate = w_dir.clone().slice([start..end, 0..input_size]).transpose();
        let w_gate_data = w_gate.into_data();

        let path = format!("{}.{}.input_transform.weight", field_name, gate_name);
        snapshots.push(create_snapshot_from_data(
            w_gate_data,
            &path,
            "Linear",
            dtype,
        ));

        // Input transform bias: Wb for this gate
        if let Some(ref b) = b_dir {
            let wb_start = gate_idx * hidden_size;
            let wb_end = wb_start + hidden_size;
            let wb: Tensor<SerializationBackend, 1> = b.clone().slice([wb_start..wb_end]);
            let bias_data = wb.into_data();

            let path = format!("{}.{}.input_transform.bias", field_name, gate_name);
            snapshots.push(create_snapshot_from_data(bias_data, &path, "Linear", dtype));
        }

        // Hidden transform weight: slice from R and transpose
        // ONNX: [hidden_size, hidden_size] -> Burn: [hidden_size, hidden_size]
        let r_gate = r_dir
            .clone()
            .slice([start..end, 0..hidden_size])
            .transpose();
        let r_gate_data = r_gate.into_data();

        let path = format!("{}.{}.hidden_transform.weight", field_name, gate_name);
        snapshots.push(create_snapshot_from_data(
            r_gate_data,
            &path,
            "Linear",
            dtype,
        ));

        // Hidden transform bias: Rb for this gate
        if let Some(ref b) = b_dir {
            let rb_start = 3 * hidden_size + gate_idx * hidden_size;
            let rb_end = rb_start + hidden_size;
            let rb: Tensor<SerializationBackend, 1> = b.clone().slice([rb_start..rb_end]);
            let bias_data = rb.into_data();

            let path = format!("{}.{}.hidden_transform.bias", field_name, gate_name);
            snapshots.push(create_snapshot_from_data(bias_data, &path, "Linear", dtype));
        }
    }

    snapshots
}

/// Create a TensorSnapshot from TensorData.
fn create_snapshot_from_data(
    data: burn::tensor::TensorData,
    path: &str,
    container_type: &str,
    dtype: burn::tensor::DType,
) -> TensorSnapshot {
    use burn::module::ParamId;
    use burn_store::TensorSnapshotError;
    use std::rc::Rc;

    let data = data.convert_dtype(dtype);
    let shape = data.shape.clone();
    let path_stack: Vec<String> = path.split('.').map(String::from).collect();
    let container_stack = vec![format!("Struct:{}", container_type)];

    let data_fn = Rc::new(
        move || -> Result<burn::tensor::TensorData, TensorSnapshotError> { Ok(data.clone()) },
    );

    TensorSnapshot::from_closure(
        data_fn,
        dtype,
        shape,
        path_stack,
        container_stack,
        ParamId::new(),
    )
}

impl NodeCodegen for onnx_ir::gru::GruNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // Validate direction
        if self.config.direction != GruDirection::Forward {
            panic!(
                "GRU import only supports forward direction. \
                 burn-nn GRU does not support reverse or bidirectional modes. \
                 Got direction: {:?}",
                self.config.direction
            );
        }

        // Validate activations
        if self.config.gate_activation != GruActivationFunction::Sigmoid {
            panic!(
                "GRU import only supports Sigmoid gate activation. \
                 burn-nn GRU uses fixed Sigmoid activation. \
                 Got: {:?}",
                self.config.gate_activation
            );
        }
        if self.config.hidden_activation != GruActivationFunction::Tanh {
            panic!(
                "GRU import only supports Tanh hidden activation. \
                 burn-nn GRU uses fixed Tanh activation. \
                 Got: {:?}",
                self.config.hidden_activation
            );
        }

        // Validate clip is not used
        if self.config.clip.is_some() {
            panic!(
                "GRU import does not support clip threshold. \
                 burn-nn GRU does not implement state clipping."
            );
        }

        let name = Ident::new(&self.name, Span::call_site());
        let d_input = self.config.input_size.to_tokens();
        let d_hidden = self.config.hidden_size.to_tokens();
        let bias = self.config.has_bias;

        // ONNX linear_before_reset=1 means reset after matmul (PyTorch style)
        // This maps to burn's reset_after=true
        let reset_after = self.config.linear_before_reset;

        Some(Field::new(
            self.name.clone(),
            quote! { Gru<B> },
            quote! {
                let #name = GruConfig::new(#d_input, #d_hidden, #bias)
                    .with_reset_after(#reset_after)
                    .init(device);
            },
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        collect_gru_snapshots(field_name, &self.inputs, &self.config)
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        // Get output variable names
        let output_y = self.outputs.first().map(arg_to_ident);
        let output_y_h = self.outputs.get(1).map(arg_to_ident);

        // Handle initial state if provided
        // Input indices: 0=X, 1=W, 2=R, 3=B, 4=sequence_lens, 5=initial_h
        let has_initial_h = self.config.has_initial_h;

        // ONNX initial_h shape: [num_directions, batch_size, hidden_size]
        // Burn expects: [batch_size, hidden_size]
        let initial_state_expr = if has_initial_h {
            let h_input = scope.arg(&self.inputs[5]);
            // Squeeze out the direction dimension (index 0) for unidirectional GRU
            quote! { Some(#h_input.squeeze_dim(0)) }
        } else {
            quote! { None }
        };

        // Burn GRU expects batch_first input: [batch_size, seq_length, input_size]
        // ONNX default (layout=0): [seq_length, batch_size, input_size]
        // ONNX layout=1 (batch_first): [batch_size, seq_length, input_size]
        let batch_first = self.config.batch_first;

        let input_transform = if batch_first {
            quote! { #input }
        } else {
            // Transpose from [seq, batch, features] to [batch, seq, features]
            quote! { #input.swap_dims(0, 1) }
        };

        let hidden_size = self.config.hidden_size;

        // Burn GRU output: [batch_size, seq_length, hidden_size]
        // ONNX Y output (layout=0): [seq_length, num_directions, batch_size, hidden_size]
        // ONNX Y output (layout=1): [batch_size, seq_length, num_directions, hidden_size]
        // For unidirectional, num_directions=1

        // Y_h output: [num_directions, batch_size, hidden_size]
        // For unidirectional, add dim at index 0

        let y_output_transform = if batch_first {
            // Burn: [batch, seq, hidden] -> ONNX layout=1: [batch, seq, 1, hidden]
            quote! { output.unsqueeze_dim(2) }
        } else {
            // Burn: [batch, seq, hidden] -> swap to [seq, batch, hidden] -> [seq, 1, batch, hidden]
            quote! { output.swap_dims(0, 1).unsqueeze_dim(1) }
        };

        // Y_h: final hidden state
        // Burn output is all states, take last one: [batch, hidden]
        // ONNX expects: [1, batch, hidden]
        let y_h_transform = quote! {
            {
                let [batch_size, seq_len, _hidden] = output.dims();
                output.clone().slice([0..batch_size, (seq_len - 1)..seq_len, 0..#hidden_size])
                    .squeeze_dim::<2>(1)
                    .unsqueeze_dim(0)
            }
        };

        let forward_call = quote! {
            let output = self.#field.forward(#input_transform, #initial_state_expr);
        };

        match (output_y, output_y_h) {
            (Some(y), Some(y_h)) => {
                quote! {
                    let (#y, #y_h) = {
                        #forward_call
                        (
                            #y_output_transform,
                            #y_h_transform
                        )
                    };
                }
            }
            (Some(y), None) => {
                quote! {
                    let #y = {
                        #forward_call
                        #y_output_transform
                    };
                }
            }
            (None, Some(y_h)) => {
                quote! {
                    let #y_h = {
                        #forward_call
                        #y_h_transform
                    };
                }
            }
            (None, None) => {
                // Just run forward, discard output
                quote! {
                    {
                        #forward_call
                    }
                }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::Gru");
        imports.register("burn::nn::GruConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::gru::{GruActivationFunction, GruConfig, GruDirection, GruNode};
    use onnx_ir::ir::{ArgType, Argument, TensorType};

    fn create_gru_node(
        name: &str,
        batch_first: bool,
        has_initial_h: bool,
        num_outputs: usize,
    ) -> GruNode {
        let config = GruConfig::new(
            4,                               // input_size
            8,                               // hidden_size
            GruDirection::Forward,           // direction
            true,                            // has_bias
            has_initial_h,                   // has_initial_h
            batch_first,                     // batch_first
            None,                            // clip
            true,                            // linear_before_reset (reset_after)
            GruActivationFunction::Sigmoid,  // gate_activation
            GruActivationFunction::Tanh,     // hidden_activation
        );

        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
        );
        let w = Argument::new("W", ArgType::Tensor(TensorType::new(DType::F32, 3, None)));
        let r = Argument::new("R", ArgType::Tensor(TensorType::new(DType::F32, 3, None)));
        let b = Argument::new("B", ArgType::Tensor(TensorType::new(DType::F32, 2, None)));

        let mut inputs = vec![input, w, r, b];

        // Add optional inputs only if needed
        if has_initial_h {
            // sequence_lens placeholder (index 4) - not used
            inputs.push(Argument::new(
                "sequence_lens",
                ArgType::Tensor(TensorType::new(DType::I64, 1, None)),
            ));
            // initial_h (index 5)
            inputs.push(Argument::new(
                "initial_h",
                ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
            ));
        }

        let mut outputs = vec![];
        if num_outputs > 0 {
            outputs.push(Argument::new(
                "Y",
                ArgType::Tensor(TensorType::new(DType::F32, 4, None)),
            ));
        }
        if num_outputs > 1 {
            outputs.push(Argument::new(
                "Y_h",
                ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
            ));
        }

        GruNode {
            name: name.to_string(),
            inputs,
            outputs,
            config,
        }
    }

    #[test]
    fn test_gru_forward_basic() {
        let node = create_gru_node("gru1", false, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>) {
            let (Y, Y_h) = {
                let output = self.gru1.forward(input.swap_dims(0, 1), None);
                (
                    output.swap_dims(0, 1).unsqueeze_dim(1),
                    {
                        let [batch_size, seq_len, _hidden] = output.dims();
                        output
                            .clone()
                            .slice([0..batch_size, (seq_len - 1)..seq_len, 0..8usize])
                            .squeeze_dim::<2>(1)
                            .unsqueeze_dim(0)
                    },
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_batch_first() {
        let node = create_gru_node("gru1", true, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>) {
            let (Y, Y_h) = {
                let output = self.gru1.forward(input, None);
                (
                    output.unsqueeze_dim(2),
                    {
                        let [batch_size, seq_len, _hidden] = output.dims();
                        output
                            .clone()
                            .slice([0..batch_size, (seq_len - 1)..seq_len, 0..8usize])
                            .squeeze_dim::<2>(1)
                            .unsqueeze_dim(0)
                    },
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_with_initial_state() {
        let node = create_gru_node("gru1", false, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
            sequence_lens: Tensor<B, 1, Int>,
            initial_h: Tensor<B, 3>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>) {
            let (Y, Y_h) = {
                let output = self
                    .gru1
                    .forward(input.swap_dims(0, 1), Some(initial_h.squeeze_dim(0)));
                (
                    output.swap_dims(0, 1).unsqueeze_dim(1),
                    {
                        let [batch_size, seq_len, _hidden] = output.dims();
                        output
                            .clone()
                            .slice([0..batch_size, (seq_len - 1)..seq_len, 0..8usize])
                            .squeeze_dim::<2>(1)
                            .unsqueeze_dim(0)
                    },
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_y_only() {
        let node = create_gru_node("gru1", false, false, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> Tensor<B, 4> {
            let Y = {
                let output = self.gru1.forward(input.swap_dims(0, 1), None);
                output.swap_dims(0, 1).unsqueeze_dim(1)
            };
            Y
        }
        ");
    }
}
