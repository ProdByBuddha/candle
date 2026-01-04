//! BitLinear layer for BitNet 1.58b
//!
//! Implements 1.58-bit quantization for weights (ternary: -1, 0, 1) 
//! and 8-bit quantization for activations.
//! Supports unpacking from 2-bit packed formats common in BitNet safetensors.

use candle::{Result, Tensor, DType};
use crate::Module;

#[derive(Clone, Debug)]
pub struct BitLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    gamma: f32,
}

impl BitLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        // Pre-calculate gamma: mean(abs(weight))
        let gamma = weight.abs().and_then(|a| a.mean_all()).and_then(|m| m.to_vec0::<f32>()).unwrap_or(1.0);
        Self { weight, bias, gamma }
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Self {
        let gamma = weight.abs().and_then(|a| a.mean_all()).and_then(|m| m.to_vec0::<f32>()).unwrap_or(1.0);
        Self { weight, bias, gamma }
    }

    pub fn load(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Self> {
        let weight = if let Ok(w) = vb.get((out_dim, in_dim), "weight") {
            w
        } else {
            // Check for packed format: [out_dim, in_dim / 4]
            if let Ok(packed) = vb.get((out_dim, in_dim / 4), "weight") {
                Self::unpack_2bit(packed, in_dim, out_dim)?
            } else if let Ok(packed) = vb.get((out_dim / 4, in_dim), "weight") {
                Self::unpack_2bit(packed.t()?, out_dim, in_dim)?.t()?
            } else {
                vb.get((out_dim, in_dim), "weight")?
            }
        };

        let bias = vb.get(out_dim, "bias").ok();
        Ok(Self::new(weight, bias))
    }

    fn unpack_2bit(packed: Tensor, in_dim: usize, out_dim: usize) -> Result<Tensor> {
        let device = packed.device();
        let packed_u8 = packed.to_dtype(DType::U8)?;
        let data = packed_u8.to_vec2::<u8>()?;
        
        let mut unpacked_data = Vec::with_capacity(out_dim * in_dim);
        for row in data {
            for &byte in &row {
                for i in 0..4 {
                    let val = (byte >> (i * 2)) & 0x03;
                    let float_val = match val {
                        0 => -1.0f32,
                        1 => 0.0f32,
                        2 => 1.0f32,
                        _ => 0.0f32,
                    };
                    unpacked_data.push(float_val);
                }
            }
        }
        Tensor::from_vec(unpacked_data, (out_dim, in_dim), device)
    }

    fn quantize_activations(&self, x: &Tensor) -> Result<(Tensor, f32)> {
        let abs_max = x.abs()?.flatten_all()?.max(0)?;
        let max_val = abs_max.to_vec0::<f32>()?.max(1e-5);
        let scale = 127.0 / max_val;
        
        let quantized = x.affine(scale as f64, 0.0)?.clamp(-128.0, 127.0)?.round()?;
        Ok((quantized, max_val / 127.0))
    }
}

impl Module for BitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (x_q, x_scale) = self.quantize_activations(x)?;
        
        // Weights are already ternary, we just need to use them
        let w_t = self.weight.t()?;
        
        let res = match *x_q.dims() {
            [b1, b2, m, k] => {
                if x_q.is_contiguous() {
                    x_q.reshape((b1 * b2 * m, k))?
                        .matmul(&w_t)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let w_br = self.weight.broadcast_left((b1, b2))?.t()?;
                    x_q.matmul(&w_br)?
                }
            }
            [bsize, m, k] => {
                if x_q.is_contiguous() {
                    x_q.reshape((bsize * m, k))?
                        .matmul(&w_t)?
                        .reshape((bsize, m, ()))?
                } else {
                    let w_br = self.weight.broadcast_left(bsize)?.t()?;
                    x_q.matmul(&w_br)?
                }
            }
            _ => x_q.matmul(&w_t)?,
        };

        // Rescale output: y = (x_q @ w_q) * x_scale * gamma
        let final_scale = x_scale * self.gamma;
        let res = res.affine(final_scale as f64, 0.0)?;

        match &self.bias {
            None => Ok(res),
            Some(bias) => res.broadcast_add(bias),
        }
    }
}

pub fn bit_linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<BitLinear> {
    BitLinear::load(in_dim, out_dim, vb)
}

pub fn bit_linear_no_bias(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<BitLinear> {
    BitLinear::load(in_dim, out_dim, vb)
}
