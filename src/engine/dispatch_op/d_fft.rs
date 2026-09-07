//! Dispatch arms: FFT and complex. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Universal FFT & Complex Suite
        "fft" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = node.kwargs.get("n").and_then(|v| v.as_i64());
            let dim = node.kwargs.get("dim").and_then(|v| v.as_i64());
            slots.push(Slot::Owned(fft_complex::fft(&x, n, dim)?));
        }
        "ifft" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = node.kwargs.get("n").and_then(|v| v.as_i64());
            let dim = node.kwargs.get("dim").and_then(|v| v.as_i64());
            slots.push(Slot::Owned(fft_complex::ifft(&x, n, dim)?));
        }
        "rfft" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = node.kwargs.get("n").and_then(|v| v.as_i64());
            let dim = node.kwargs.get("dim").and_then(|v| v.as_i64());
            slots.push(Slot::Owned(fft_complex::rfft(&x, n, dim)?));
        }
        "irfft" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = node.kwargs.get("n").and_then(|v| v.as_i64());
            let dim = node.kwargs.get("dim").and_then(|v| v.as_i64());
            slots.push(Slot::Owned(fft_complex::irfft(&x, n, dim)?));
        }
        "fft2" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::fft2(&x)?));
        }
        "ifft2" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::ifft2(&x)?));
        }
        "fftn" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::fftn(&x)?));
        }
        "ifftn" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::ifftn(&x)?));
        }
        "fftshift" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::fftshift(&x)?));
        }
        "ifftshift" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::ifftshift(&x)?));
        }
        "complex" => {
            let re = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let im = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(fft_complex::complex(&re, &im)?));
        }
        "real" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::real(&x)?));
        }
        "imag" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::imag(&x)?));
        }
        "angle" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::angle(&x)?));
        }
        "polar" => {
            let abs = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ang = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(fft_complex::polar(&abs, &ang)?));
        }
        "conj" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::conj(&x)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
