pub mod chunks;
pub mod compression;
pub mod errors;
pub mod messages;
pub mod moments;
pub mod msg2;
pub mod msg5;
pub mod parser;
pub mod python;
pub mod sweep;
pub mod volume_header;

use pyo3::prelude::*;

/// NEXRAD Level2 Rust parser module for xradar
#[pymodule]
fn _nexrad_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::NexradRustFile>()?;
    Ok(())
}
