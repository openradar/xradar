use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NexradError {
    #[error("Invalid volume header: {0}")]
    InvalidVolumeHeader(String),

    #[error("Invalid message header: {0}")]
    InvalidMessageHeader(String),

    #[error("Decompression error: {0}")]
    DecompressionError(String),

    #[error("Invalid record: {0}")]
    InvalidRecord(String),

    #[error("Missing message: {0}")]
    MissingMessage(String),

    #[error("Invalid sweep: {0}")]
    InvalidSweep(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Data not loaded: {0}")]
    DataNotLoaded(String),
}

impl From<NexradError> for PyErr {
    fn from(err: NexradError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
