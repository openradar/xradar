use std::fs;
use std::path::Path;

use crate::errors::NexradError;
use crate::volume_header::{VolumeHeader, VOLUME_HEADER_SIZE};

/// Concatenate a list of NEXRAD Level 2 chunk files into a single byte buffer.
///
/// Chunks must be in order: S file first (with AR2V volume header),
/// then I/E chunks in sequence order. Only the S file has a volume header;
/// I/E chunks are raw compressed data that get appended directly.
///
/// Matches the Python `_concatenate_chunks` behavior: simple byte concatenation
/// with validation that at most one chunk has a volume header and it's first.
pub fn concatenate_chunks(chunks: &[Vec<u8>]) -> Result<Vec<u8>, NexradError> {
    if chunks.is_empty() {
        return Err(NexradError::InvalidVolumeHeader(
            "Empty chunk list".to_string(),
        ));
    }

    if chunks.len() == 1 {
        return Ok(chunks[0].clone());
    }

    let prefix = b"AR2V";

    // Find which chunks have volume headers
    let vol_header_indices: Vec<usize> = chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| c.len() >= 4 && &c[..4] == prefix)
        .map(|(i, _)| i)
        .collect();

    if vol_header_indices.len() > 1 {
        return Err(NexradError::InvalidVolumeHeader(format!(
            "Multiple chunks contain a volume header (indices {:?}). \
             Pass chunks from a single volume only.",
            vol_header_indices
        )));
    }

    if vol_header_indices.len() == 1 && vol_header_indices[0] != 0 {
        return Err(NexradError::InvalidVolumeHeader(format!(
            "The chunk with a volume header must be the first item \
             (found at index {}).",
            vol_header_indices[0]
        )));
    }

    // Simple concatenation (matching Python behavior)
    let total_size: usize = chunks.iter().map(|c| c.len()).sum();
    let mut result = Vec::with_capacity(total_size);
    for chunk in chunks {
        result.extend_from_slice(chunk);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_s_chunk(icao: &[u8; 4], extra: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"AR2V0006."); // tape (9 bytes)
        buf.extend_from_slice(b"001"); // extension (3 bytes)
        buf.extend_from_slice(&100u32.to_be_bytes()); // date
        buf.extend_from_slice(&200u32.to_be_bytes()); // time
        buf.extend_from_slice(icao);
        buf.extend_from_slice(extra);
        buf
    }

    fn make_ie_chunk(extra: &[u8]) -> Vec<u8> {
        // I/E chunks have no volume header, just raw compressed data
        extra.to_vec()
    }

    #[test]
    fn test_single_chunk_passthrough() {
        let chunk = make_s_chunk(b"KLOT", b"DATA");
        let result = concatenate_chunks(&[chunk.clone()]).unwrap();
        assert_eq!(result, chunk);
    }

    #[test]
    fn test_s_plus_ie_concatenation() {
        let s_chunk = make_s_chunk(b"KLOT", b"AAA");
        let ie_chunk = make_ie_chunk(b"BBB");
        let result = concatenate_chunks(&[s_chunk.clone(), ie_chunk.clone()]).unwrap();
        assert_eq!(result.len(), s_chunk.len() + ie_chunk.len());
        assert!(result.starts_with(b"AR2V"));
        assert!(result.ends_with(b"BBB"));
    }

    #[test]
    fn test_multiple_vol_headers_rejected() {
        let s1 = make_s_chunk(b"KLOT", b"A");
        let s2 = make_s_chunk(b"KLOT", b"B");
        assert!(concatenate_chunks(&[s1, s2]).is_err());
    }

    #[test]
    fn test_vol_header_not_first_rejected() {
        let ie = make_ie_chunk(b"DATA");
        let s = make_s_chunk(b"KLOT", b"MORE");
        assert!(concatenate_chunks(&[ie, s]).is_err());
    }

    #[test]
    fn test_empty_chunk_list() {
        let empty: Vec<Vec<u8>> = Vec::new();
        assert!(concatenate_chunks(&empty).is_err());
    }

    #[test]
    fn test_all_ie_chunks() {
        // Valid: no volume header at all (all I/E chunks)
        let ie1 = make_ie_chunk(b"AAA");
        let ie2 = make_ie_chunk(b"BBB");
        let result = concatenate_chunks(&[ie1, ie2]).unwrap();
        assert_eq!(result, b"AAABBB");
    }
}
