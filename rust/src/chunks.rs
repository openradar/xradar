use std::fs;
use std::path::Path;

use crate::errors::NexradError;
use crate::volume_header::{VolumeHeader, VOLUME_HEADER_SIZE};

/// Concatenate a list of NEXRAD Level 2 chunk files into a single byte buffer.
///
/// Each chunk must start with an AR2V volume header. The ICAO from the first
/// chunk is used as reference; subsequent chunks must match.
///
/// Chunks are sorted by their volume header date+time before concatenation.
pub fn concatenate_chunks(chunks: &[Vec<u8>]) -> Result<Vec<u8>, NexradError> {
    if chunks.is_empty() {
        return Err(NexradError::InvalidVolumeHeader(
            "Empty chunk list".to_string(),
        ));
    }

    if chunks.len() == 1 {
        return Ok(chunks[0].clone());
    }

    // Validate all chunks have AR2V headers
    let mut parsed: Vec<(usize, &Vec<u8>, VolumeHeader)> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let header = VolumeHeader::parse(chunk).map_err(|e| {
            NexradError::InvalidVolumeHeader(format!("Chunk {}: {}", i, e))
        })?;
        parsed.push((i, chunk, header));
    }

    // Validate all chunks have matching ICAO
    let reference_icao = &parsed[0].2.icao;
    for (i, _, header) in &parsed[1..] {
        if header.icao != *reference_icao {
            return Err(NexradError::InvalidVolumeHeader(format!(
                "Chunk {} ICAO {:?} doesn't match reference {:?}",
                i, header.icao, reference_icao
            )));
        }
    }

    // Sort by date, then time
    parsed.sort_by(|a, b| {
        a.2.date
            .cmp(&b.2.date)
            .then(a.2.time.cmp(&b.2.time))
    });

    // Concatenate: first chunk in full, subsequent chunks skip the volume header
    let total_size: usize = parsed[0].1.len()
        + parsed[1..]
            .iter()
            .map(|(_, chunk, _)| chunk.len() - VOLUME_HEADER_SIZE)
            .sum::<usize>();

    let mut result = Vec::with_capacity(total_size);
    result.extend_from_slice(parsed[0].1);

    for (_, chunk, _) in &parsed[1..] {
        result.extend_from_slice(&chunk[VOLUME_HEADER_SIZE..]);
    }

    Ok(result)
}

/// Load chunk files from paths and concatenate them
pub fn load_and_concatenate_chunk_files(paths: &[&Path]) -> Result<Vec<u8>, NexradError> {
    let mut chunks = Vec::with_capacity(paths.len());
    for path in paths {
        let data = fs::read(path)?;
        chunks.push(data);
    }
    concatenate_chunks(&chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(icao: &[u8; 4], date: u32, time: u32, extra: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"AR2V0006."); // tape (9 bytes)
        buf.extend_from_slice(b"001"); // extension (3 bytes)
        buf.extend_from_slice(&date.to_be_bytes());
        buf.extend_from_slice(&time.to_be_bytes());
        buf.extend_from_slice(icao);
        buf.extend_from_slice(extra);
        buf
    }

    #[test]
    fn test_single_chunk_passthrough() {
        let chunk = make_chunk(b"KLOT", 100, 200, b"DATA");
        let result = concatenate_chunks(&[chunk.clone()]).unwrap();
        assert_eq!(result, chunk);
    }

    #[test]
    fn test_two_chunks_concatenation() {
        let chunk1 = make_chunk(b"KLOT", 100, 200, b"AAA");
        let chunk2 = make_chunk(b"KLOT", 100, 300, b"BBB");
        let result = concatenate_chunks(&[chunk1, chunk2]).unwrap();
        // Should be: full chunk1 + chunk2 without header
        assert_eq!(result.len(), 27 + 3); // 24+3 + 3
        assert!(result.ends_with(b"BBB"));
    }

    #[test]
    fn test_mismatched_icao_rejected() {
        let chunk1 = make_chunk(b"KLOT", 100, 200, b"A");
        let chunk2 = make_chunk(b"KATX", 100, 300, b"B");
        assert!(concatenate_chunks(&[chunk1, chunk2]).is_err());
    }

    #[test]
    fn test_empty_chunk_list() {
        let empty: Vec<Vec<u8>> = Vec::new();
        assert!(concatenate_chunks(&empty).is_err());
    }

    #[test]
    fn test_chunks_sorted_by_time() {
        let chunk_late = make_chunk(b"KLOT", 100, 500, b"LATE");
        let chunk_early = make_chunk(b"KLOT", 100, 100, b"EARLY");
        let result = concatenate_chunks(&[chunk_late, chunk_early]).unwrap();
        // Early chunk should come first
        // Its extension bytes will be at position 9-12
        // The first chunk in result should be the early one
        let header = VolumeHeader::parse(&result).unwrap();
        assert_eq!(
            u32::from_be_bytes(result[16..20].try_into().unwrap()),
            100 // time of early chunk
        );
    }
}
