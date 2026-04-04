use bzip2::read::BzDecoder;
use std::collections::HashMap;
use std::io::Read;

use crate::errors::NexradError;
use crate::messages::RECORD_BYTES;

/// BZ2 magic bytes
const BZ2_MAGIC: &[u8] = b"BZh";

/// Number of records in first uncompressed section
const FIRST_UNCOMPRESSED_RECORDS: usize = 134;

/// Records per LDM block
const RECORDS_PER_LDM: usize = 120;

/// Manages compressed and uncompressed NEXRAD data access.
///
/// Uncompressed files: backed by raw bytes (could be memmapped).
/// BZ2 files: first 134 records uncompressed, then LDM blocks of 120 records each.
#[derive(Debug)]
pub struct RecordStore {
    /// Raw file bytes
    raw_data: Vec<u8>,
    /// Whether the file uses BZ2 compression
    pub is_compressed: bool,
    /// Offset where data records begin (after volume header)
    pub data_offset: usize,
    /// Cache of decompressed LDM blocks: ldm_index -> decompressed bytes
    ldm_cache: HashMap<usize, Vec<u8>>,
    /// Byte offsets of each BZ2 block in the raw data
    bz2_block_offsets: Vec<usize>,
}

impl RecordStore {
    pub fn new(raw_data: Vec<u8>, data_offset: usize) -> Self {
        let is_compressed = Self::detect_compression(&raw_data, data_offset);

        let mut store = RecordStore {
            raw_data,
            is_compressed,
            data_offset,
            ldm_cache: HashMap::new(),
            bz2_block_offsets: Vec::new(),
        };

        if store.is_compressed {
            store.find_bz2_block_offsets();
        }

        store
    }

    /// Detect if file is BZ2 compressed by checking bytes after volume header
    fn detect_compression(data: &[u8], offset: usize) -> bool {
        if data.len() <= offset + 3 {
            return false;
        }
        &data[offset..offset + 3] == BZ2_MAGIC
    }

    /// Find all BZ2 block boundaries in the raw data
    fn find_bz2_block_offsets(&mut self) {
        let data = &self.raw_data;
        let mut offsets = Vec::new();

        // Search for BZ2 magic sequence "BZh" followed by digit
        let start = self.data_offset;
        let mut i = start;
        while i + 3 < data.len() {
            if &data[i..i + 3] == BZ2_MAGIC && i + 3 < data.len() && data[i + 3].is_ascii_digit()
            {
                offsets.push(i);
                // Skip ahead to avoid finding magic inside a compressed block
                // Minimum BZ2 block is much larger than 10 bytes
                i += 10;
            } else {
                i += 1;
            }
        }

        self.bz2_block_offsets = offsets;
    }

    /// Get the raw bytes for a record at the given record number.
    /// Record 0 starts at data_offset.
    pub fn get_record(&mut self, record_num: usize) -> Result<&[u8], NexradError> {
        if self.is_compressed {
            self.get_compressed_record(record_num)
        } else {
            self.get_uncompressed_record(record_num)
        }
    }

    fn get_uncompressed_record(&self, record_num: usize) -> Result<&[u8], NexradError> {
        let offset = self.data_offset + record_num * RECORD_BYTES;
        if offset + RECORD_BYTES > self.raw_data.len() {
            return Err(NexradError::InvalidRecord(format!(
                "Record {} out of bounds (offset {}, file size {})",
                record_num,
                offset,
                self.raw_data.len()
            )));
        }
        Ok(&self.raw_data[offset..offset + RECORD_BYTES])
    }

    fn get_compressed_record(&mut self, record_num: usize) -> Result<&[u8], NexradError> {
        // First FIRST_UNCOMPRESSED_RECORDS records are in the first BZ2 block
        // which we treat as LDM block 0
        let ldm_index = record_num / RECORDS_PER_LDM;
        let record_in_ldm = record_num % RECORDS_PER_LDM;

        // Ensure the LDM block is decompressed
        if !self.ldm_cache.contains_key(&ldm_index) {
            self.decompress_ldm_block(ldm_index)?;
        }

        let block_data = self.ldm_cache.get(&ldm_index).ok_or_else(|| {
            NexradError::DecompressionError(format!("LDM block {} not in cache", ldm_index))
        })?;

        let offset = record_in_ldm * RECORD_BYTES;
        if offset + RECORD_BYTES > block_data.len() {
            return Err(NexradError::InvalidRecord(format!(
                "Record {} (ldm {}, offset {}) exceeds block size {}",
                record_num,
                ldm_index,
                offset,
                block_data.len()
            )));
        }

        Ok(&block_data[offset..offset + RECORD_BYTES])
    }

    /// Decompress a single LDM block
    fn decompress_ldm_block(&mut self, ldm_index: usize) -> Result<(), NexradError> {
        if ldm_index >= self.bz2_block_offsets.len() {
            return Err(NexradError::DecompressionError(format!(
                "LDM block {} out of range (only {} blocks found)",
                ldm_index,
                self.bz2_block_offsets.len()
            )));
        }

        let block_start = self.bz2_block_offsets[ldm_index];
        let block_end = if ldm_index + 1 < self.bz2_block_offsets.len() {
            self.bz2_block_offsets[ldm_index + 1]
        } else {
            self.raw_data.len()
        };

        let compressed = &self.raw_data[block_start..block_end];
        let mut decoder = BzDecoder::new(compressed);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).map_err(|e| {
            NexradError::DecompressionError(format!("BZ2 decompression failed: {}", e))
        })?;

        self.ldm_cache.insert(ldm_index, decompressed);
        Ok(())
    }

    /// Decompress all LDM blocks in parallel using rayon
    pub fn decompress_all_parallel(&mut self) -> Result<(), NexradError> {
        if !self.is_compressed || self.bz2_block_offsets.is_empty() {
            return Ok(());
        }

        use rayon::prelude::*;

        let offsets = &self.bz2_block_offsets;
        let raw_data = &self.raw_data;
        let total_len = raw_data.len();

        let results: Vec<(usize, Result<Vec<u8>, NexradError>)> = (0..offsets.len())
            .into_par_iter()
            .map(|i| {
                let block_start = offsets[i];
                let block_end = if i + 1 < offsets.len() {
                    offsets[i + 1]
                } else {
                    total_len
                };
                let compressed = &raw_data[block_start..block_end];
                let mut decoder = BzDecoder::new(compressed);
                let mut decompressed = Vec::new();
                match decoder.read_to_end(&mut decompressed) {
                    Ok(_) => (i, Ok(decompressed)),
                    Err(e) => (
                        i,
                        Err(NexradError::DecompressionError(format!(
                            "BZ2 block {} failed: {}",
                            i, e
                        ))),
                    ),
                }
            })
            .collect();

        for (idx, result) in results {
            self.ldm_cache.insert(idx, result?);
        }

        Ok(())
    }

    /// Total number of records available
    pub fn num_records(&self) -> usize {
        if self.is_compressed {
            // Estimate from decompressed data
            // This will be refined as blocks are decompressed
            self.bz2_block_offsets.len() * RECORDS_PER_LDM
        } else {
            (self.raw_data.len() - self.data_offset) / RECORD_BYTES
        }
    }

    /// Get total file size
    pub fn file_size(&self) -> usize {
        self.raw_data.len()
    }

    /// Get raw data reference
    pub fn raw_data(&self) -> &[u8] {
        &self.raw_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_compression_bz2() {
        let mut data = vec![0u8; 30];
        // Place BZ2 magic at offset 24 (after volume header)
        data[24] = b'B';
        data[25] = b'Z';
        data[26] = b'h';
        assert!(RecordStore::detect_compression(&data, 24));
    }

    #[test]
    fn test_detect_compression_not_bz2() {
        let data = vec![0u8; 30];
        assert!(!RecordStore::detect_compression(&data, 24));
    }

    #[test]
    fn test_uncompressed_record_access() {
        let header_size = 24;
        let mut data = vec![0u8; header_size + RECORD_BYTES * 3];
        // Put known bytes at record 1
        data[header_size + RECORD_BYTES] = 0xAB;
        data[header_size + RECORD_BYTES + 1] = 0xCD;

        let store = RecordStore::new(data, header_size);
        assert!(!store.is_compressed);
    }

    #[test]
    fn test_uncompressed_out_of_bounds() {
        let data = vec![0u8; 100];
        let store = RecordStore::new(data, 24);
        // Record 0 would need 24 + 2432 bytes = 2456, but we only have 100
        assert!(store.get_uncompressed_record(0).is_err());
    }
}
