use std::collections::{HashMap, HashSet};

use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;

use crate::errors::NexradError;
use crate::messages::{Msg1Header, Msg31Header, MsgHeader, CTM_HEADER_SIZE, MSG_HEADER_SIZE};
use crate::moments::{
    DataBlockHeader, ElevationDataBlock, GenericDataBlock, RadialDataBlock, VolumeDataBlock,
};
use crate::msg2::Msg2Data;
use crate::msg5::Msg5Data;
use crate::sweep::{LegacyRadial, Radial, Sweep, SweepAssembler};
use crate::volume_header::{VolumeHeader, VOLUME_HEADER_SIZE};

/// Record size for metadata records and MSG 1 (fixed-length)
const RECORD_BYTES: usize = 2432;

/// BZ2 magic header
const BZ2_MAGIC: &[u8; 3] = b"BZh";

/// Parsed NEXRAD Level2 file
pub struct NexradLevel2 {
    pub volume_header: VolumeHeader,
    pub msg2: Option<Msg2Data>,
    pub msg5: Option<Msg5Data>,
    pub meta_headers: MetaHeaders,
    pub sweeps: Vec<Sweep>,
    pub incomplete_sweeps: HashSet<usize>,
    pub is_legacy: bool,
}

/// Collection of metadata message headers
#[derive(Debug, Default)]
pub struct MetaHeaders {
    pub msg_2: Vec<MsgHeader>,
    pub msg_3: Vec<MsgHeader>,
    pub msg_5: Vec<MsgHeader>,
    pub msg_13: Vec<MsgHeader>,
    pub msg_15: Vec<MsgHeader>,
    pub msg_18: Vec<MsgHeader>,
}

/// Detect if file is BZ2 compressed by checking the 4-byte size field
/// at offset 24 (right after the volume header).
fn is_compressed(data: &[u8]) -> bool {
    if data.len() < 28 {
        return false;
    }
    // The 4 bytes at offset 24 are a size field; if > 0, file is compressed
    let size = u32::from_be_bytes([data[24], data[25], data[26], data[27]]);
    size > 0
}

/// Find BZ2 block offsets by searching for the BZ2 magic pattern.
/// The pattern is: 4-byte negative-size prefix, then "BZh" followed by a digit.
fn find_bz2_offsets(data: &[u8]) -> Vec<usize> {
    // The BZ2 compressed data starts with a 4-byte size field followed by BZ2 data.
    // We search for all positions where "BZh" + digit occurs, and the 4-byte prefix
    // at offset-4 is the compressed block size.
    let mut offsets = Vec::new();
    let start = VOLUME_HEADER_SIZE;
    let mut i = start;
    while i + 7 < data.len() {
        // Look for the 10-byte BZ2 magic: BZh[0-9]1AY&SY
        if data[i] == b'B' && data[i + 1] == b'Z' && data[i + 2] == b'h'
            && data[i + 3].is_ascii_digit()
            && i + 9 < data.len()
            && data[i + 4] == 0x31
            && data[i + 5] == 0x41
            && data[i + 6] == 0x59
            && data[i + 7] == 0x26
            && data[i + 8] == 0x53
            && data[i + 9] == 0x59
        {
            // BZ2 block starts 4 bytes before the "BZh" (at the size field)
            let block_offset = if i >= 4 { i - 4 } else { i };
            offsets.push(block_offset);
            i += 10; // skip past this magic to find next
        } else {
            i += 1;
        }
    }
    offsets
}

/// Decompress a single BZ2 block. The block starts with a 4-byte BE size,
/// then the compressed data follows.
fn decompress_bz2_block(data: &[u8], offset: usize) -> Result<Vec<u8>, NexradError> {
    if offset + 4 > data.len() {
        return Err(NexradError::DecompressionError("Block offset out of range".into()));
    }
    let size = u32::from_be_bytes([
        data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
    ]) as usize;
    let compressed_start = offset + 4;
    let compressed_end = (compressed_start + size).min(data.len());
    if compressed_start >= data.len() {
        return Err(NexradError::DecompressionError("Compressed data out of range".into()));
    }
    let compressed = &data[compressed_start..compressed_end];
    let mut decoder = bzip2::read::BzDecoder::new(compressed);
    let mut decompressed = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut decompressed).map_err(|e| {
        NexradError::DecompressionError(format!("BZ2 decompression failed: {}", e))
    })?;
    Ok(decompressed)
}

/// Iterator-style record reader that handles both compressed and uncompressed files.
/// It yields variable-length record slices matching the Python init_record logic.
struct RecordIterator {
    /// Decompressed LDM blocks (for compressed files)
    ldm_blocks: Vec<Vec<u8>>,
    /// Whether the file is compressed
    compressed: bool,
    /// Raw file data (for uncompressed files)
    raw_data: Vec<u8>,
    /// Current logical record number
    record_number: usize,
    /// Current file position (for uncompressed non-meta records)
    file_pos: usize,
    /// Current record size
    record_size: usize,
}

impl RecordIterator {
    fn new(data: Vec<u8>) -> Result<Self, NexradError> {
        let compressed = is_compressed(&data);
        let mut ldm_blocks = Vec::new();

        if compressed {
            let offsets = find_bz2_offsets(&data);
            // Decompress all blocks in parallel
            use rayon::prelude::*;
            let results: Vec<Result<Vec<u8>, NexradError>> = offsets
                .par_iter()
                .map(|&offset| decompress_bz2_block(&data, offset))
                .collect();
            for result in results {
                ldm_blocks.push(result?);
            }
        }

        Ok(RecordIterator {
            ldm_blocks,
            compressed,
            raw_data: data,
            record_number: 0,
            file_pos: 0,
            record_size: 0,
        })
    }

    /// Get the record data for the given record number.
    /// Returns None if beyond end of data.
    fn get_record(&mut self, recnum: usize) -> Option<Vec<u8>> {
        self.record_number = recnum;

        if self.compressed {
            self.get_compressed_record(recnum)
        } else {
            self.get_uncompressed_record(recnum)
        }
    }

    fn get_uncompressed_record(&mut self, recnum: usize) -> Option<Vec<u8>> {
        // For first 134 records: fixed size, offset from volume header
        if recnum < 134 {
            let start = recnum * RECORD_BYTES + VOLUME_HEADER_SIZE;
            let stop = start + RECORD_BYTES;
            if stop > self.raw_data.len() {
                return None;
            }
            self.file_pos = start;
            self.record_size = RECORD_BYTES;
            return Some(self.raw_data[start..stop].to_vec());
        }

        // For records >= 134: variable length, track position
        if recnum == 134 {
            // First data record after metadata
            self.file_pos = 134 * RECORD_BYTES + VOLUME_HEADER_SIZE;
            self.record_size = 0;
        }

        let start = self.file_pos + self.record_size;
        if start + CTM_HEADER_SIZE + MSG_HEADER_SIZE >= self.raw_data.len() {
            return None;
        }

        // Read MSG header to determine actual record size
        let hdr_offset = start + CTM_HEADER_SIZE;
        if hdr_offset + 16 > self.raw_data.len() {
            return None;
        }
        let msg_size = u16::from_be_bytes([
            self.raw_data[hdr_offset], self.raw_data[hdr_offset + 1],
        ]);
        let msg_type = self.raw_data[hdr_offset + 3];

        let mut size = msg_size as usize * 2 + CTM_HEADER_SIZE;
        if msg_type != 31 {
            size = size.max(RECORD_BYTES);
        }

        let stop = (start + size).min(self.raw_data.len());
        self.file_pos = start;
        self.record_size = stop - start;

        Some(self.raw_data[start..stop].to_vec())
    }

    fn get_compressed_record(&mut self, recnum: usize) -> Option<Vec<u8>> {
        // Map record number to LDM block
        let ldm_idx = if recnum < 134 { 0 } else { ((recnum - 134) / 120) + 1 };
        if ldm_idx >= self.ldm_blocks.len() {
            return None;
        }

        let ldm = &self.ldm_blocks[ldm_idx];

        if recnum < 134 {
            // Fixed-size records in first LDM block
            let start = recnum * RECORD_BYTES;
            let stop = start + RECORD_BYTES;
            if stop > ldm.len() {
                return None;
            }
            self.file_pos = start;
            self.record_size = RECORD_BYTES;
            return Some(ldm[start..stop].to_vec());
        }

        // Variable-length records in subsequent LDM blocks
        let rnum = (recnum - 134) % 120;
        let start = if rnum == 0 {
            0
        } else {
            self.file_pos + self.record_size
        };

        if start + CTM_HEADER_SIZE + MSG_HEADER_SIZE >= ldm.len() {
            return None;
        }

        // Read MSG header at offset 12 (after CTM header)
        let hdr_offset = start + CTM_HEADER_SIZE;
        if hdr_offset + 16 > ldm.len() {
            return None;
        }
        let msg_size = u16::from_be_bytes([ldm[hdr_offset], ldm[hdr_offset + 1]]);
        let msg_type = ldm[hdr_offset + 3];

        let mut size = msg_size as usize * 2 + CTM_HEADER_SIZE;
        if msg_type != 31 {
            size = size.max(RECORD_BYTES);
        }

        let stop = (start + size).min(ldm.len());
        if stop <= start {
            return None;
        }

        self.file_pos = start;
        self.record_size = stop - start;

        Some(ldm[start..stop].to_vec())
    }
}

impl NexradLevel2 {
    /// Parse a complete NEXRAD Level2 file from raw bytes.
    pub fn parse(data: Vec<u8>, loaddata: bool) -> Result<Self, NexradError> {
        // Parse volume header
        let volume_header = VolumeHeader::parse(&data)?;

        // Create record iterator
        let mut iter = RecordIterator::new(data)?;

        // Single sequential pass through all records
        let mut meta_headers = MetaHeaders::default();
        let mut msg2_data: Option<Msg2Data> = None;
        let mut msg5_data: Option<Msg5Data> = None;
        let mut assembler = SweepAssembler::new();
        let mut is_legacy = false;
        let mut in_metadata = true; // Start reading metadata records

        let mut recnum = 0;
        loop {
            let record = match iter.get_record(recnum) {
                Some(r) => r,
                None => break,
            };

            if record.len() < CTM_HEADER_SIZE + MSG_HEADER_SIZE {
                recnum += 1;
                continue;
            }

            let msg_header = match MsgHeader::parse(
                &record[CTM_HEADER_SIZE..],
                recnum,
            ) {
                Ok(h) => h,
                Err(_) => { recnum += 1; continue; }
            };

            let payload_start = CTM_HEADER_SIZE + MSG_HEADER_SIZE;

            if msg_header.msg_type == 0 {
                recnum += 1;
                continue;
            }

            // Classify: metadata or data message
            if in_metadata && [2, 3, 5, 13, 15, 18, 32].contains(&msg_header.msg_type) {
                // Metadata record
                match msg_header.msg_type {
                    2 => {
                        meta_headers.msg_2.push(msg_header.clone());
                        if msg2_data.is_none() && record.len() > payload_start {
                            msg2_data = Msg2Data::parse(&record[payload_start..]).ok();
                        }
                    }
                    3 => meta_headers.msg_3.push(msg_header),
                    5 => {
                        meta_headers.msg_5.push(msg_header.clone());
                        if msg5_data.is_none() && record.len() > payload_start {
                            let payload = &record[payload_start..];
                            msg5_data = Msg5Data::parse(payload).ok();
                        }
                    }
                    13 => meta_headers.msg_13.push(msg_header),
                    15 => meta_headers.msg_15.push(msg_header),
                    18 => meta_headers.msg_18.push(msg_header),
                    _ => {}
                }
            } else {
                // Transition to data records
                in_metadata = false;

                if loaddata && record.len() > payload_start {
                    if msg_header.msg_type == 31 {
                        let payload = &record[payload_start..];
                        if let Ok(radial) = Self::parse_msg31_radial(payload) {
                            assembler.add_radial(radial);
                        }
                    } else if msg_header.msg_type == 1 {
                        is_legacy = true;
                        let payload = &record[payload_start..];
                        if let Ok(radial) = Self::parse_msg1_radial(payload) {
                            assembler.add_legacy_radial(radial);
                        }
                    }
                }
            }

            recnum += 1;
        }

        if loaddata {
            assembler.finalize();
        }

        Ok(NexradLevel2 {
            volume_header,
            msg2: msg2_data,
            msg5: msg5_data,
            meta_headers,
            sweeps: assembler.sweeps,
            incomplete_sweeps: assembler.incomplete_sweeps,
            is_legacy,
        })
    }

    /// Parse a MSG 31 radial from the payload (after CTM + MSG headers)
    fn parse_msg31_radial(data: &[u8]) -> Result<Radial, NexradError> {
        let header = Msg31Header::parse(data)?;

        let mut volume_block = None;
        let mut elevation_block = None;
        let mut radial_block = None;
        let mut moment_blocks = HashMap::new();

        // Parse each data block using the offsets from the header
        for &offset in &header.block_offsets {
            let offset = offset as usize;
            if offset + 4 > data.len() {
                continue;
            }

            let block_header = match DataBlockHeader::parse(&data[offset..]) {
                Ok(h) => h,
                Err(_) => continue,
            };
            let name = block_header.name_str().to_string();

            if block_header.is_constant_block() {
                match name.as_str() {
                    "VOL" => {
                        volume_block = VolumeDataBlock::parse(&data[offset..]).ok();
                    }
                    "ELV" => {
                        elevation_block = ElevationDataBlock::parse(&data[offset..]).ok();
                    }
                    "RAD" => {
                        radial_block = RadialDataBlock::parse(&data[offset..]).ok();
                    }
                    _ => {}
                }
            } else if block_header.is_data_block() {
                if let Ok(gdb) = GenericDataBlock::parse(&data[offset..]) {
                    moment_blocks.insert(name, gdb);
                }
            }
        }

        Ok(Radial {
            header,
            volume_block,
            elevation_block,
            radial_block,
            moment_blocks,
        })
    }

    /// Parse a MSG 1 radial from the payload
    fn parse_msg1_radial(data: &[u8]) -> Result<LegacyRadial, NexradError> {
        let header = Msg1Header::parse(data)?;

        let ref_data = if header.ref_ptr > 0 && header.ngates_ref > 0 {
            let start = header.ref_ptr as usize;
            let end = start + header.ngates_ref as usize;
            if end <= data.len() {
                Some(data[start..end].to_vec())
            } else {
                None
            }
        } else {
            None
        };

        let vel_data = if header.vel_ptr > 0 && header.ngates_dop > 0 {
            let start = header.vel_ptr as usize;
            let end = start + header.ngates_dop as usize;
            if end <= data.len() {
                Some(data[start..end].to_vec())
            } else {
                None
            }
        } else {
            None
        };

        let sw_data = if header.sw_ptr > 0 && header.ngates_dop > 0 {
            let start = header.sw_ptr as usize;
            let end = start + header.ngates_dop as usize;
            if end <= data.len() {
                Some(data[start..end].to_vec())
            } else {
                None
            }
        } else {
            None
        };

        Ok(LegacyRadial {
            header,
            ref_data,
            vel_data,
            sw_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_headers_default() {
        let mh = MetaHeaders::default();
        assert!(mh.msg_2.is_empty());
        assert!(mh.msg_3.is_empty());
        assert!(mh.msg_5.is_empty());
        assert!(mh.msg_13.is_empty());
        assert!(mh.msg_15.is_empty());
        assert!(mh.msg_18.is_empty());
    }
}
