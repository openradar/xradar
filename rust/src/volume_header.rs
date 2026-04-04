use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;

use crate::errors::NexradError;

/// NEXRAD Volume Header (24 bytes)
/// Format: tape(9s) extension(3s) date(u32) time(u32) icao(4s)
pub const VOLUME_HEADER_SIZE: usize = 24;
pub const VOLUME_HEADER_PREFIX: &[u8] = b"AR2V";

#[derive(Debug, Clone)]
pub struct VolumeHeader {
    pub tape: Vec<u8>,      // 9 bytes
    pub extension: Vec<u8>, // 3 bytes
    pub date: u32,
    pub time: u32,
    pub icao: Vec<u8>, // 4 bytes
}

impl VolumeHeader {
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < VOLUME_HEADER_SIZE {
            return Err(NexradError::InvalidVolumeHeader(format!(
                "Data too short: {} bytes, need {}",
                data.len(),
                VOLUME_HEADER_SIZE
            )));
        }

        if !data.starts_with(VOLUME_HEADER_PREFIX) {
            return Err(NexradError::InvalidVolumeHeader(format!(
                "Missing AR2V prefix, got {:?}",
                &data[..4]
            )));
        }

        let tape = data[0..9].to_vec();
        let extension = data[9..12].to_vec();

        let mut cursor = Cursor::new(&data[12..20]);
        let date = cursor.read_u32::<BigEndian>()?;
        let time = cursor.read_u32::<BigEndian>()?;

        let icao = data[20..24].to_vec();

        Ok(VolumeHeader {
            tape,
            extension,
            date,
            time,
            icao,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_header() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"AR2V0006."); // tape (9 bytes)
        buf.extend_from_slice(b"736"); // extension (3 bytes)
        buf.extend_from_slice(&0x3A4D_6400u32.to_be_bytes()); // date
        buf.extend_from_slice(&0x105E_3803u32.to_be_bytes()); // time
        buf.extend_from_slice(b"KLBB"); // icao (4 bytes)
        buf
    }

    #[test]
    fn test_parse_valid_header() {
        let data = make_valid_header();
        let header = VolumeHeader::parse(&data).unwrap();
        assert_eq!(header.tape, b"AR2V0006.");
        assert_eq!(header.extension, b"736");
        assert_eq!(header.icao, b"KLBB");
    }

    #[test]
    fn test_reject_non_ar2v() {
        let mut data = make_valid_header();
        data[0..4].copy_from_slice(b"XXXX");
        assert!(VolumeHeader::parse(&data).is_err());
    }

    #[test]
    fn test_reject_truncated() {
        let data = b"AR2V0006.73";
        assert!(VolumeHeader::parse(data).is_err());
    }
}
