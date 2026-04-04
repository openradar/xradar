use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;

use crate::errors::NexradError;

/// Message header size in bytes (H B B H H I H H = 2+1+1+2+2+4+2+2 = 16)
pub const MSG_HEADER_SIZE: usize = 16;

/// CTM header size (precedes message header in records)
pub const CTM_HEADER_SIZE: usize = 12;

/// Record size in bytes
pub const RECORD_BYTES: usize = 2432;

/// Compression record header size
pub const COMPRESSION_RECORD_SIZE: usize = 12;

/// Control word size
pub const CONTROL_WORD_SIZE: usize = 4;

/// Message header structure (12 bytes, big-endian)
#[derive(Debug, Clone)]
pub struct MsgHeader {
    pub size: u16,      // Data size in halfwords (not including header)
    pub channels: u8,   // Number of channels
    pub msg_type: u8,   // Message type (1, 2, 3, 5, 13, 15, 18, 31)
    pub seq_id: u16,    // Sequence ID
    pub date: u16,      // Julian date
    pub ms: u32,        // Milliseconds since midnight
    pub segments: u16,  // Number of segments
    pub seg_num: u16,   // Segment number
    pub record_number: usize, // Logical record number (not in binary, set by caller)
}

impl MsgHeader {
    pub fn parse(data: &[u8], record_number: usize) -> Result<Self, NexradError> {
        if data.len() < MSG_HEADER_SIZE {
            return Err(NexradError::InvalidMessageHeader(format!(
                "Data too short: {} bytes, need {}",
                data.len(),
                MSG_HEADER_SIZE
            )));
        }

        let mut cursor = Cursor::new(data);
        let size = cursor.read_u16::<BigEndian>()?;
        let channels = cursor.read_u8()?;
        let msg_type = cursor.read_u8()?;
        let seq_id = cursor.read_u16::<BigEndian>()?;
        let date = cursor.read_u16::<BigEndian>()?;
        let ms = cursor.read_u32::<BigEndian>()?;
        // segments and seg_num follow immediately (bytes 12-15)
        let (segments, seg_num) = if data.len() >= 16 {
            let segments = cursor.read_u16::<BigEndian>()?;
            let seg_num = cursor.read_u16::<BigEndian>()?;
            (segments, seg_num)
        } else {
            (1, 1)
        };

        Ok(MsgHeader {
            size,
            channels,
            msg_type,
            seq_id,
            date,
            ms,
            segments,
            seg_num,
            record_number,
        })
    }
}

/// MSG 31 header (for digital radar data generic format)
#[derive(Debug, Clone)]
pub struct Msg31Header {
    pub radar_id: [u8; 4],
    pub collect_ms: u32,
    pub collect_date: u16,
    pub azimuth_number: u16,
    pub azimuth_angle: f32,
    pub compression_indicator: u8,
    pub spare: u8,
    pub radial_length: u16,
    pub azimuth_resolution_spacing: u8,
    pub radial_status: u8,
    pub elevation_number: u8,
    pub cut_sector_number: u8,
    pub elevation_angle: f32,
    pub radial_spot_blanking: u8,
    pub azimuth_indexing_mode: u8,
    pub data_block_count: u16,
    /// Byte offsets to data blocks (up to 10)
    pub block_offsets: Vec<u32>,
}

impl Msg31Header {
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < 28 {
            return Err(NexradError::ParseError(
                "MSG 31 header too short".to_string(),
            ));
        }

        let mut cursor = Cursor::new(data);

        let mut radar_id = [0u8; 4];
        radar_id.copy_from_slice(&data[0..4]);
        cursor.set_position(4);

        let collect_ms = cursor.read_u32::<BigEndian>()?;
        let collect_date = cursor.read_u16::<BigEndian>()?;
        let azimuth_number = cursor.read_u16::<BigEndian>()?;
        let azimuth_angle = cursor.read_f32::<BigEndian>()?;
        let compression_indicator = cursor.read_u8()?;
        let spare = cursor.read_u8()?;
        let radial_length = cursor.read_u16::<BigEndian>()?;
        let azimuth_resolution_spacing = cursor.read_u8()?;
        let radial_status = cursor.read_u8()?;
        let elevation_number = cursor.read_u8()?;
        let cut_sector_number = cursor.read_u8()?;
        let elevation_angle = cursor.read_f32::<BigEndian>()?;
        let radial_spot_blanking = cursor.read_u8()?;
        let azimuth_indexing_mode = cursor.read_u8()?;
        let data_block_count = cursor.read_u16::<BigEndian>()?;

        let num_blocks = data_block_count as usize;
        let mut block_offsets = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            if cursor.position() as usize + 4 <= data.len() {
                block_offsets.push(cursor.read_u32::<BigEndian>()?);
            }
        }

        Ok(Msg31Header {
            radar_id,
            collect_ms,
            collect_date,
            azimuth_number,
            azimuth_angle,
            compression_indicator,
            spare,
            radial_length,
            azimuth_resolution_spacing,
            radial_status,
            elevation_number,
            cut_sector_number,
            elevation_angle,
            radial_spot_blanking,
            azimuth_indexing_mode,
            data_block_count,
            block_offsets,
        })
    }
}

/// MSG 1 header (legacy digital radar data)
#[derive(Debug, Clone)]
pub struct Msg1Header {
    pub collect_ms: u32,
    pub collect_date: u16,
    pub unambig_range: u16,
    pub azimuth_angle: u16,   // Raw value, needs scaling by 180/(4096*8)
    pub azimuth_number: u16,
    pub radial_status: u16,
    pub elevation_angle: u16, // Raw value, needs scaling by 180/(4096*8)
    pub elevation_number: u16,
    pub first_gate_range_ref: i16,
    pub first_gate_range_dop: i16,
    pub gate_spacing_ref: u16,
    pub gate_spacing_dop: u16,
    pub ngates_ref: u16,
    pub ngates_dop: u16,
    pub cut_sector_number: u16,
    pub calibration_constant: f32,
    pub ref_ptr: u16,
    pub vel_ptr: u16,
    pub sw_ptr: u16,
    pub velocity_resolution: u16,
    pub vcp_number: u16,
    pub nyquist_vel: u16,
    pub atmos: u16,
    pub tover: u16,
}

/// Angle scaling for MSG 1 format
pub const MSG1_ANGLE_SCALE: f32 = 180.0 / (4096.0 * 8.0);

impl Msg1Header {
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < 64 {
            return Err(NexradError::ParseError(
                "MSG 1 header too short".to_string(),
            ));
        }

        let mut cursor = Cursor::new(data);
        let collect_ms = cursor.read_u32::<BigEndian>()?;
        let collect_date = cursor.read_u16::<BigEndian>()?;
        let unambig_range = cursor.read_u16::<BigEndian>()?;
        let azimuth_angle = cursor.read_u16::<BigEndian>()?;
        let azimuth_number = cursor.read_u16::<BigEndian>()?;
        let radial_status = cursor.read_u16::<BigEndian>()?;
        let elevation_angle = cursor.read_u16::<BigEndian>()?;
        let elevation_number = cursor.read_u16::<BigEndian>()?;
        let first_gate_range_ref = cursor.read_i16::<BigEndian>()?;
        let first_gate_range_dop = cursor.read_i16::<BigEndian>()?;
        let gate_spacing_ref = cursor.read_u16::<BigEndian>()?;
        let gate_spacing_dop = cursor.read_u16::<BigEndian>()?;
        let ngates_ref = cursor.read_u16::<BigEndian>()?;
        let ngates_dop = cursor.read_u16::<BigEndian>()?;
        let cut_sector_number = cursor.read_u16::<BigEndian>()?;
        let calibration_constant = cursor.read_f32::<BigEndian>()?;
        let ref_ptr = cursor.read_u16::<BigEndian>()?;
        let vel_ptr = cursor.read_u16::<BigEndian>()?;
        let sw_ptr = cursor.read_u16::<BigEndian>()?;
        let velocity_resolution = cursor.read_u16::<BigEndian>()?;
        let vcp_number = cursor.read_u16::<BigEndian>()?;

        // Skip 8 bytes of spare
        cursor.set_position(cursor.position() + 8);

        let nyquist_vel = cursor.read_u16::<BigEndian>()?;
        let atmos = cursor.read_u16::<BigEndian>()?;
        let tover = cursor.read_u16::<BigEndian>()?;

        Ok(Msg1Header {
            collect_ms,
            collect_date,
            unambig_range,
            azimuth_angle,
            azimuth_number,
            radial_status,
            elevation_angle,
            elevation_number,
            first_gate_range_ref,
            first_gate_range_dop,
            gate_spacing_ref,
            gate_spacing_dop,
            ngates_ref,
            ngates_dop,
            cut_sector_number,
            calibration_constant,
            ref_ptr,
            vel_ptr,
            sw_ptr,
            velocity_resolution,
            vcp_number,
            nyquist_vel,
            atmos,
            tover,
        })
    }

    pub fn azimuth_degrees(&self) -> f32 {
        self.azimuth_angle as f32 * MSG1_ANGLE_SCALE
    }

    pub fn elevation_degrees(&self) -> f32 {
        self.elevation_angle as f32 * MSG1_ANGLE_SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_msg_header() {
        // Construct a 16-byte MSG header
        let mut data = Vec::new();
        data.extend_from_slice(&1208u16.to_be_bytes()); // size
        data.push(8); // channels
        data.push(15); // type
        data.extend_from_slice(&5u16.to_be_bytes()); // seq_id
        data.extend_from_slice(&16940u16.to_be_bytes()); // date
        data.extend_from_slice(&65175863u32.to_be_bytes()); // ms
        data.extend_from_slice(&5u16.to_be_bytes()); // segments
        data.extend_from_slice(&1u16.to_be_bytes()); // seg_num

        let header = MsgHeader::parse(&data, 0).unwrap();
        assert_eq!(header.size, 1208);
        assert_eq!(header.channels, 8);
        assert_eq!(header.msg_type, 15);
        assert_eq!(header.seq_id, 5);
        assert_eq!(header.date, 16940);
        assert_eq!(header.ms, 65175863);
        assert_eq!(header.segments, 5);
        assert_eq!(header.seg_num, 1);
    }

    #[test]
    fn test_msg_header_too_short() {
        let data = [0u8; 8];
        assert!(MsgHeader::parse(&data, 0).is_err());
    }

    #[test]
    fn test_msg1_angle_scaling() {
        // 32768 * 180 / (4096 * 8) = 180.0 degrees
        let scaled = 32768u16 as f32 * MSG1_ANGLE_SCALE;
        assert!((scaled - 180.0).abs() < 0.01);
    }
}
