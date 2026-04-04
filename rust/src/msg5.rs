use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;

use crate::errors::NexradError;

/// MSG 5 - Volume Coverage Pattern (VCP) Data
#[derive(Debug, Clone)]
pub struct Msg5Data {
    pub message_size: u16,
    pub pattern_type: u16,
    pub pattern_number: u16,
    pub number_elevation_cuts: u16,
    pub clutter_map_group_number: u16,
    pub doppler_velocity_resolution: u8,
    pub pulse_width: u8,
    pub vcp_sequencing: u16,
    pub vcp_supplemental: u16,
    pub vcp_sequencing_decoded: VcpSequencing,
    pub vcp_supplemental_decoded: VcpSupplemental,
    pub elevation_data: Vec<ElevationData>,
}

#[derive(Debug, Clone)]
pub struct VcpSequencing {
    pub num_elevations: u16,
    pub max_sails_cuts: u16,
    pub sequence_active: bool,
    pub truncated_vcp: bool,
}

#[derive(Debug, Clone)]
pub struct VcpSupplemental {
    pub sails_vcp: bool,
    pub num_sails_cuts: u16,
    pub mrle_vcp: bool,
    pub num_mrle_cuts: u16,
    pub mpda_vcp: bool,
    pub base_tilt_vcp: bool,
    pub num_base_tilts: u16,
}

#[derive(Debug, Clone)]
pub struct ElevationData {
    pub elevation_angle: f32,
    pub channel_config: u8,
    pub waveform_type: u8,
    pub super_resolution: u8,
    pub prf_number: u8,
    pub prf_pulse_count: u16,
    pub azimuth_rate: u16,
    pub ref_thresh: u16,
    pub vel_thresh: u16,
    pub sw_thresh: u16,
    pub zdr_thres: u16,
    pub phi_thres: u16,
    pub rho_thres: u16,
    pub edge_angle_1: u16,
    pub dop_prf_num_1: u16,
    pub dop_prf_pulse_count_1: u16,
    pub supplemental_data: u16,
    pub edge_angle_2: u16,
    pub dop_prf_num_2: u16,
    pub dop_prf_pulse_count_2: u16,
    pub edge_angle_3: u16,
    pub dop_prf_num_3: u16,
    pub dop_prf_pulse_count_3: u16,
    pub supplemental_data_decoded: ElevationSupplemental,
}

#[derive(Debug, Clone)]
pub struct ElevationSupplemental {
    pub sails_cut: bool,
    pub sails_sequence_number: u16,
    pub mrle_cut: bool,
    pub mrle_sequence_number: u16,
    pub mpda_cut: bool,
    pub base_tilt_cut: bool,
}

/// Decode VCP sequencing bits (from vcp_sequencing u16)
pub fn decode_vcp_sequencing(value: u16) -> VcpSequencing {
    VcpSequencing {
        num_elevations: value & 0x1F,            // bits 0-4
        max_sails_cuts: (value >> 5) & 0x03,     // bits 5-6
        sequence_active: (value >> 13) & 0x01 != 0, // bit 13
        truncated_vcp: (value >> 14) & 0x01 != 0,   // bit 14
    }
}

/// Decode VCP supplemental bits (from vcp_supplemental u16)
pub fn decode_vcp_supplemental(value: u16) -> VcpSupplemental {
    VcpSupplemental {
        sails_vcp: value & 0x01 != 0,              // bit 0
        num_sails_cuts: (value >> 1) & 0x07,       // bits 1-3
        mrle_vcp: (value >> 4) & 0x01 != 0,        // bit 4
        num_mrle_cuts: (value >> 5) & 0x07,        // bits 5-7
        mpda_vcp: (value >> 11) & 0x01 != 0,       // bit 11
        base_tilt_vcp: (value >> 12) & 0x01 != 0,  // bit 12
        num_base_tilts: (value >> 13) & 0x07,      // bits 13-15
    }
}

/// Decode per-elevation supplemental data bits
pub fn decode_elevation_supplemental(value: u16) -> ElevationSupplemental {
    ElevationSupplemental {
        sails_cut: value & 0x01 != 0,                 // bit 0
        sails_sequence_number: (value >> 1) & 0x07,   // bits 1-3
        mrle_cut: (value >> 4) & 0x01 != 0,           // bit 4
        mrle_sequence_number: (value >> 5) & 0x07,    // bits 5-7
        mpda_cut: (value >> 9) & 0x01 != 0,           // bit 9
        base_tilt_cut: (value >> 10) & 0x01 != 0,     // bit 10
    }
}

/// Elevation angle scaling: raw u16 value * 360.0 / 65536.0 = degrees
/// This is the BIN2 scaling used in MSG 5
const ELEV_ANGLE_SCALE: f32 = 360.0 / 65536.0;

impl Msg5Data {
    /// Parse MSG 5 data from the raw message payload (after MSG_HEADER).
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < 12 {
            return Err(NexradError::ParseError(
                "MSG 5 data too short".to_string(),
            ));
        }

        let mut cursor = Cursor::new(data);
        let message_size = cursor.read_u16::<BigEndian>()?;
        let pattern_type = cursor.read_u16::<BigEndian>()?;
        let pattern_number = cursor.read_u16::<BigEndian>()?;
        let number_elevation_cuts = cursor.read_u16::<BigEndian>()?;
        let clutter_map_group_number = cursor.read_u16::<BigEndian>()?;

        let dop_vel_res = cursor.read_u8()?;
        let pulse_width = cursor.read_u8()?;

        // Skip 2 bytes (spare)
        cursor.set_position(cursor.position() + 2);

        let vcp_sequencing = if cursor.position() as usize + 2 <= data.len() {
            cursor.read_u16::<BigEndian>()?
        } else {
            0
        };

        let vcp_supplemental = if cursor.position() as usize + 2 <= data.len() {
            cursor.read_u16::<BigEndian>()?
        } else {
            0
        };

        // Skip to elevation data (offset 40 bytes from start of MSG 5 payload)
        // The MSG 5 has some spare/reserved bytes between the header and elevation data
        let elev_data_offset = 40;
        let elev_record_size = 46; // Each elevation record is 46 bytes

        let mut elevation_data = Vec::new();
        for i in 0..number_elevation_cuts as usize {
            let offset = elev_data_offset + i * elev_record_size;
            if offset + elev_record_size > data.len() {
                break;
            }
            let elev = Self::parse_elevation(&data[offset..offset + elev_record_size])?;
            elevation_data.push(elev);
        }

        Ok(Msg5Data {
            message_size,
            pattern_type,
            pattern_number,
            number_elevation_cuts,
            clutter_map_group_number,
            doppler_velocity_resolution: dop_vel_res,
            pulse_width,
            vcp_sequencing,
            vcp_supplemental,
            vcp_sequencing_decoded: decode_vcp_sequencing(vcp_sequencing),
            vcp_supplemental_decoded: decode_vcp_supplemental(vcp_supplemental),
            elevation_data,
        })
    }

    fn parse_elevation(data: &[u8]) -> Result<ElevationData, NexradError> {
        let mut cursor = Cursor::new(data);

        let elevation_angle_raw = cursor.read_u16::<BigEndian>()?;
        let elevation_angle = elevation_angle_raw as f32 * ELEV_ANGLE_SCALE;
        let channel_config = cursor.read_u8()?;
        let waveform_type = cursor.read_u8()?;
        let super_resolution = cursor.read_u8()?;
        // Skip 1 byte spare
        cursor.set_position(cursor.position() + 1);
        let prf_number = cursor.read_u8()?;
        // Skip 1 byte spare
        cursor.set_position(cursor.position() + 1);
        let prf_pulse_count = cursor.read_u16::<BigEndian>()?;
        let azimuth_rate = cursor.read_u16::<BigEndian>()?;
        let ref_thresh = cursor.read_u16::<BigEndian>()?;
        let vel_thresh = cursor.read_u16::<BigEndian>()?;
        let sw_thresh = cursor.read_u16::<BigEndian>()?;
        let zdr_thres = cursor.read_u16::<BigEndian>()?;
        let phi_thres = cursor.read_u16::<BigEndian>()?;
        let rho_thres = cursor.read_u16::<BigEndian>()?;
        let edge_angle_1 = cursor.read_u16::<BigEndian>()?;
        let dop_prf_num_1 = cursor.read_u16::<BigEndian>()?;
        let dop_prf_pulse_count_1 = cursor.read_u16::<BigEndian>()?;
        let supplemental_data = cursor.read_u16::<BigEndian>()?;
        let edge_angle_2 = cursor.read_u16::<BigEndian>()?;
        let dop_prf_num_2 = cursor.read_u16::<BigEndian>()?;
        let dop_prf_pulse_count_2 = cursor.read_u16::<BigEndian>()?;
        let edge_angle_3 = cursor.read_u16::<BigEndian>()?;
        let dop_prf_num_3 = cursor.read_u16::<BigEndian>()?;
        let dop_prf_pulse_count_3 = cursor.read_u16::<BigEndian>()?;

        Ok(ElevationData {
            elevation_angle,
            channel_config,
            waveform_type,
            super_resolution,
            prf_number,
            prf_pulse_count,
            azimuth_rate,
            ref_thresh,
            vel_thresh,
            sw_thresh,
            zdr_thres,
            phi_thres,
            rho_thres,
            edge_angle_1,
            dop_prf_num_1,
            dop_prf_pulse_count_1,
            supplemental_data,
            edge_angle_2,
            dop_prf_num_2,
            dop_prf_pulse_count_2,
            edge_angle_3,
            dop_prf_num_3,
            dop_prf_pulse_count_3,
            supplemental_data_decoded: decode_elevation_supplemental(supplemental_data),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_vcp_sequencing_zero() {
        let decoded = decode_vcp_sequencing(0);
        assert_eq!(decoded.num_elevations, 0);
        assert_eq!(decoded.max_sails_cuts, 0);
        assert!(!decoded.sequence_active);
        assert!(!decoded.truncated_vcp);
    }

    #[test]
    fn test_decode_vcp_sequencing_all_bits() {
        // bits 0-4: 11 (0b01011), bits 5-6: 3 (0b11), bit 13: 1, bit 14: 1
        let value: u16 = 0b0110_0000_0110_1011;
        let decoded = decode_vcp_sequencing(value);
        assert_eq!(decoded.num_elevations, 11);
        assert_eq!(decoded.max_sails_cuts, 3);
        assert!(decoded.sequence_active);
        assert!(decoded.truncated_vcp);
    }

    #[test]
    fn test_decode_vcp_supplemental_zero() {
        let decoded = decode_vcp_supplemental(0);
        assert!(!decoded.sails_vcp);
        assert_eq!(decoded.num_sails_cuts, 0);
        assert!(!decoded.mrle_vcp);
        assert_eq!(decoded.num_mrle_cuts, 0);
        assert!(!decoded.mpda_vcp);
        assert!(!decoded.base_tilt_vcp);
        assert_eq!(decoded.num_base_tilts, 0);
    }

    #[test]
    fn test_decode_vcp_supplemental_sails() {
        // bit 0: sails_vcp=true, bits 1-3: num_sails_cuts=3
        let value: u16 = 0b0000_0000_0000_0111; // bits 0-2 set
        let decoded = decode_vcp_supplemental(value);
        assert!(decoded.sails_vcp);
        assert_eq!(decoded.num_sails_cuts, 3);
    }

    #[test]
    fn test_decode_vcp_supplemental_mrle() {
        // bit 4: mrle_vcp=true, bits 5-7: num_mrle_cuts=2
        let value: u16 = 0b0000_0000_0101_0000;
        let decoded = decode_vcp_supplemental(value);
        assert!(decoded.mrle_vcp);
        assert_eq!(decoded.num_mrle_cuts, 2);
    }

    #[test]
    fn test_decode_vcp_supplemental_mpda_base_tilt() {
        // bit 11: mpda=true, bit 12: base_tilt=true, bits 13-15: num_base_tilts=5
        let value: u16 = 0b1011_1000_0000_0000;
        let decoded = decode_vcp_supplemental(value);
        assert!(decoded.mpda_vcp);
        assert!(decoded.base_tilt_vcp);
        assert_eq!(decoded.num_base_tilts, 5);
    }

    #[test]
    fn test_decode_elevation_supplemental_zero() {
        let decoded = decode_elevation_supplemental(0);
        assert!(!decoded.sails_cut);
        assert_eq!(decoded.sails_sequence_number, 0);
        assert!(!decoded.mrle_cut);
        assert_eq!(decoded.mrle_sequence_number, 0);
        assert!(!decoded.mpda_cut);
        assert!(!decoded.base_tilt_cut);
    }

    #[test]
    fn test_decode_elevation_supplemental_sails() {
        // bit 0: sails_cut=true, bits 1-3: sails_seq=2
        let value: u16 = 0b0000_0000_0000_0101;
        let decoded = decode_elevation_supplemental(value);
        assert!(decoded.sails_cut);
        assert_eq!(decoded.sails_sequence_number, 2);
    }

    #[test]
    fn test_decode_elevation_supplemental_mrle() {
        // bit 4: mrle_cut=true, bits 5-7: mrle_seq=3
        let value: u16 = 0b0000_0000_0111_0000;
        let decoded = decode_elevation_supplemental(value);
        assert!(decoded.mrle_cut);
        assert_eq!(decoded.mrle_sequence_number, 3);
    }

    #[test]
    fn test_decode_elevation_supplemental_mpda_base_tilt() {
        // bit 9: mpda=true, bit 10: base_tilt=true
        let value: u16 = 0b0000_0110_0000_0000;
        let decoded = decode_elevation_supplemental(value);
        assert!(decoded.mpda_cut);
        assert!(decoded.base_tilt_cut);
    }
}
