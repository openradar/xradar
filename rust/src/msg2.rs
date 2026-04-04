use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;

use crate::errors::NexradError;

/// MSG 2 - RDA Status Data
#[derive(Debug, Clone)]
pub struct Msg2Data {
    pub rda_status: u16,
    pub operability_status: u16,
    pub control_status: u16,
    pub aux_power_gen_state: u16,
    pub avg_transmitter_power: u16,
    pub ref_calib_correction: u16,
    pub data_transmission_enabled: u16,
    pub vcp_number: u16,
    pub rda_control_auth: u16,
    pub rda_build: u16,
    pub operational_mode: u16,
    pub super_res_status: u16,
    pub cmd_status: u16,
    pub avset_status: u16,
    pub rda_alarm_summary: u16,
    pub command_ack: u16,
    pub channel_control_status: u16,
    pub spot_blanking_status: u16,
    pub bypass_map_gen_date: u16,
    pub bypass_map_gen_time: u16,
    pub clutter_filter_map_gen_date: u16,
    pub clutter_filter_map_gen_time: u16,
    /// Scan data flags for AVSET/EBC decoding
    pub scan_data_flags: u16,
}

/// Decoded scan data flags from MSG 2
#[derive(Debug, Clone)]
pub struct ScanDataFlags {
    pub avset_enabled: bool,
    pub ebc_enabled: bool,
}

pub fn decode_scan_data_flags(value: u16) -> ScanDataFlags {
    ScanDataFlags {
        avset_enabled: (value >> 8) & 0x01 != 0,  // bit 8
        ebc_enabled: (value >> 9) & 0x01 != 0,    // bit 9
    }
}

impl Msg2Data {
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < 48 {
            return Err(NexradError::ParseError(
                "MSG 2 data too short".to_string(),
            ));
        }

        let mut cursor = Cursor::new(data);
        let rda_status = cursor.read_u16::<BigEndian>()?;
        let operability_status = cursor.read_u16::<BigEndian>()?;
        let control_status = cursor.read_u16::<BigEndian>()?;
        let aux_power_gen_state = cursor.read_u16::<BigEndian>()?;
        let avg_transmitter_power = cursor.read_u16::<BigEndian>()?;
        let ref_calib_correction = cursor.read_u16::<BigEndian>()?;
        let data_transmission_enabled = cursor.read_u16::<BigEndian>()?;
        let vcp_number = cursor.read_u16::<BigEndian>()?;
        let rda_control_auth = cursor.read_u16::<BigEndian>()?;
        let rda_build = cursor.read_u16::<BigEndian>()?;
        let operational_mode = cursor.read_u16::<BigEndian>()?;
        let super_res_status = cursor.read_u16::<BigEndian>()?;
        let cmd_status = cursor.read_u16::<BigEndian>()?;
        let avset_status = cursor.read_u16::<BigEndian>()?;
        let rda_alarm_summary = cursor.read_u16::<BigEndian>()?;
        let command_ack = cursor.read_u16::<BigEndian>()?;
        let channel_control_status = cursor.read_u16::<BigEndian>()?;
        let spot_blanking_status = cursor.read_u16::<BigEndian>()?;
        let bypass_map_gen_date = cursor.read_u16::<BigEndian>()?;
        let bypass_map_gen_time = cursor.read_u16::<BigEndian>()?;
        let clutter_filter_map_gen_date = cursor.read_u16::<BigEndian>()?;
        let clutter_filter_map_gen_time = cursor.read_u16::<BigEndian>()?;

        // scan_data_flags is at halfword 39 (byte 78) in the full MSG 2 structure
        // but our data starts after the MSG header, so offset depends on structure
        // For now, read from remaining data or default to 0
        let scan_data_flags = if data.len() >= 80 {
            let mut c = Cursor::new(&data[76..78]);
            c.read_u16::<BigEndian>().unwrap_or(0)
        } else {
            0
        };

        Ok(Msg2Data {
            rda_status,
            operability_status,
            control_status,
            aux_power_gen_state,
            avg_transmitter_power,
            ref_calib_correction,
            data_transmission_enabled,
            vcp_number,
            rda_control_auth,
            rda_build,
            operational_mode,
            super_res_status,
            cmd_status,
            avset_status,
            rda_alarm_summary,
            command_ack,
            channel_control_status,
            spot_blanking_status,
            bypass_map_gen_date,
            bypass_map_gen_time,
            clutter_filter_map_gen_date,
            clutter_filter_map_gen_time,
            scan_data_flags,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_scan_data_flags_none() {
        let flags = decode_scan_data_flags(0);
        assert!(!flags.avset_enabled);
        assert!(!flags.ebc_enabled);
    }

    #[test]
    fn test_decode_scan_data_flags_avset() {
        let flags = decode_scan_data_flags(0x0100); // bit 8 set
        assert!(flags.avset_enabled);
        assert!(!flags.ebc_enabled);
    }

    #[test]
    fn test_decode_scan_data_flags_ebc() {
        let flags = decode_scan_data_flags(0x0200); // bit 9 set
        assert!(!flags.avset_enabled);
        assert!(flags.ebc_enabled);
    }

    #[test]
    fn test_decode_scan_data_flags_both() {
        let flags = decode_scan_data_flags(0x0300); // bits 8 and 9 set
        assert!(flags.avset_enabled);
        assert!(flags.ebc_enabled);
    }
}
