use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;

use crate::errors::NexradError;

/// NEXRAD moment name mapping to CfRadial2/ODIM names
pub fn nexrad_to_cfradial(name: &str) -> &str {
    match name {
        "REF" => "DBZH",
        "VEL" => "VRADH",
        "SW " => "WRADH",
        "ZDR" => "ZDR",
        "PHI" => "PHIDP",
        "RHO" => "RHOHV",
        "CFP" => "CCORH",
        _ => name,
    }
}

/// Data block header (at the start of each block in MSG 31)
#[derive(Debug, Clone)]
pub struct DataBlockHeader {
    pub block_type: u8,   // 'R' for constant, 'D' for data
    pub data_name: [u8; 3], // 3-char identifier (REF, VEL, VOL, ELV, RAD, etc.)
}

impl DataBlockHeader {
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < 4 {
            return Err(NexradError::ParseError(
                "Data block header too short".to_string(),
            ));
        }
        let mut name = [0u8; 3];
        name.copy_from_slice(&data[1..4]);
        Ok(DataBlockHeader {
            block_type: data[0],
            data_name: name,
        })
    }

    pub fn name_str(&self) -> &str {
        std::str::from_utf8(&self.data_name).unwrap_or("???")
    }

    pub fn is_constant_block(&self) -> bool {
        self.block_type == b'R'
    }

    pub fn is_data_block(&self) -> bool {
        self.block_type == b'D'
    }
}

/// Generic data block for moment data (REF, VEL, SW, ZDR, PHI, RHO, CFP)
#[derive(Debug, Clone)]
pub struct GenericDataBlock {
    pub reserved: u32,
    pub ngates: u16,
    pub first_gate: i16,     // meters
    pub gate_spacing: i16,   // meters
    pub thresh: i16,
    pub snr_thres: i16,
    pub flags: u8,
    pub word_size: u8,       // 8 or 16 bits
    pub scale: f32,
    pub offset: f32,
    /// Raw moment data bytes (ngates * word_size/8 bytes)
    pub data: Vec<u8>,
}

impl GenericDataBlock {
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < 28 {
            return Err(NexradError::ParseError(
                "Generic data block too short".to_string(),
            ));
        }

        // Skip 4-byte data block header (block_type + data_name)
        let payload = &data[4..];

        let mut cursor = Cursor::new(payload);
        let reserved = cursor.read_u32::<BigEndian>()?;
        let ngates = cursor.read_u16::<BigEndian>()?;
        let first_gate = cursor.read_i16::<BigEndian>()?;
        let gate_spacing = cursor.read_i16::<BigEndian>()?;
        let thresh = cursor.read_i16::<BigEndian>()?;
        let snr_thres = cursor.read_i16::<BigEndian>()?;
        let flags = cursor.read_u8()?;
        let word_size = cursor.read_u8()?;
        let scale = cursor.read_f32::<BigEndian>()?;
        let offset = cursor.read_f32::<BigEndian>()?;

        let data_start = 4 + 24; // header(4) + fields(24)
        let data_bytes = ngates as usize * (word_size as usize / 8);
        let data_end = (data_start + data_bytes).min(data.len());

        let moment_data = if data_start <= data.len() {
            data[data_start..data_end].to_vec()
        } else {
            Vec::new()
        };

        Ok(GenericDataBlock {
            reserved,
            ngates,
            first_gate,
            gate_spacing,
            thresh,
            snr_thres,
            flags,
            word_size,
            scale,
            offset,
            data: moment_data,
        })
    }

    /// Get data as u8 array (for 8-bit word size)
    pub fn data_u8(&self) -> &[u8] {
        &self.data
    }

    /// Get data as u16 array (for 16-bit word size)
    pub fn data_u16(&self) -> Result<Vec<u16>, NexradError> {
        if self.word_size != 16 {
            return Err(NexradError::ParseError(
                "Cannot read u16 from non-16-bit data".to_string(),
            ));
        }
        let mut result = Vec::with_capacity(self.ngates as usize);
        let mut cursor = Cursor::new(&self.data);
        for _ in 0..self.ngates {
            if cursor.position() as usize + 2 <= self.data.len() {
                result.push(cursor.read_u16::<BigEndian>()?);
            } else {
                result.push(0);
            }
        }
        Ok(result)
    }
}

/// Volume data block (constant block "VOL")
#[derive(Debug, Clone)]
pub struct VolumeDataBlock {
    pub lrtup: u16,
    pub version_major: u8,
    pub version_minor: u8,
    pub latitude: f32,
    pub longitude: f32,
    pub site_height: i16,
    pub feedhorn_height: u16,
    pub calibration_constant: f32,
    pub horizontal_shv_tx_power: f32,
    pub vertical_shv_tx_power: f32,
    pub system_differential_reflectivity: f32,
    pub initial_system_differential_phase: f32,
    pub vcp_number: u16,
    pub processing_status: u16,
}

impl VolumeDataBlock {
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < 44 {
            return Err(NexradError::ParseError(
                "Volume data block too short".to_string(),
            ));
        }

        // Skip 4-byte data block header
        let payload = &data[4..];
        let mut cursor = Cursor::new(payload);

        let lrtup = cursor.read_u16::<BigEndian>()?;
        let version_major = cursor.read_u8()?;
        let version_minor = cursor.read_u8()?;
        let latitude = cursor.read_f32::<BigEndian>()?;
        let longitude = cursor.read_f32::<BigEndian>()?;
        let site_height = cursor.read_i16::<BigEndian>()?;
        let feedhorn_height = cursor.read_u16::<BigEndian>()?;
        let calibration_constant = cursor.read_f32::<BigEndian>()?;
        let horizontal_shv_tx_power = cursor.read_f32::<BigEndian>()?;
        let vertical_shv_tx_power = cursor.read_f32::<BigEndian>()?;
        let system_differential_reflectivity = cursor.read_f32::<BigEndian>()?;
        let initial_system_differential_phase = cursor.read_f32::<BigEndian>()?;
        let vcp_number = cursor.read_u16::<BigEndian>()?;
        let processing_status = cursor.read_u16::<BigEndian>()?;

        Ok(VolumeDataBlock {
            lrtup,
            version_major,
            version_minor,
            latitude,
            longitude,
            site_height,
            feedhorn_height,
            calibration_constant,
            horizontal_shv_tx_power,
            vertical_shv_tx_power,
            system_differential_reflectivity,
            initial_system_differential_phase,
            vcp_number,
            processing_status,
        })
    }
}

/// Elevation data block (constant block "ELV")
#[derive(Debug, Clone)]
pub struct ElevationDataBlock {
    pub lrtup: u16,
    pub atmos: i16,
    pub calibration_constant: f32,
}

impl ElevationDataBlock {
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < 12 {
            return Err(NexradError::ParseError(
                "Elevation data block too short".to_string(),
            ));
        }

        let payload = &data[4..];
        let mut cursor = Cursor::new(payload);
        let lrtup = cursor.read_u16::<BigEndian>()?;
        let atmos = cursor.read_i16::<BigEndian>()?;
        let calibration_constant = cursor.read_f32::<BigEndian>()?;

        Ok(ElevationDataBlock {
            lrtup,
            atmos,
            calibration_constant,
        })
    }
}

/// Radial data block (constant block "RAD")
#[derive(Debug, Clone)]
pub struct RadialDataBlock {
    pub lrtup: u16,
    pub unambiguous_range: u16,
    pub noise_level_h: f32,
    pub noise_level_v: f32,
    pub nyquist_velocity: u16,
    pub spare: u16,
    pub calibration_constant_h: f32,
    pub calibration_constant_v: f32,
}

impl RadialDataBlock {
    pub fn parse(data: &[u8]) -> Result<Self, NexradError> {
        if data.len() < 24 {
            return Err(NexradError::ParseError(
                "Radial data block too short".to_string(),
            ));
        }

        let payload = &data[4..];
        let mut cursor = Cursor::new(payload);
        let lrtup = cursor.read_u16::<BigEndian>()?;
        let unambiguous_range = cursor.read_u16::<BigEndian>()?;
        let noise_level_h = cursor.read_f32::<BigEndian>()?;
        let noise_level_v = cursor.read_f32::<BigEndian>()?;
        let nyquist_velocity = cursor.read_u16::<BigEndian>()?;
        let spare = cursor.read_u16::<BigEndian>()?;
        let calibration_constant_h = if cursor.position() as usize + 4 <= payload.len() {
            cursor.read_f32::<BigEndian>()?
        } else {
            0.0
        };
        let calibration_constant_v = if cursor.position() as usize + 4 <= payload.len() {
            cursor.read_f32::<BigEndian>()?
        } else {
            0.0
        };

        Ok(RadialDataBlock {
            lrtup,
            unambiguous_range,
            noise_level_h,
            noise_level_v,
            nyquist_velocity,
            spare,
            calibration_constant_h,
            calibration_constant_v,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nexrad_to_cfradial_mapping() {
        assert_eq!(nexrad_to_cfradial("REF"), "DBZH");
        assert_eq!(nexrad_to_cfradial("VEL"), "VRADH");
        assert_eq!(nexrad_to_cfradial("SW "), "WRADH");
        assert_eq!(nexrad_to_cfradial("ZDR"), "ZDR");
        assert_eq!(nexrad_to_cfradial("PHI"), "PHIDP");
        assert_eq!(nexrad_to_cfradial("RHO"), "RHOHV");
        assert_eq!(nexrad_to_cfradial("CFP"), "CCORH");
        assert_eq!(nexrad_to_cfradial("UNK"), "UNK");
    }

    #[test]
    fn test_data_block_header_constant() {
        let data = [b'R', b'V', b'O', b'L'];
        let header = DataBlockHeader::parse(&data).unwrap();
        assert!(header.is_constant_block());
        assert!(!header.is_data_block());
        assert_eq!(header.name_str(), "VOL");
    }

    #[test]
    fn test_data_block_header_data() {
        let data = [b'D', b'R', b'E', b'F'];
        let header = DataBlockHeader::parse(&data).unwrap();
        assert!(!header.is_constant_block());
        assert!(header.is_data_block());
        assert_eq!(header.name_str(), "REF");
    }

    #[test]
    fn test_generic_data_block_parse() {
        // Build a minimal generic data block
        let mut data = Vec::new();
        // Header: block_type='D', data_name="REF"
        data.push(b'D');
        data.extend_from_slice(b"REF");
        // reserved (4 bytes)
        data.extend_from_slice(&0u32.to_be_bytes());
        // ngates=4
        data.extend_from_slice(&4u16.to_be_bytes());
        // first_gate=2125
        data.extend_from_slice(&2125i16.to_be_bytes());
        // gate_spacing=250
        data.extend_from_slice(&250i16.to_be_bytes());
        // thresh=2
        data.extend_from_slice(&2i16.to_be_bytes());
        // snr_thres=0
        data.extend_from_slice(&0i16.to_be_bytes());
        // flags=0
        data.push(0);
        // word_size=8
        data.push(8);
        // scale=2.0
        data.extend_from_slice(&2.0f32.to_be_bytes());
        // offset=66.0
        data.extend_from_slice(&66.0f32.to_be_bytes());
        // 4 bytes of moment data (8-bit, 4 gates)
        data.extend_from_slice(&[100, 110, 120, 130]);

        let block = GenericDataBlock::parse(&data).unwrap();
        assert_eq!(block.ngates, 4);
        assert_eq!(block.first_gate, 2125);
        assert_eq!(block.gate_spacing, 250);
        assert_eq!(block.word_size, 8);
        assert!((block.scale - 2.0).abs() < 0.001);
        assert!((block.offset - 66.0).abs() < 0.001);
        assert_eq!(block.data_u8(), &[100, 110, 120, 130]);
    }
}
