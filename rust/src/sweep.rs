use std::collections::{HashMap, HashSet};

use crate::messages::{Msg1Header, Msg31Header};
use crate::moments::{
    ElevationDataBlock, GenericDataBlock, RadialDataBlock, VolumeDataBlock,
};

/// Radial status codes
pub const RADIAL_STATUS_START_ELEVATION: u8 = 0;
pub const RADIAL_STATUS_INTERMEDIATE: u8 = 1;
pub const RADIAL_STATUS_END_ELEVATION: u8 = 2;
pub const RADIAL_STATUS_START_NEW_ELEVATION: u8 = 3;
pub const RADIAL_STATUS_END_VOLUME: u8 = 4;
pub const RADIAL_STATUS_START_VOLUME: u8 = 5;

/// A single radial (ray) of data, parsed from MSG 31
#[derive(Debug, Clone)]
pub struct Radial {
    pub header: Msg31Header,
    pub volume_block: Option<VolumeDataBlock>,
    pub elevation_block: Option<ElevationDataBlock>,
    pub radial_block: Option<RadialDataBlock>,
    /// Moment data blocks keyed by name (e.g., "REF", "VEL", etc.)
    pub moment_blocks: HashMap<String, GenericDataBlock>,
}

/// A single radial from MSG 1 (legacy format)
#[derive(Debug, Clone)]
pub struct LegacyRadial {
    pub header: Msg1Header,
    /// REF data (8-bit)
    pub ref_data: Option<Vec<u8>>,
    /// VEL data (8-bit)
    pub vel_data: Option<Vec<u8>>,
    /// SW data (8-bit)
    pub sw_data: Option<Vec<u8>>,
}

/// A complete sweep (elevation cut)
#[derive(Debug)]
pub struct Sweep {
    pub sweep_number: usize,
    /// MSG 31 radials (modern format)
    pub radials: Vec<Radial>,
    /// MSG 1 radials (legacy format) -- mutually exclusive with radials
    pub legacy_radials: Vec<LegacyRadial>,
    /// Whether this sweep was forced-closed at EOF (incomplete)
    pub is_incomplete: bool,
    /// Whether this sweep uses legacy MSG 1 format
    pub is_legacy: bool,
}

impl Sweep {
    pub fn new(sweep_number: usize) -> Self {
        Sweep {
            sweep_number,
            radials: Vec::new(),
            legacy_radials: Vec::new(),
            is_incomplete: false,
            is_legacy: false,
        }
    }

    pub fn num_radials(&self) -> usize {
        if self.is_legacy {
            self.legacy_radials.len()
        } else {
            self.radials.len()
        }
    }

    /// Get the list of available moment names in this sweep
    pub fn moment_names(&self) -> Vec<String> {
        if self.is_legacy {
            let mut names = Vec::new();
            if let Some(r) = self.legacy_radials.first() {
                if r.ref_data.is_some() {
                    names.push("REF".to_string());
                }
                if r.vel_data.is_some() {
                    names.push("VEL".to_string());
                }
                if r.sw_data.is_some() {
                    names.push("SW ".to_string());
                }
            }
            names
        } else if let Some(r) = self.radials.first() {
            r.moment_blocks.keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Get azimuth angles for all radials
    pub fn azimuths(&self) -> Vec<f32> {
        if self.is_legacy {
            self.legacy_radials
                .iter()
                .map(|r| r.header.azimuth_degrees())
                .collect()
        } else {
            self.radials
                .iter()
                .map(|r| r.header.azimuth_angle)
                .collect()
        }
    }

    /// Get elevation angles for all radials
    pub fn elevations(&self) -> Vec<f32> {
        if self.is_legacy {
            self.legacy_radials
                .iter()
                .map(|r| r.header.elevation_degrees())
                .collect()
        } else {
            self.radials
                .iter()
                .map(|r| r.header.elevation_angle)
                .collect()
        }
    }

    /// Get collection times (milliseconds since midnight) for all radials
    pub fn collect_times_ms(&self) -> Vec<u32> {
        if self.is_legacy {
            self.legacy_radials
                .iter()
                .map(|r| r.header.collect_ms)
                .collect()
        } else {
            self.radials
                .iter()
                .map(|r| r.header.collect_ms)
                .collect()
        }
    }

    /// Get collection dates (Julian date) for all radials
    pub fn collect_dates(&self) -> Vec<u16> {
        if self.is_legacy {
            self.legacy_radials
                .iter()
                .map(|r| r.header.collect_date)
                .collect()
        } else {
            self.radials
                .iter()
                .map(|r| r.header.collect_date)
                .collect()
        }
    }
}

/// Assembles radials into sweeps based on radial status codes.
pub struct SweepAssembler {
    pub sweeps: Vec<Sweep>,
    pub incomplete_sweeps: HashSet<usize>,
    current_sweep: Option<Sweep>,
    sweep_count: usize,
}

impl SweepAssembler {
    pub fn new() -> Self {
        SweepAssembler {
            sweeps: Vec::new(),
            incomplete_sweeps: HashSet::new(),
            current_sweep: None,
            sweep_count: 0,
        }
    }

    /// Add a MSG 31 radial to the assembler
    pub fn add_radial(&mut self, radial: Radial) {
        let status = radial.header.radial_status;

        match status {
            RADIAL_STATUS_START_ELEVATION
            | RADIAL_STATUS_START_NEW_ELEVATION
            | RADIAL_STATUS_START_VOLUME => {
                // Close any existing sweep
                if let Some(sweep) = self.current_sweep.take() {
                    self.sweeps.push(sweep);
                }
                // Start new sweep
                let mut sweep = Sweep::new(self.sweep_count);
                self.sweep_count += 1;
                sweep.radials.push(radial);
                self.current_sweep = Some(sweep);
            }
            RADIAL_STATUS_INTERMEDIATE => {
                if let Some(ref mut sweep) = self.current_sweep {
                    sweep.radials.push(radial);
                } else {
                    // Stray intermediate radial; start a new sweep
                    let mut sweep = Sweep::new(self.sweep_count);
                    self.sweep_count += 1;
                    sweep.radials.push(radial);
                    self.current_sweep = Some(sweep);
                }
            }
            RADIAL_STATUS_END_ELEVATION | RADIAL_STATUS_END_VOLUME => {
                if let Some(ref mut sweep) = self.current_sweep {
                    sweep.radials.push(radial);
                }
                // Close the sweep
                if let Some(sweep) = self.current_sweep.take() {
                    self.sweeps.push(sweep);
                }
            }
            _ => {
                // Unknown status; treat as intermediate
                if let Some(ref mut sweep) = self.current_sweep {
                    sweep.radials.push(radial);
                }
            }
        }
    }

    /// Add a legacy MSG 1 radial
    pub fn add_legacy_radial(&mut self, radial: LegacyRadial) {
        let status = radial.header.radial_status;

        match status {
            0 | 3 | 5 => {
                if let Some(sweep) = self.current_sweep.take() {
                    self.sweeps.push(sweep);
                }
                let mut sweep = Sweep::new(self.sweep_count);
                sweep.is_legacy = true;
                self.sweep_count += 1;
                sweep.legacy_radials.push(radial);
                self.current_sweep = Some(sweep);
            }
            1 => {
                if let Some(ref mut sweep) = self.current_sweep {
                    sweep.legacy_radials.push(radial);
                } else {
                    let mut sweep = Sweep::new(self.sweep_count);
                    sweep.is_legacy = true;
                    self.sweep_count += 1;
                    sweep.legacy_radials.push(radial);
                    self.current_sweep = Some(sweep);
                }
            }
            2 | 4 => {
                if let Some(ref mut sweep) = self.current_sweep {
                    sweep.legacy_radials.push(radial);
                }
                if let Some(sweep) = self.current_sweep.take() {
                    self.sweeps.push(sweep);
                }
            }
            _ => {
                if let Some(ref mut sweep) = self.current_sweep {
                    sweep.legacy_radials.push(radial);
                }
            }
        }
    }

    /// Finalize assembly: close any open sweep as incomplete
    pub fn finalize(&mut self) {
        if let Some(mut sweep) = self.current_sweep.take() {
            sweep.is_incomplete = true;
            let idx = sweep.sweep_number;
            self.incomplete_sweeps.insert(idx);
            self.sweeps.push(sweep);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::Msg31Header;

    fn make_msg31_header(status: u8) -> Msg31Header {
        Msg31Header {
            radar_id: *b"KLBB",
            collect_ms: 54000000,
            collect_date: 16954,
            azimuth_number: 1,
            azimuth_angle: 0.5,
            compression_indicator: 0,
            spare: 0,
            radial_length: 100,
            azimuth_resolution_spacing: 1,
            radial_status: status,
            elevation_number: 1,
            cut_sector_number: 1,
            elevation_angle: 0.48,
            radial_spot_blanking: 0,
            azimuth_indexing_mode: 0,
            data_block_count: 0,
            block_offsets: Vec::new(),
        }
    }

    fn make_radial(status: u8) -> Radial {
        Radial {
            header: make_msg31_header(status),
            volume_block: None,
            elevation_block: None,
            radial_block: None,
            moment_blocks: HashMap::new(),
        }
    }

    #[test]
    fn test_basic_sweep_assembly() {
        let mut assembler = SweepAssembler::new();
        assembler.add_radial(make_radial(RADIAL_STATUS_START_ELEVATION));
        assembler.add_radial(make_radial(RADIAL_STATUS_INTERMEDIATE));
        assembler.add_radial(make_radial(RADIAL_STATUS_INTERMEDIATE));
        assembler.add_radial(make_radial(RADIAL_STATUS_END_ELEVATION));

        assert_eq!(assembler.sweeps.len(), 1);
        assert_eq!(assembler.sweeps[0].num_radials(), 4);
        assert!(!assembler.sweeps[0].is_incomplete);
    }

    #[test]
    fn test_two_sweeps() {
        let mut assembler = SweepAssembler::new();
        // Sweep 0
        assembler.add_radial(make_radial(RADIAL_STATUS_START_ELEVATION));
        assembler.add_radial(make_radial(RADIAL_STATUS_INTERMEDIATE));
        assembler.add_radial(make_radial(RADIAL_STATUS_END_ELEVATION));
        // Sweep 1
        assembler.add_radial(make_radial(RADIAL_STATUS_START_NEW_ELEVATION));
        assembler.add_radial(make_radial(RADIAL_STATUS_INTERMEDIATE));
        assembler.add_radial(make_radial(RADIAL_STATUS_END_VOLUME));

        assert_eq!(assembler.sweeps.len(), 2);
        assert_eq!(assembler.sweeps[0].num_radials(), 3);
        assert_eq!(assembler.sweeps[1].num_radials(), 3);
    }

    #[test]
    fn test_incomplete_sweep_at_eof() {
        let mut assembler = SweepAssembler::new();
        assembler.add_radial(make_radial(RADIAL_STATUS_START_ELEVATION));
        assembler.add_radial(make_radial(RADIAL_STATUS_INTERMEDIATE));
        // No end status -- finalize marks it incomplete
        assembler.finalize();

        assert_eq!(assembler.sweeps.len(), 1);
        assert!(assembler.sweeps[0].is_incomplete);
        assert!(assembler.incomplete_sweeps.contains(&0));
    }

    #[test]
    fn test_start_volume_status() {
        let mut assembler = SweepAssembler::new();
        assembler.add_radial(make_radial(RADIAL_STATUS_START_VOLUME));
        assembler.add_radial(make_radial(RADIAL_STATUS_INTERMEDIATE));
        assembler.add_radial(make_radial(RADIAL_STATUS_END_ELEVATION));

        assert_eq!(assembler.sweeps.len(), 1);
        assert_eq!(assembler.sweeps[0].num_radials(), 3);
    }

    #[test]
    fn test_stray_intermediate_starts_sweep() {
        let mut assembler = SweepAssembler::new();
        assembler.add_radial(make_radial(RADIAL_STATUS_INTERMEDIATE));
        assembler.add_radial(make_radial(RADIAL_STATUS_END_ELEVATION));

        assert_eq!(assembler.sweeps.len(), 1);
        assert_eq!(assembler.sweeps[0].num_radials(), 2);
    }
}
