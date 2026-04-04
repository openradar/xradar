use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet};

use crate::chunks::concatenate_chunks;
use crate::errors::NexradError;
use crate::moments::nexrad_to_cfradial;
use crate::msg2::decode_scan_data_flags;
use crate::msg5::{decode_vcp_sequencing, decode_vcp_supplemental};
use crate::parser::NexradLevel2;

/// Python-accessible NEXRAD Level2 file parser.
#[pyclass]
pub struct NexradRustFile {
    inner: NexradLevel2,
}

#[pymethods]
impl NexradRustFile {
    /// Create a new NexradRustFile from raw bytes.
    ///
    /// Args:
    ///     data: Raw file bytes
    ///     loaddata: If True, parse sweep data. If False, only parse headers.
    #[new]
    #[pyo3(signature = (data, loaddata=true))]
    fn new(data: &[u8], loaddata: bool) -> PyResult<Self> {
        let inner = NexradLevel2::parse(data.to_vec(), loaddata)?;
        Ok(NexradRustFile { inner })
    }

    /// Create from a list of chunk byte buffers (S/I/E files)
    #[staticmethod]
    #[pyo3(signature = (chunks, loaddata=true))]
    fn from_chunks(chunks: Vec<Vec<u8>>, loaddata: bool) -> PyResult<Self> {
        let concatenated = concatenate_chunks(&chunks)?;
        let inner = NexradLevel2::parse(concatenated, loaddata)?;
        Ok(NexradRustFile { inner })
    }

    /// Volume header as a Python dict
    #[getter]
    fn volume_header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let vh = &self.inner.volume_header;
        dict.set_item("tape", &vh.tape[..])?;
        dict.set_item("extension", &vh.extension[..])?;
        dict.set_item("date", vh.date)?;
        dict.set_item("time", vh.time)?;
        dict.set_item("icao", &vh.icao[..])?;
        Ok(dict)
    }

    /// Number of sweeps
    #[getter]
    fn num_sweeps(&self) -> usize {
        self.inner.sweeps.len()
    }

    /// Whether the file uses legacy MSG 1 format
    #[getter]
    fn is_legacy(&self) -> bool {
        self.inner.is_legacy
    }

    /// Set of incomplete sweep indices
    #[getter]
    fn incomplete_sweeps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let set = PySet::empty(py)?;
        for &idx in &self.inner.incomplete_sweeps {
            set.add(idx)?;
        }
        Ok(set)
    }

    /// MSG 5 data as a Python dict (or None if not present)
    #[getter]
    fn msg_5<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let msg5 = match &self.inner.msg5 {
            Some(m) => m,
            None => return Ok(None),
        };

        let dict = PyDict::new(py);
        dict.set_item("message_size", msg5.message_size)?;
        dict.set_item("pattern_type", msg5.pattern_type)?;
        dict.set_item("pattern_number", msg5.pattern_number)?;
        dict.set_item("number_elevation_cuts", msg5.number_elevation_cuts)?;
        dict.set_item("clutter_map_group_number", msg5.clutter_map_group_number)?;
        dict.set_item("doppler_velocity_resolution", msg5.doppler_velocity_resolution)?;
        dict.set_item("pulse_width", msg5.pulse_width)?;
        dict.set_item("vcp_sequencing", msg5.vcp_sequencing)?;
        dict.set_item("vcp_supplemental", msg5.vcp_supplemental)?;

        // Decoded sequencing
        let seq_dict = PyDict::new(py);
        let seq = &msg5.vcp_sequencing_decoded;
        seq_dict.set_item("num_elevations", seq.num_elevations)?;
        seq_dict.set_item("max_sails_cuts", seq.max_sails_cuts)?;
        seq_dict.set_item("sequence_active", seq.sequence_active)?;
        seq_dict.set_item("truncated_vcp", seq.truncated_vcp)?;
        dict.set_item("vcp_sequencing_decoded", seq_dict)?;

        // Decoded supplemental
        let sup_dict = PyDict::new(py);
        let sup = &msg5.vcp_supplemental_decoded;
        sup_dict.set_item("sails_vcp", sup.sails_vcp)?;
        sup_dict.set_item("num_sails_cuts", sup.num_sails_cuts)?;
        sup_dict.set_item("mrle_vcp", sup.mrle_vcp)?;
        sup_dict.set_item("num_mrle_cuts", sup.num_mrle_cuts)?;
        sup_dict.set_item("mpda_vcp", sup.mpda_vcp)?;
        sup_dict.set_item("base_tilt_vcp", sup.base_tilt_vcp)?;
        sup_dict.set_item("num_base_tilts", sup.num_base_tilts)?;
        dict.set_item("vcp_supplemental_decoded", sup_dict)?;

        // Elevation data
        let elev_list = PyList::empty(py);
        for elev in &msg5.elevation_data {
            let ed = PyDict::new(py);
            ed.set_item("elevation_angle", elev.elevation_angle)?;
            ed.set_item("channel_config", elev.channel_config)?;
            ed.set_item("waveform_type", elev.waveform_type)?;
            ed.set_item("super_resolution", elev.super_resolution)?;
            ed.set_item("prf_number", elev.prf_number)?;
            ed.set_item("prf_pulse_count", elev.prf_pulse_count)?;
            ed.set_item("azimuth_rate", elev.azimuth_rate)?;
            ed.set_item("ref_thresh", elev.ref_thresh)?;
            ed.set_item("vel_thresh", elev.vel_thresh)?;
            ed.set_item("sw_thresh", elev.sw_thresh)?;
            ed.set_item("zdr_thres", elev.zdr_thres)?;
            ed.set_item("phi_thres", elev.phi_thres)?;
            ed.set_item("rho_thres", elev.rho_thres)?;
            ed.set_item("edge_angle_1", elev.edge_angle_1)?;
            ed.set_item("dop_prf_num_1", elev.dop_prf_num_1)?;
            ed.set_item("dop_prf_pulse_count_1", elev.dop_prf_pulse_count_1)?;
            ed.set_item("supplemental_data", elev.supplemental_data)?;
            ed.set_item("edge_angle_2", elev.edge_angle_2)?;
            ed.set_item("dop_prf_num_2", elev.dop_prf_num_2)?;
            ed.set_item("dop_prf_pulse_count_2", elev.dop_prf_pulse_count_2)?;
            ed.set_item("edge_angle_3", elev.edge_angle_3)?;
            ed.set_item("dop_prf_num_3", elev.dop_prf_num_3)?;
            ed.set_item("dop_prf_pulse_count_3", elev.dop_prf_pulse_count_3)?;

            // Decoded supplemental
            let sd = PyDict::new(py);
            let sup = &elev.supplemental_data_decoded;
            sd.set_item("sails_cut", sup.sails_cut)?;
            sd.set_item("sails_sequence_number", sup.sails_sequence_number)?;
            sd.set_item("mrle_cut", sup.mrle_cut)?;
            sd.set_item("mrle_sequence_number", sup.mrle_sequence_number)?;
            sd.set_item("mpda_cut", sup.mpda_cut)?;
            sd.set_item("base_tilt_cut", sup.base_tilt_cut)?;
            ed.set_item("supplemental_data_decoded", sd)?;

            elev_list.append(ed)?;
        }
        dict.set_item("elevation_data", elev_list)?;

        Ok(Some(dict))
    }

    /// MSG 2 data as a Python dict (or None if not present)
    #[getter]
    fn msg_2<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let msg2 = match &self.inner.msg2 {
            Some(m) => m,
            None => return Ok(None),
        };

        let dict = PyDict::new(py);
        dict.set_item("rda_status", msg2.rda_status)?;
        dict.set_item("operability_status", msg2.operability_status)?;
        dict.set_item("control_status", msg2.control_status)?;
        dict.set_item("vcp_number", msg2.vcp_number)?;
        dict.set_item("rda_build", msg2.rda_build)?;
        dict.set_item("operational_mode", msg2.operational_mode)?;
        dict.set_item("super_res_status", msg2.super_res_status)?;
        dict.set_item("avset_status", msg2.avset_status)?;
        dict.set_item("scan_data_flags", msg2.scan_data_flags)?;

        let flags = decode_scan_data_flags(msg2.scan_data_flags);
        let flags_dict = PyDict::new(py);
        flags_dict.set_item("avset_enabled", flags.avset_enabled)?;
        flags_dict.set_item("ebc_enabled", flags.ebc_enabled)?;
        dict.set_item("scan_data_flags_decoded", flags_dict)?;

        Ok(Some(dict))
    }

    /// Metadata headers as a Python dict
    #[getter]
    fn meta_header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        fn msg_header_list<'py>(
            py: Python<'py>,
            headers: &[crate::messages::MsgHeader],
        ) -> PyResult<Bound<'py, PyList>> {
            let list = PyList::empty(py);
            for h in headers {
                let d = PyDict::new(py);
                d.set_item("size", h.size)?;
                d.set_item("channels", h.channels)?;
                d.set_item("type", h.msg_type)?;
                d.set_item("seq_id", h.seq_id)?;
                d.set_item("date", h.date)?;
                d.set_item("ms", h.ms)?;
                d.set_item("segments", h.segments)?;
                d.set_item("seg_num", h.seg_num)?;
                d.set_item("record_number", h.record_number)?;
                list.append(d)?;
            }
            Ok(list)
        }

        let mh = &self.inner.meta_headers;
        dict.set_item("msg_2", msg_header_list(py, &mh.msg_2)?)?;
        dict.set_item("msg_3", msg_header_list(py, &mh.msg_3)?)?;
        dict.set_item("msg_5", msg_header_list(py, &mh.msg_5)?)?;
        dict.set_item("msg_13", msg_header_list(py, &mh.msg_13)?)?;
        dict.set_item("msg_15", msg_header_list(py, &mh.msg_15)?)?;
        dict.set_item("msg_18", msg_header_list(py, &mh.msg_18)?)?;

        Ok(dict)
    }

    /// Get azimuth angles for a sweep as a numpy array
    fn get_azimuth<'py>(&self, py: Python<'py>, sweep: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let sw = self.get_sweep(sweep)?;
        let azimuths = sw.azimuths();
        Ok(PyArray1::from_vec(py, azimuths))
    }

    /// Get elevation angles for a sweep as a numpy array
    fn get_elevation<'py>(&self, py: Python<'py>, sweep: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let sw = self.get_sweep(sweep)?;
        let elevations = sw.elevations();
        Ok(PyArray1::from_vec(py, elevations))
    }

    /// Get collection times as (dates_array, ms_array) for a sweep
    fn get_time_components<'py>(
        &self,
        py: Python<'py>,
        sweep: usize,
    ) -> PyResult<(Bound<'py, PyArray1<u16>>, Bound<'py, PyArray1<u32>>)> {
        let sw = self.get_sweep(sweep)?;
        let dates = sw.collect_dates();
        let times = sw.collect_times_ms();
        Ok((
            PyArray1::from_vec(py, dates),
            PyArray1::from_vec(py, times),
        ))
    }

    /// Get number of radials in a sweep
    fn get_num_radials(&self, sweep: usize) -> PyResult<usize> {
        let sw = self.get_sweep(sweep)?;
        Ok(sw.num_radials())
    }

    /// Get available moment names for a sweep
    fn get_moment_names(&self, sweep: usize) -> PyResult<Vec<String>> {
        let sw = self.get_sweep(sweep)?;
        Ok(sw.moment_names())
    }

    /// Get moment data for a sweep as a 2D numpy array (nrays x ngates, u16)
    ///
    /// For 8-bit data, values are zero-extended to u16.
    fn get_moment_data<'py>(
        &self,
        py: Python<'py>,
        sweep: usize,
        moment: &str,
    ) -> PyResult<Bound<'py, PyArray2<u16>>> {
        let sw = self.get_sweep(sweep)?;

        if sw.is_legacy {
            return self.get_legacy_moment_data(py, sw, moment);
        }

        // Find ngates from first radial that has this moment
        let first_block = sw.radials.iter()
            .find_map(|r| r.moment_blocks.get(moment))
            .ok_or_else(|| NexradError::DataNotLoaded(format!(
                "Moment '{}' not found in sweep {}", moment, sweep
            )))?;

        let ngates = first_block.ngates as usize;
        let nrays = sw.radials.len();
        let word_size = first_block.word_size;

        let mut data = vec![0u16; nrays * ngates];

        for (ray_idx, radial) in sw.radials.iter().enumerate() {
            if let Some(block) = radial.moment_blocks.get(moment) {
                let row_start = ray_idx * ngates;
                if word_size == 16 {
                    if let Ok(values) = block.data_u16() {
                        let copy_len = values.len().min(ngates);
                        data[row_start..row_start + copy_len]
                            .copy_from_slice(&values[..copy_len]);
                    }
                } else {
                    // 8-bit: zero-extend to u16
                    let raw = block.data_u8();
                    let copy_len = raw.len().min(ngates);
                    for (i, &v) in raw[..copy_len].iter().enumerate() {
                        data[row_start + i] = v as u16;
                    }
                }
            }
        }

        let arr = PyArray1::from_vec(py, data);
        Ok(arr.reshape([nrays, ngates])?)
    }

    /// Get scale and offset for a moment variable in a sweep
    fn get_scale_offset(&self, sweep: usize, moment: &str) -> PyResult<(f32, f32)> {
        let sw = self.get_sweep(sweep)?;

        if sw.is_legacy {
            // Legacy MSG 1 has fixed scale/offset
            return Ok(match moment {
                "REF" => (2.0, 66.0),
                "VEL" | "SW " => (2.0, 129.0),
                _ => (1.0, 0.0),
            });
        }

        let block = sw.radials.iter()
            .find_map(|r| r.moment_blocks.get(moment))
            .ok_or_else(|| NexradError::DataNotLoaded(format!(
                "Moment '{}' not found in sweep {}", moment, sweep
            )))?;

        Ok((block.scale, block.offset))
    }

    /// Get range parameters (first_gate, gate_spacing, ngates) for a moment
    fn get_range_params(&self, sweep: usize, moment: &str) -> PyResult<(i16, i16, u16)> {
        let sw = self.get_sweep(sweep)?;

        if sw.is_legacy {
            let r = sw.legacy_radials.first()
                .ok_or_else(|| NexradError::InvalidSweep("Empty legacy sweep".to_string()))?;
            return Ok(match moment {
                "REF" => (r.header.first_gate_range_ref, r.header.gate_spacing_ref as i16, r.header.ngates_ref),
                _ => (r.header.first_gate_range_dop, r.header.gate_spacing_dop as i16, r.header.ngates_dop),
            });
        }

        let block = sw.radials.iter()
            .find_map(|r| r.moment_blocks.get(moment))
            .ok_or_else(|| NexradError::DataNotLoaded(format!(
                "Moment '{}' not found in sweep {}", moment, sweep
            )))?;

        Ok((block.first_gate, block.gate_spacing, block.ngates))
    }

    /// Get volume constant data (lat, lon, alt, etc.) from first radial of a sweep
    fn get_volume_data<'py>(&self, py: Python<'py>, sweep: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
        let sw = self.get_sweep(sweep)?;
        if sw.is_legacy || sw.radials.is_empty() {
            return Ok(None);
        }

        let vol = match sw.radials.iter().find_map(|r| r.volume_block.as_ref()) {
            Some(v) => v,
            None => return Ok(None),
        };

        let dict = PyDict::new(py);
        dict.set_item("latitude", vol.latitude)?;
        dict.set_item("longitude", vol.longitude)?;
        dict.set_item("site_height", vol.site_height)?;
        dict.set_item("feedhorn_height", vol.feedhorn_height)?;
        dict.set_item("calibration_constant", vol.calibration_constant)?;
        dict.set_item("horizontal_shv_tx_power", vol.horizontal_shv_tx_power)?;
        dict.set_item("vertical_shv_tx_power", vol.vertical_shv_tx_power)?;
        dict.set_item("system_differential_reflectivity", vol.system_differential_reflectivity)?;
        dict.set_item("initial_system_differential_phase", vol.initial_system_differential_phase)?;
        dict.set_item("vcp_number", vol.vcp_number)?;

        Ok(Some(dict))
    }

    /// Get radial constant data (unambig_range, noise, nyquist) from a sweep
    fn get_radial_data<'py>(&self, py: Python<'py>, sweep: usize, radial_idx: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
        let sw = self.get_sweep(sweep)?;
        if sw.is_legacy || radial_idx >= sw.radials.len() {
            return Ok(None);
        }

        let rad = match sw.radials[radial_idx].radial_block.as_ref() {
            Some(r) => r,
            None => return Ok(None),
        };

        let dict = PyDict::new(py);
        dict.set_item("unambiguous_range", rad.unambiguous_range)?;
        dict.set_item("noise_level_h", rad.noise_level_h)?;
        dict.set_item("noise_level_v", rad.noise_level_v)?;
        dict.set_item("nyquist_velocity", rad.nyquist_velocity)?;

        Ok(Some(dict))
    }

    /// Get the CfRadial2 name for a NEXRAD moment name
    #[staticmethod]
    fn moment_cfradial_name(nexrad_name: &str) -> String {
        nexrad_to_cfradial(nexrad_name).to_string()
    }

    /// Decode VCP sequencing bits (utility, exposed for testing)
    #[staticmethod]
    fn decode_vcp_sequencing_bits<'py>(py: Python<'py>, value: u16) -> PyResult<Bound<'py, PyDict>> {
        let decoded = decode_vcp_sequencing(value);
        let dict = PyDict::new(py);
        dict.set_item("num_elevations", decoded.num_elevations)?;
        dict.set_item("max_sails_cuts", decoded.max_sails_cuts)?;
        dict.set_item("sequence_active", decoded.sequence_active)?;
        dict.set_item("truncated_vcp", decoded.truncated_vcp)?;
        Ok(dict)
    }

    /// Decode VCP supplemental bits (utility, exposed for testing)
    #[staticmethod]
    fn decode_vcp_supplemental_bits<'py>(py: Python<'py>, value: u16) -> PyResult<Bound<'py, PyDict>> {
        let decoded = decode_vcp_supplemental(value);
        let dict = PyDict::new(py);
        dict.set_item("sails_vcp", decoded.sails_vcp)?;
        dict.set_item("num_sails_cuts", decoded.num_sails_cuts)?;
        dict.set_item("mrle_vcp", decoded.mrle_vcp)?;
        dict.set_item("num_mrle_cuts", decoded.num_mrle_cuts)?;
        dict.set_item("mpda_vcp", decoded.mpda_vcp)?;
        dict.set_item("base_tilt_vcp", decoded.base_tilt_vcp)?;
        dict.set_item("num_base_tilts", decoded.num_base_tilts)?;
        Ok(dict)
    }
}

// Private helper methods
impl NexradRustFile {
    fn get_sweep(&self, sweep: usize) -> Result<&crate::sweep::Sweep, NexradError> {
        self.inner.sweeps.get(sweep).ok_or_else(|| {
            NexradError::InvalidSweep(format!(
                "Sweep {} out of range (only {} sweeps)",
                sweep,
                self.inner.sweeps.len()
            ))
        })
    }

    fn get_legacy_moment_data<'py>(
        &self,
        py: Python<'py>,
        sw: &crate::sweep::Sweep,
        moment: &str,
    ) -> PyResult<Bound<'py, PyArray2<u16>>> {
        let first = sw.legacy_radials.first()
            .ok_or_else(|| NexradError::InvalidSweep("Empty legacy sweep".to_string()))?;

        let ngates = match moment {
            "REF" => first.header.ngates_ref as usize,
            _ => first.header.ngates_dop as usize,
        };
        let nrays = sw.legacy_radials.len();

        let mut data = vec![0u16; nrays * ngates];

        for (ray_idx, radial) in sw.legacy_radials.iter().enumerate() {
            let raw = match moment {
                "REF" => radial.ref_data.as_deref(),
                "VEL" => radial.vel_data.as_deref(),
                "SW " => radial.sw_data.as_deref(),
                _ => None,
            };

            if let Some(raw) = raw {
                let row_start = ray_idx * ngates;
                let copy_len = raw.len().min(ngates);
                for (i, &v) in raw[..copy_len].iter().enumerate() {
                    data[row_start + i] = v as u16;
                }
            }
        }

        let arr = PyArray1::from_vec(py, data);
        Ok(arr.reshape([nrays, ngates])?)
    }
}
