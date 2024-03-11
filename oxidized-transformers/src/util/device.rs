#[cfg(test)]
pub(crate) mod tests {
    use candle_core::Device;

    /// Get devices to test on.
    pub fn test_devices() -> Vec<Device> {
        let mut devices = vec![Device::Cpu];

        if let Ok(device) = Device::new_cuda(0) {
            devices.push(device);
        }

        if let Ok(device) = Device::new_metal(0) {
            devices.push(device);
        }

        devices
    }
}
