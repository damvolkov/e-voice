use std::time::Duration;

/// Process-wide CPU time and peak resident memory, read from `/proc/self` (Linux).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Probe {
    pub cpu: Duration,
    pub peak_rss_mb: f64,
}

impl Probe {
    // ##### PRIVATE #####

    fn read_cpu() -> Option<Duration> {
        let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
        let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
        let ticks = |index: usize| fields.get(index).and_then(|value| value.parse::<u64>().ok());
        let total = ticks(11)?.saturating_add(ticks(12)?);
        Some(Duration::from_millis(total.saturating_mul(10)))
    }

    fn read_peak() -> Option<f64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
        let kib: u32 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(f64::from(kib) / 1024.0)
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// Zeroes when `/proc` is unavailable; clock ticks are assumed at the Linux default of 100 Hz.
    #[must_use]
    pub fn read() -> Self {
        Self {
            cpu: Self::read_cpu().unwrap_or_default(),
            peak_rss_mb: Self::read_peak().unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::probe::Probe;

    #[test]
    fn test_read_reports_this_process() {
        let before = Probe::read();
        let spin: u64 = (0..20_000_000u64).fold(0, u64::wrapping_add);
        let after = Probe::read();
        assert!(spin > 0);
        assert!(after.cpu >= before.cpu);
        assert!(after.peak_rss_mb > 1.0);
    }
}
