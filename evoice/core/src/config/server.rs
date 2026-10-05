use std::net::{IpAddr, Ipv4Addr};

use serde::{Deserialize, Serialize};

use crate::config::log::LogConfig;

/// What every service shares: the bind address and logging. Ports live in each service's section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: IpAddr,
    pub log: LogConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            log: LogConfig::default(),
        }
    }
}
