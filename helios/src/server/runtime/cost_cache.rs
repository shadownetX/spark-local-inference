/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Persistent, stale graph duration priors for authenticated single-device cores.

use crate::{execution::ExecutionManifest, model::ModelConfig, scheduler::RequestScheduler};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
};

type CacheResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCHEMA: u32 = 1;
const MAX_BYTES: u64 = 1024 * 1024;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Scope {
    core: [u8; 32],
    gpu: [u8; 16],
    executable: [u8; 32],
    cuda_driver: i32,
    kernel_driver: Option<String>,
    abi: u32,
    prepared_sequence_tokens: usize,
    max_sequence_tokens: usize,
    max_active: usize,
    prefill_chunk_size: usize,
    graphs: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct Stored {
    schema: u32,
    scope: Scope,
    costs: Vec<(String, u64)>,
}

pub(super) struct Cache {
    path: PathBuf,
    scope: Scope,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn executable_digest() -> CacheResult<[u8; 32]> {
    let mut file = File::open("/proc/self/exe")?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(digest.finalize().into())
}

impl Cache {
    pub(super) fn new(
        root: PathBuf,
        core: [u8; 32],
        model: &ModelConfig,
        manifest: &ExecutionManifest,
    ) -> CacheResult<Self> {
        let gpu = crate::cuda::device_uuid(manifest.device)?;
        let mut graphs: Vec<_> = manifest.graphs.iter().map(|graph|
            format!("{:?}:{}:{}:{}:{}:{}:{:?}:{}", graph.kind,
                graph.batch_capacity, graph.draft_tokens, graph.token_capacity,
                graph.prefill_capacity, graph.packed_token_capacity, graph.sampling_mode, graph.name)
        ).collect();
        graphs.sort_unstable();
        let scope = Scope {
            core,
            gpu,
            executable: executable_digest()?,
            cuda_driver: crate::cuda::driver_version()?,
            kernel_driver: fs::read_to_string("/proc/driver/nvidia/version").ok(),
            abi: manifest.abi_version,
            prepared_sequence_tokens: model.prepared_sequence_tokens,
            max_sequence_tokens: model.limits.max_sequence_tokens,
            max_active: manifest.max_active,
            prefill_chunk_size: model.limits.prefill_chunk_size,
            graphs,
        };
        // A runtime upgrade replaces the incompatible record instead of
        // accumulating one file per executable.
        let path = root.join(format!("{}-{}.json", hex(&core), hex(&gpu)));
        Ok(Self { path, scope })
    }

    fn read_records(&self) -> CacheResult<Option<Vec<(String, u64)>>> {
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut bytes = Vec::new();
        file.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err("graph cost cache exceeds size limit".into());
        }
        let stored: Stored = serde_json::from_slice(&bytes)?;
        if stored.schema != SCHEMA || stored.scope != self.scope {
            return Ok(None);
        }
        Ok(Some(stored.costs))
    }

    pub(super) fn load(&self, scheduler: &mut RequestScheduler) -> CacheResult<usize> {
        match self.read_records()? {
            Some(costs) => Ok(scheduler.restore_graph_costs(&costs)?),
            None => Ok(0),
        }
    }

    fn write_records(&self, costs: &[(String, u64)]) -> CacheResult<usize> {
        let count = costs.iter().filter(|(_, nanos)| *nanos != 0).count();
        if count == 0 {
            return Ok(0);
        }
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema": SCHEMA, "scope": &self.scope, "costs": costs,
        }))?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err("graph cost cache exceeds size limit".into());
        }
        let parent = self.path.parent().ok_or("graph cost cache requires a directory")?;
        fs::create_dir_all(parent)?;
        // Distinct temporary names allow independent writers to publish
        // complete snapshots without truncating a live file.
        let mut nonce = [0u8; 16];
        SystemRandom::new().fill(&mut nonce).map_err(|_| "cannot create cache temporary name")?;
        let temporary = parent.join(format!(".graph-cost-{}.tmp", hex(&nonce)));
        let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temporary)?;
        let result = (|| -> std::io::Result<()> {
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &self.path)?;
            File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result?;
        Ok(count)
    }

    pub(super) fn save(&self, scheduler: &RequestScheduler) -> CacheResult<usize> {
        self.write_records(&scheduler.retained_graph_costs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Directory(PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }

    fn fixture() -> (Directory, Cache) {
        let mut nonce = [0u8; 16];
        SystemRandom::new().fill(&mut nonce).unwrap();
        let directory = Directory(std::env::temp_dir().join(format!("flare-graph-cost-{}", hex(&nonce))));
        let cache = Cache {
            path: directory.0.join("costs.json"),
            scope: Scope {
                core: [1; 32], gpu: [2; 16], executable: [3; 32],
                cuda_driver: 13020, kernel_driver: Some("driver".into()), abi: 1,
                prepared_sequence_tokens: 262144, max_sequence_tokens: 262144,
                max_active: 8, prefill_chunk_size: 4096, graphs: vec!["Decode:1".into()],
            },
        };
        (directory, cache)
    }

    #[test]
    fn graph_cost_cache_round_trips_and_replaces_a_complete_file() {
        let (_directory, cache) = fixture();
        assert!(cache.read_records().unwrap().is_none());
        let first = vec![("Decode:1".into(), 100), ("Decode:2".into(), 0)];
        assert_eq!(cache.write_records(&first).unwrap(), 1);
        assert_eq!(cache.read_records().unwrap(), Some(first));
        let second = vec![("Decode:1".into(), 200), ("Decode:2".into(), 300)];
        assert_eq!(cache.write_records(&second).unwrap(), 2);
        assert_eq!(cache.read_records().unwrap(), Some(second.clone()));
        assert_eq!(cache.write_records(&[("Decode:1".into(), 0)]).unwrap(), 0);
        assert_eq!(cache.read_records().unwrap(), Some(second));
        assert_eq!(fs::read_dir(cache.path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn graph_cost_cache_rejects_changed_identity_and_corrupt_or_oversized_records() {
        let (_directory, mut cache) = fixture();
        let costs = vec![("Decode:1".into(), 100)];
        cache.write_records(&costs).unwrap();
        cache.scope.core[0] ^= 1;
        assert!(cache.read_records().unwrap().is_none());
        cache.scope.core[0] ^= 1;
        cache.scope.gpu[0] ^= 1;
        assert!(cache.read_records().unwrap().is_none());
        cache.scope.gpu[0] ^= 1;
        cache.scope.executable[0] ^= 1;
        assert!(cache.read_records().unwrap().is_none());
        cache.scope.executable[0] ^= 1;
        cache.scope.cuda_driver += 1;
        assert!(cache.read_records().unwrap().is_none());
        cache.scope.cuda_driver -= 1;
        cache.scope.max_sequence_tokens -= 1;
        assert!(cache.read_records().unwrap().is_none());
        cache.scope.max_sequence_tokens += 1;
        cache.scope.graphs.push("Prefill:1".into());
        assert!(cache.read_records().unwrap().is_none());
        cache.scope.graphs.pop();
        assert_eq!(cache.read_records().unwrap(), Some(costs.clone()));
        let too_large = vec![("x".repeat(MAX_BYTES as usize), 1)];
        assert!(cache.write_records(&too_large).is_err());
        assert_eq!(cache.read_records().unwrap(), Some(costs));
        fs::write(&cache.path, b"{").unwrap();
        assert!(cache.read_records().is_err());
        fs::write(&cache.path, vec![0; MAX_BYTES as usize + 1]).unwrap();
        assert!(cache.read_records().is_err());
    }
}
