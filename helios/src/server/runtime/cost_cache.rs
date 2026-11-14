/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Experimental single-rank, best-effort graph timing priors.
//! No filesystem operations occur in the scheduling loop.

use crate::{execution::ExecutionManifest, model::ModelConfig, scheduler::RequestScheduler};
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
    model: [u8; 32],
    gpu: [u8; 16],
    executable: [u8; 32],
    abi: u32,
    context: usize,
    max_active: usize,
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

fn executable_digest() -> CacheResult<[u8; 32]> {
    // Linux /proc/self/exe refers to this running executable, even if replaced on disk.
    let mut file = File::open("/proc/self/exe")?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 { break; }
        digest.update(&buffer[..n]);
    }
    Ok(digest.finalize().into())
}

impl Cache {
    pub(super) fn new(path: PathBuf, model: &ModelConfig, manifest: &ExecutionManifest) -> CacheResult<Self> {
        if path.file_name().is_none() {
            return Err("scheduler cost cache requires a file path".into());
        }
        let mut graphs: Vec<_> = manifest.graphs.iter().map(|graph|
            format!("{:?}:{}:{}:{}:{}:{}:{:?}:{}", graph.kind,
                graph.batch_capacity, graph.draft_tokens, graph.token_capacity,
                graph.prefill_capacity, graph.packed_token_capacity, graph.sampling_mode, graph.name)
        ).collect();
        graphs.sort_unstable();
        Ok(Self {
            path,
            scope: Scope {
                model: model.identity,
                gpu: crate::cuda::device_uuid(manifest.device)?,
                executable: executable_digest()?,
                abi: manifest.abi_version,
                context: manifest.context,
                max_active: manifest.max_active,
                graphs,
            },
        })
    }

    pub(super) fn load(&self, scheduler: &mut RequestScheduler) -> CacheResult<usize> {
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error.into()),
        };
        let mut bytes = Vec::new();
        file.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err("scheduler cost cache exceeds size limit".into());
        }
        let stored: Stored = serde_json::from_slice(&bytes)?;
        if stored.schema != SCHEMA || stored.scope != self.scope {
            return Err("scheduler cost cache scope mismatch".into());
        }
        Ok(scheduler.restore_graph_costs(&stored.costs)?)
    }

    pub(super) fn save(&self, scheduler: &RequestScheduler) -> CacheResult<usize> {
        let costs = scheduler.retained_graph_costs();
        let count = costs.iter().filter(|(_, nanos)| *nanos != 0).count();
        if count == 0 { return Ok(0); }
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema": SCHEMA, "scope": &self.scope, "costs": costs,
        }))?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err("scheduler cost cache exceeds size limit".into());
        }
        // This file is per deployment. A failed or interrupted write never
        // truncates the last complete snapshot.
        let mut name = self.path.file_name().ok_or("invalid cache path")?.to_os_string();
        name.push(format!(".{}.tmp", std::process::id()));
        let temporary = self.path.with_file_name(name);
        let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temporary)?;
        let result = (|| -> std::io::Result<()> {
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &self.path)?;
            Ok(())
        })();
        if result.is_err() { let _ = fs::remove_file(&temporary); }
        result?;
        Ok(count)
    }
}
