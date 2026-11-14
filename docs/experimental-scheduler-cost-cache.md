# Experimental scheduler graph cost persistence

Proof of concept for single-rank Linux serving. Disabled by default.

Set `FLARE_EXPERIMENTAL_SCHEDULER_COST_CACHE` to a writable file path **unique to
one prepared export and deployment**. The parent directory must already exist;
mount it persistently when serving in a container. Do not share the path among
concurrent model runtimes. Clear the file after changing drivers, tuning,
hardware clocks or prepared export. Multi-rank serving ignores this option.

On startup a matching, bounded JSON snapshot supplies retained physical graph
duration priors. The model identity, GPU UUID, executable SHA-256, ABI, selected
context and graph catalog must match. The model identity alone is not an exact
prepared-export digest; the dedicated path is required. Alpha is intentionally
not a key: it changes expected token yield rather than physical graph identity.

Restored timings are known but stale. They are eligible for ordinary ranking,
not prioritized as unknown, and require normal isolated live feedback. No old
monotonic timestamp, replay window, acceptance history or exploration credit
is restored. A valid live sample replaces the prior. Unseen graphs remain
unknown. Mismatched, missing or unreadable files fall back to ordinary discovery.

The worker saves after a clean drained shutdown, using a temporary file and
rename. There is no disk I/O in the scheduling loop. This prototype does not
save on a crash or during long-running sessions; no durability guarantee is
made for sudden power loss. It is not a production configuration interface.

## Validation plan

- CPU: restore known/stale costs, exact catalog validation, atomic rejection,
  ordinary (not discovery) classification, and fresh observations replacing restored values.
- GB10: same prepared core, runtime and asynchronous workload, compare a fresh
  cache file vs a populated cache after graceful restart. Record discovery
  time/count, time to first token, streaming gaps, total wall time and throughput.
- Both alpha values can reuse the same physical priors. Include unseen graph
  shapes and changed/mismatched scope as controls.
- No expected performance gain is claimed until the hardware test.
