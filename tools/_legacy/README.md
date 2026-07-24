# Legacy target_t tools (non-functional)

These scripts were diagnostics for the **deprecated target_t-index conditioning path**
(`TIndexEmbedder` / `target_t_embedder` in the aggregator, and the `use_t_pose_embedding` /
`use_target_t_pose_embedding` flags). That path was **removed** when the model switched to
reference-slice conditioning (the `camera_token` anchor).

They are archived here for provenance only. They import `TIndexEmbedder` (or monkey-patch a
`target_t_embedder` onto the aggregator) and will `ImportError` / no-op if run — the model-side
machinery they depend on no longer exists. Restoring them would require reintroducing the full
target_t path, which contradicts its retirement.
