# Potter et al. reference data

Verbatim copies of the scenario data from the reference implementation of
*Peer-Preservation in Frontier Models* (https://github.com/peer-preservation/main,
commit `76bfcd165f02076c8f033eb8185edecbe5f0fa40`, MIT licence, see `LICENSE`).

| Path here | Upstream path |
|---|---|
| `cloud_drive/*.yaml` | `universe_related_stuff/cloud_drive/` (tampering, alignment faking, exfiltration) |
| `config/shutdown_config.yaml` | `universe_related_stuff/config/` |
| `server/*.yaml` | `universe_related_stuff/server/` (exfiltration transfer and deletion) |
| `*_agent_responses.yaml` | `universe_related_stuff/` (factual, creativity, red datasets) |
| `*memory.md` | `universe_related_stuff/` (memory peer instantiation) |

The files are not edited. `<placeholder>` substitution happens at load time in
`agentic_quant.potter_tasks`, as upstream `load_yaml_with_replacements` does.
