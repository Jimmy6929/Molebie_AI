# molebie-satellite

Standalone satellite-side blob storage service for [Molebie AI](https://github.com/Jimmy6929/Molebie_AI)
fleets. Install on a second machine to extend a Molebie primary with
content-addressable storage over Tailscale.

A satellite is a passive blob store — no database, no LLM models, no
webapp. The primary tiers cold documents to it under disk pressure;
reads come back transparently. See the parent repo's documentation
for the broader fleet architecture.

## Install (any OS — macOS, Linux, Windows)

```bash
pipx install 'git+https://github.com/Jimmy6929/Molebie_AI.git@main#subdirectory=satellite_storage'
```

Then on the primary:

```bash
molebie-ai extend invite --role storage
```

Copy the printed one-liner onto the new satellite machine and paste it.
The installer handles platform-specific service registration (launchd
on macOS, systemd on Linux, Task Scheduler on Windows) so the satellite
survives reboots.

## Manual usage

```bash
molebie-satellite install --primary <ip>   # 6-phase wizard: prereqs, data dir,
                                           #   OS service install, start, register, verify
molebie-satellite uninstall [--purge]      # remove the service unit; --purge also
                                           #   deletes the blob data directory
molebie-satellite join --primary <ip>      # one-shot register without installing as a service
molebie-satellite serve                    # run the blob service in the foreground
molebie-satellite version                  # print package version
```

The `install` wizard installs `molebie-satellite serve` as the appropriate
OS service for your platform — launchd agent on macOS, systemd user unit on
Linux, Scheduled Task on Windows — so the satellite survives reboots. Pass
`--foreground` to skip the OS service install (useful for dev work where
you want to run the satellite manually).

## Requirements

- Python 3.13+
- Tailscale installed and signed in to the same tailnet as the primary
- pipx (`python3 -m pip install --user pipx && python3 -m pipx ensurepath`)

## License

MIT
