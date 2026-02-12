# LocalBot Commands

These are OpenClaw LocalBot helper commands provided by the **localbot-ctl** plugin.
Repo: https://github.com/githabideri/localbot-ctl

## Working commands

Use `/lbh` for the command summary, `/lbs` for active backend/model/slot state, `/lbe` for endpoint probe output, and `/lbp` for a quick performance check on the active endpoint.

For control actions, `/lbw <backend>` switches runtime backend (`llama-cpp | vllm | stop`), and `/lbn <room>` resets a LocalBot session in the specified room.

## Architecture (sanitized)

```text
config/inference-endpoints.json  -> endpoint registry + optional wechsler integration
config/localbot-models.json      -> model metadata (speed/context aliases)
config/localbot-rooms.json       -> room map + reset policy
plugins/localbot-ctl/            -> command implementation
```

## Notes

Room auto-detection for `/lbn` is limited by command context, so an explicit room argument remains the reliable path. Reasoning budgets and prompt policy are controlled by backend service configuration plus agent prompt policy, not by `localbot-ctl` itself.
