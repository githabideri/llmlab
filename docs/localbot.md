# LocalBot Commands

These are OpenClaw LocalBot helper commands provided by the **localbot-ctl** plugin.
Repo: https://github.com/githabideri/localbot-ctl

## Working commands

- `/lbh` — help/command summary
- `/lbs` — status (active backend + model + slot/cache state)
- `/lbe` — list endpoints + probe results
- `/lbp` — quick perf benchmark on active endpoint
- `/lbw <backend>` — switch backend (`llama-cpp | vllm | stop`)
- `/lbn <room>` — reset LocalBot session (room-scoped)

## Architecture (sanitized)

```text
config/inference-endpoints.json  -> endpoint registry + optional wechsler integration
config/localbot-models.json      -> model metadata (speed/context aliases)
config/localbot-rooms.json       -> room map + reset policy
plugins/localbot-ctl/            -> command implementation
```

## Notes

- Room auto-detection for `/lbn` is limited by command context; explicit room args are more reliable.
- Reasoning budgets and prompt policy are controlled by backend service config + agent prompt policy, not by `localbot-ctl` itself.
