# LocalBot Commands

These are OpenClaw LocalBot helper commands via the **localbot-ctl** plugin.

## Working commands
- `/lbh` — help
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
- Room auto-detection for `/lbn` is limited by command context; explicit room arg is used.
- Reasoning budgets/prompt policy are managed by backend service config + agent prompt policy (not by localbot-ctl).
