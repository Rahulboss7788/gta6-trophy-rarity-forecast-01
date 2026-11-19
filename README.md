# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rahulboss7788/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rahulboss7788/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Leonida Odds — a refresh-aware forecaster for GTA VI trophy earn rates. For every PS5 trophy it forecasts five quantiles (5/25/50/75/95%) of its Exophase earn rate at the case's target, scored by pinball loss on the logit scale. It fits each trophy's launch S-curve (logit rate against log time, recency-weighted), expects Exophase's stale page to refresh and dilute every rate (the launch-night page sat frozen at 638 players tracked until a refresh to 3,868 cut every rate), and sets honest, wide spreads in logit units — calibrated so the median sits on the refresh dilution the house baseline's published practice numbers imply, instead of trusting the last read with a narrow band.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network, deterministic — the same input always produces the same output bytes. The program reads every `*.in` case in the input directory and writes one `*.out` forecast per case.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
