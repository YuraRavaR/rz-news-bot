# Everything runs on free tiers

The bot is a hobby project. It runs only on free plans: GitHub Actions (free for public repos on standard runners), the Gemini API free tier, Turso, healthchecks.io Hobbyist and the Telegram Bot API. We add no paid services. Any change is checked for what it costs in quota, and Gemini daily requests are the scarcest resource. That is why cheap non-LLM mechanisms come first (URL normalisation, title similarity, time windows), and extra Gemini calls are a last resort that has to be justified with data.

## Consequences

- Running more often than about every 3 hours is ruled out: each extra Run spends Gemini quota on the same Articles.
- Running out of Gemini quota is a normal event, not an exception. Articles hit by it are Deferred and do not spend their Error attempts.
- In a public repo GitHub disables scheduled workflows after 60 days without repository activity, and it can delay or drop scheduled runs. An external dead-man's switch (healthchecks.io) is the safety net, not a paid scheduler.
