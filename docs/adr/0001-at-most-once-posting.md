# Posts are at-most-once, guarded by a write-ahead Publishing record

Telegram's Bot API has no idempotency key, so after an ambiguous failure (a timeout after the request was sent, or a crash between sending and saving) we cannot tell whether the Post went out. We record the Article as Publishing before calling Telegram. A later Run that finds a Publishing record never posts it again: it marks the Article Unconfirmed and alerts the admin once. Only failures that prove nothing was delivered (no connection, 429, other 4xx) allow a retry. Timeouts and 5xx count as possibly delivered. A Duplicate in a public channel costs more than a missed Article, and the alert keeps the loss visible.

## Considered Options

- **At-least-once (the previous behaviour):** retry on any failure. Each ambiguous failure turned into a Duplicate.
- **At-most-once without a record:** do not retry ambiguous failures. Losses are silent.
