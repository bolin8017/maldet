# Event Stream

Every stage emits structured events through the `EventLogger` protocol.
The `CompositeEventLogger` fans out to three sinks simultaneously:
JSONL file, stdout, and MLflow.

## Event kinds

| Kind | Required fields |
|------|----------------|
| `stage_begin` | `stage` |
| `stage_end` | `stage`, `status` |
| `data_loaded` | _(none)_ |
| `epoch_begin` | `epoch` |
| `epoch_end` | `epoch` |
| `metric` | `name`, `value` |
| `artifact_written` | `path` |
| `checkpoint_saved` | `path` |
| `warning` | `message` |
| `error` | `message` |

Payload validation is enforced by `maldet.events.kinds.validate_payload`.
Extra fields are allowed and round-trip transparently.

## JSONL format

Each line in `events.jsonl` is a JSON object with a leading `ts` (ISO-8601
UTC) timestamp followed by `kind` and the event payload:

```json
{"ts": "2026-04-24T08:00:01.234567Z", "kind": "stage_begin", "stage": "train"}
{"ts": "2026-04-24T08:00:01.300000Z", "kind": "data_loaded", "n_train": 50000}
{"ts": "2026-04-24T08:00:05.100000Z", "kind": "metric", "name": "val_accuracy", "value": 0.9812}
{"ts": "2026-04-24T08:00:05.200000Z", "kind": "stage_end", "stage": "train", "status": "success"}
```

Each line is fsynced immediately, so a pod kill does not lose buffered events.

## Stdout sink

The `StdoutEventLogger` prefixes each line with `maldet.event: ` so platform
log collectors can filter maldet events from application logs.
