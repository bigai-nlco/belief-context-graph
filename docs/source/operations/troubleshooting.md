---
title: "Troubleshooting"
description: "Common installation, model, server, merge, and viewer problems."
icon: "wrench"
---

<AccordionGroup>

<Accordion title="`bcg` opens the Agent when I expected help">
Running `bcg` without a subcommand launches the reference Agent. Use `bcg --help` or `bcg construct --help`.
</Accordion>

<Accordion title="The graph server is not ready">
Check:

```bash
curl -s http://127.0.0.1:8848/health
tail -f ~/.bcg/logs/graph-server.log
```

Confirm model endpoints, credentials, config paths, and local model assets.
</Accordion>

<Accordion title="The hybrid backend fails during configuration">
Copy the full `bcg/model_config.example.json`. Hybrid configuration normalizers require complete sections and report missing keys.
</Accordion>

<Accordion title="No beliefs are extracted">
Check `min_content_len`, model output logs, prompt logs, evidence mode, and whether the turn was finalized as a complete message.
</Accordion>

<Accordion title="Too many duplicate nodes">
Increase merge coverage by lowering the similarity threshold carefully, enabling incremental merge, and considering merge verification.
</Accordion>

<Accordion title="Distinct beliefs are being merged">
Raise the threshold, enable model verification, and inspect `logs/merge_*` before choosing a production value.
</Accordion>

<Accordion title="The viewer cannot load samples">
Run `python3 dashboard/bcg_viewer/serve_viewer.py`, or use **Open data folder** and select a directory containing compatible results or stream files.
</Accordion>

<Accordion title="A graph remains in server memory">
Finalize it and call:

```bash
curl -X POST localhost:8848/release \
  -H 'content-type: application/json' \
  -d '{"problem_id":"p1"}'
```
</Accordion>

<Accordion title="A streamed message was split into multiple beliefs unexpectedly">
Ensure fragments set `is_message_end=false` until the final fragment. The server only ingests the assembled message when it is complete.
</Accordion>

</AccordionGroup>
