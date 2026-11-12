#!/bin/bash
set -e
curl -s http://localhost:8080/predictions/storymodel -H "Content-Type: application/json" \
  -d '{"prompt":"One night, the library door opened by itself.","max_new_tokens":50,"temperature":0.8}'
echo
