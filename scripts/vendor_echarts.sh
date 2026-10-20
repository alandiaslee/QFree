#!/bin/sh
# Vendor echarts.min.js into web/ (one-time; CDN fallback exists in index.html).
set -e
cd "$(dirname "$0")/.."
curl -fL --retry 3 -o web/echarts.min.js \
  https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js
ls -lh web/echarts.min.js
