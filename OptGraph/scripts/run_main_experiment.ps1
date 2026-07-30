$ErrorActionPreference = "Stop"

# Configure these variables before running.
$env:DATASET_FILENAME = if ($env:DATASET_FILENAME) { $env:DATASET_FILENAME } else { "NL4OPT_old.csv" }
$env:QUERY_COLUMN = if ($env:QUERY_COLUMN) { $env:QUERY_COLUMN } else { "Query" }
$env:ENABLE_VALIDATION_AGENT = if ($env:ENABLE_VALIDATION_AGENT) { $env:ENABLE_VALIDATION_AGENT } else { "1" }
$env:MAX_SAMPLES = if ($env:MAX_SAMPLES) { $env:MAX_SAMPLES } else { "0" }

Set-Location (Split-Path -Parent $PSScriptRoot)
python WorkFlowGraphRAG.py
