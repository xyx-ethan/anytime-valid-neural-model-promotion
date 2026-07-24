# Neural experiment execution provenance

The two Python files in this directory are byte-for-byte copies of the source
used for the five confirmation runs. Their SHA-256 values are recorded in
`source_data/computational_environment.json`.

The script in `analysis/run_plasticity_retention_stream.py` is a self-contained
equivalent. It inlines the model, loss, training, and evidence helpers and adds
input validation. Independent recalculation reproduced all 60 paired-difference
arrays and every reported crossing, evidence value, and batch-aligned
eligibility time.
