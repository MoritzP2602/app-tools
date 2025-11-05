#!/bin/bash

# Record the start time (epoch + formatted)
start_epoch=$(date +%s)
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Job started at: $start_time"

### --------------------------------------------------- ###

INITIAL_RUN="Initial-run"
DIRECTORY="$1"
if [ -d "$INITIAL_RUN" ]; then
  if [ -d "$DIRECTORY" ]; then
    YAML_FILE=$(find "$DIRECTORY" -maxdepth 1 -name "*.yaml" | head -n 1)
    if [ -z "$YAML_FILE" ]; then
      YAML_FILE=$(find "$DIRECTORY/.." -maxdepth 1 -name "*.yaml" | head -n 1)
      if [ -z "$YAML_FILE" ]; then
      	echo "No .yaml file found!"
        exit 1
      fi
    fi
  else
    echo "Folder $DIRECTORY not found!"
    exit 1
  fi

  INITIAL_RUN=$(realpath "$INITIAL_RUN")
  YAML=$(realpath "$YAML_FILE")
  YODA_BASENAME=$(basename "$DIRECTORY")
  YODA="$YODA_BASENAME.yoda.gz"
  OUTDIR=$(realpath "$DIRECTORY")
  SEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')

  echo ""
  echo "INITIAL_RUN : $INITIAL_RUN"
  echo "YAML        : $YAML"
  echo "YODA        : $YODA"
  echo "OUTDIR      : $OUTDIR"
  echo "SEED        : $SEED"
  echo ""

  cp -r "$INITIAL_RUN"/Process "$INITIAL_RUN"/Results.zip* "$TMPDIR"
  cd "$TMPDIR" && /PATH/TO/SHERPA/INSTALLATION/bin/Sherpa -f "$YAML" -R "$SEED"
  cp -r Analysis.yoda.gz "$OUTDIR/$YODA"
else
  echo "$INITIAL_RUN directory not found!"
  exit 1
fi

### --------------------------------------------------- ###

# Record end time
end_epoch=$(date +%s)
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Job ended at:   $end_time"

# Calculate elapsed time
elapsed=$(( end_epoch - start_epoch ))

# Convert seconds to D-HH:MM:SS
days=$(( elapsed / 86400 ))
hours=$(( (elapsed % 86400) / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

printf "Total elapsed time: %d-%02d:%02d:%02d\n" "$days" "$hours" "$minutes" "$seconds"
