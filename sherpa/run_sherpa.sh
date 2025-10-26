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
  YAML=$(realpath "$YAML_FILE")
  YODA=$(basename "$DIRECTORY")
  SEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')
  echo "Entering $DIRECTORY and running Sherpa with SEED=$SEED"
  cd "$INITIAL_RUN" && ~/PATH/TO/SHERPA/INSTALLATION/bin/Sherpa -f "$YAML" -R "$SEED" -A "../$DIRECTORY/$YODA"
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
