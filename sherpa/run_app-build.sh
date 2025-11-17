#!/bin/bash

# Record the start time (epoch + formatted)
start_epoch=$(date +%s)
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Job started at: $start_time"
echo ""

### --------------------------------------------------- ###

source /etc/profile.d/modules.sh
module load mpi/openmpi-x86_64
export PATH=/usr/bin:$PATH
source /PATH/TO/RIVET/INSTALLATION/rivetenv.sh

outdir=""
folder=""
newscan=""
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o)
            outdir="$2"
            shift 2
            ;;
        -w)
            folder="$2"
            shift 2
            ;;
        *)
            if [ -z "$newscan" ] && [ -d "$1" ]; then
                newscan="$1"
                shift
            else
                args+=("$1")
                shift
            fi
            ;;
    esac
done

if [ -z "$newscan" ] || [ ! -d "$newscan" ]; then
    echo "Newscan directory not found!"
    exit 1
fi

if [ -z "$folder" ] || [ ! -f "$folder" ]; then
    echo "Weight file not found!"
    exit 1
fi

newscan=$(realpath "$newscan")
folder=$(realpath "$folder")
outdir=$(realpath "$outdir")

echo "NEWSCAN     : $newscan"
echo "WEIGHT_FILE : $folder"
echo "OUTDIR      : $outdir"
echo ""

echo "Copying data to $TMPDIR..."
cp -r "$newscan" "$TMPDIR/"
cp "$folder" "$TMPDIR/"

newscan_basename=$(basename "$newscan")
folder_basename=$(basename "$folder")

outfile="${folder_basename%.txt}.json"

cd "$TMPDIR"
echo "Running app-build..."
/PATH/TO/APPRENTICE/INSTALLATION/app-build "$newscan_basename" "${args[@]}" -o "$outfile" -w "$folder_basename"

echo "Copying results back to $outdir..."
mkdir -p "$outdir"
cp "$outfile" "$outdir/"

### --------------------------------------------------- ###

# Record end time
end_epoch=$(date +%s)
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "Job ended at:   $end_time"

# Calculate elapsed time
elapsed=$(( end_epoch - start_epoch ))

# Convert seconds to D-HH:MM:SS
days=$(( elapsed / 86400 ))
hours=$(( (elapsed % 86400) / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

printf "Total elapsed time: %d-%02d:%02d:%02d\n" "$days" "$hours" "$minutes" "$seconds"
