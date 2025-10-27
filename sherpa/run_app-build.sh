#!/bin/bash

source /etc/profile.d/modules.sh
module load mpi/openmpi-x86_64
export PATH=/usr/bin:$PATH
source /PATH/TO/RIVET/INSTALLATION/rivetenv.sh

# Parse arguments
outdir=""
folder=""
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
            args+=("$1")
            shift
            ;;
    esac
done

# Compose output file name
base=$(basename "$folder")
outfile="${outdir%/}/${base}.json"

exec /PATH/TO/APPRENTICE/INSTALLATION/app-build "${args[@]}" -o "$outfile" -w "$folder"
