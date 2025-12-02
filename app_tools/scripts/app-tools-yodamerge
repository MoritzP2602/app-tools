#!/bin/bash

REMOVE_SUBDIRS=false
POSITIONAL=()

while [ $# -gt 0 ]; do
    case "$1" in
        --rm)
            REMOVE_SUBDIRS=true; shift ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

set -- "${POSITIONAL[@]}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--rm] <folder1> [<folder2> ...] [nproc]"
    echo "  <folder> : Directory containing subfolders with YODA files"
    echo "  [nproc]  : Number of parallel jobs (default: 4)"
    echo "  --rm     : (Optional) Remove merged subdirectories after successful merge to free space"
    echo "If subdirectories have their own subfolders, merges all .yoda/.yoda.gz files from nested subdirectories into a single .yoda file in each subdirectory."
    echo "If subdirectories do not have subfolders, merges all .yoda/.yoda.gz files from all subdirectories into a single .yoda file in the parent folder."
    exit 1
fi

is_number() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

FOLDERS=()
NPROC=4

for arg in "$@"; do
    if is_number "$arg" && [ "$arg" = "${@: -1}" ]; then
        NPROC="$arg"
    else
        FOLDERS+=("$arg")
    fi
done

if [ ${#FOLDERS[@]} -eq 0 ]; then
    echo "Error: No folders provided!"
    exit 1
fi

merge_dir() {
    dir="$1"
    YODA=$(basename "$dir")
    OUTFILE="$dir/${YODA}.yoda"
    if [ -f "$OUTFILE" ]; then
        echo "Skipping $dir: $OUTFILE already exists..."
        return
    fi

    mapfile -t FILES < <(find "$dir" -mindepth 2 -maxdepth 2 -type f \( -name '*.yoda' -o -name '*.yoda.gz' \))

    if [ "${#FILES[@]}" -eq 0 ]; then
        echo "No .yoda.gz/.yoda files found in subdirectories of $dir."
    else
        echo "Merging YODA files into $OUTFILE..."
        if command -v yodamerge >/dev/null 2>&1; then
            yodamerge "${FILES[@]}" -o "$OUTFILE"
            if [ "$REMOVE_SUBDIRS" = true ] && [ -f "$OUTFILE" ]; then
                for subdir in "$dir"/*; do
                    if [ -d "$subdir" ]; then
                        echo "Removing $subdir..."
                        rm -rf "$subdir"
                    fi
                done
            fi
        else
            echo "Error: yodamerge command not found. Please install YODA or adjust the script to point to the correct path!"
            exit 1
        fi
    fi
}

merge_flat() {
    PREFIX="${1%/}"
    YODA=$(basename "$PREFIX")
    OUTFILE="$PREFIX/${YODA}.yoda"
    
    if [ -f "$OUTFILE" ]; then
        echo "Skipping: $OUTFILE already exists..."
        return
    fi
    
    mapfile -t FILES < <(find "$PREFIX" -mindepth 2 -maxdepth 2 -type f \( -name '*.yoda' -o -name '*.yoda.gz' \))
    
    if [ "${#FILES[@]}" -eq 0 ]; then
        echo "No .yoda/.yoda.gz files found in subdirectories of $PREFIX."
    else
        echo "Merging YODA files from all subdirectories into $OUTFILE..."
        if command -v yodamerge >/dev/null 2>&1; then
            yodamerge "${FILES[@]}" -o "$OUTFILE"
            if [ "$REMOVE_SUBDIRS" = true ] && [ -f "$OUTFILE" ]; then
                for subdir in "$PREFIX"/*; do
                    if [ -d "$subdir" ]; then
                        echo "Removing $subdir..."
                        rm -rf "$subdir"
                    fi
                done
            fi
        else
            echo "Error: yodamerge command not found. Please install YODA or adjust the script to point to the correct path!"
            exit 1
        fi
    fi
}

export -f merge_dir
export -f merge_flat
export REMOVE_SUBDIRS

ALL_NESTED_DIRS=()
ALL_FLAT_DIRS=()

for folder in "${FOLDERS[@]}"; do
    PREFIX="${folder%/}"
    if [ ! -d "$PREFIX" ]; then
        echo "Warning: No directory named $PREFIX! Skipping..."
        continue
    fi

    has_nested_subdirs=false
    for dir in "$PREFIX"/*; do
        if [ -d "$dir" ] && [ "$(find "$dir" -mindepth 1 -maxdepth 1 -type d | wc -l)" -gt 0 ]; then
            has_nested_subdirs=true
            break
        fi
    done

    if [ "$has_nested_subdirs" = true ]; then
        while IFS= read -r dir; do
            ALL_NESTED_DIRS+=("$dir")
        done < <(find "$PREFIX" -mindepth 1 -maxdepth 1 -type d)
    else
        ALL_FLAT_DIRS+=("$PREFIX")
    fi
done

if [ ${#ALL_NESTED_DIRS[@]} -gt 0 ]; then
    echo "Processing ${#ALL_NESTED_DIRS[@]} nested directories with $NPROC parallel jobs..."
    printf "%s\n" "${ALL_NESTED_DIRS[@]}" | xargs -n 1 -P "$NPROC" bash -c 'merge_dir "$0"'
fi

if [ ${#ALL_FLAT_DIRS[@]} -gt 0 ]; then
    echo "Processing ${#ALL_FLAT_DIRS[@]} flat directories with $NPROC parallel jobs..."
    printf "%s\n" "${ALL_FLAT_DIRS[@]}" | xargs -n 1 -P "$NPROC" bash -c 'merge_flat "$0"'
fi
