#!/bin/bash
# To be used a submit command for parsecmgmt, a.k.a. run 
# $ sudo parsecmgmt -a run -p <parsec benchmark> -i <sim size> -s "`pwd`/trace_parsec_bench.sh <-v> <-o output_path>"

# Initialize verbose flag
verbose=false
output_file="/home/garvalov/ml_prefetching_project_2/dataset_gathering/results/out.txt"
memory_limit=""
use_bpftrace=false
non_bpf_output_path="/home/garvalov/ml_prefetching_project_2/dataset_gathering/results/kernel/"
exec_pid=-1
# Function to display usage information
display_usage() {
    echo "[TraceRunner] Usage: $0 [-v] [-b] [-o output_file] [-c memory_limit] <parsec_bench_executable> [parsec executable's args...]"
    echo "Options:"
    echo "  -v             Enable verbose mode"
    echo "  -b             Use bpftrace (if custom kernel is not used; note that bpftrace misses some page faults under pressure as it relies on perf events)"
    echo "  -o output_file Specify the output file for bpftrace (default: $output_file)"
    echo "  -c memory_limit Specify memory limit for cgroup (e.g., 8MB, 2GB)"
    exit 1
}

# Function to convert human-readable size to bytes
human_readable_to_bytes() {
    local input="$1"
    local unit="${input: -2}"  # Get the last two characters (e.g., MB, GB)

    case "$unit" in
        "KB") echo "$(( ${input%KB} * 1024 ))";;
        "MB") echo "$(( ${input%MB} * 1024 * 1024 ))";;
        "GB") echo "$(( ${input%GB} * 1024 * 1024 * 1024 ))";;
        *) echo "$input";;  # Assume it's already in bytes if no unit specified
    esac
}


# Process command line options
while getopts ":vbo:c:" opt; do
    case $opt in
        v)
            verbose=true
            ;;
        b)
            use_bpftrace=true
            ;;
        o)
            output_file="$OPTARG"
            ;;
        c)
            memory_limit="$(human_readable_to_bytes "$OPTARG")"
            ;;
        \?)
            echo "[TraceRunner] Invalid option: -$OPTARG" >&2
            display_usage
            ;;
    esac
done

# Shift the options so that $1 is the parsec benchmark executable
shift "$((OPTIND-1))"

if [ "$verbose" = true ]; then
    echo "[TraceRunner] Starting shell script with PID $$"
fi

# Check if the required arguments are provided
if [ "$#" -lt 1 ]; then
    display_usage
fi


parsec_bench_executable=$1
parsec_bench_name=$(basename "$parsec_bench_executable")

# Function to create a directory for results
create_results_directory() {
    local output_dir="$1"

    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
        echo "[TraceRunner] Created results directory: $output_dir"
    else
        if [ -z "$(ls -A "$output_dir")" ]; then
            echo "[TraceRunner] Results directory already exists and is empty: $output_dir"
        else
            read -p "[TraceRunner] Results directory ($output_dir) is not empty. Do you want to overwrite? (y/n): " overwrite_choice
            case "$overwrite_choice" in
                y|Y)
                    echo "[TraceRunner] Overwriting results directory: $output_dir"
                    rm -rf "$output_dir"/*
                    ;;
                *)
                    echo "[TraceRunner] Aborted. Please choose an empty directory or specify a different one."
                    exit 1
                    ;;
            esac
        fi
    fi
}

if [ "$use_bpftrace" = true ]; then

    # Display command information if in verbose mode
    if [ "$verbose" = true ]; then
        echo "[TraceRunner] Running bpftrace in the background to trace $parsec_bench_name"
        echo "[TraceRunner] Command: bpftrace -o $output_file /home/garvalov/ml_prefetching_project_2/dataset_gathering/bpf_scripts/get_data_on_pfaults.bt \"$parsec_bench_name\" &"
    fi

    alias_name="bpftrace"
    path1="/home/garvalov/bpftrace/result/bin/bpftrace"
    path2="/home/garvalov/bp_appImage/bpftrace"

    if [[ "$(type -t bpftrace)" != "alias" ]]; then
        # If the alias doesn't exist, create it
        if [ -x "$path1" ]; then
            correct_name=$path1
        else
            correct_name=$path2
        fi
        echo "[TraceRunner] Info, using $correct_name to run bpftrace."
    else
        echo "[TraceRunner] Info - Alias $alias_name already exists"
    fi


    $correct_name -o $output_file /home/garvalov/ml_prefetching_project_2/dataset_gathering/bpf_scripts/get_data_on_pfaults.bt "$parsec_bench_name"  &
    bpftrace_pid=$!
else
    if [ -z "$output_file" ]; then
        output_file="$non_bpf_output_path"
    fi
    create_results_directory "$output_file"
fi


cleanup_and_exit() {
    if [ "$verbose" = true ]; then
        echo "[TraceRunner] Called cleanup_and_exit"
    fi

    if [ -n "$bpftrace_pid" ] && ps -p "$bpftrace_pid" > /dev/null; then
        if [ "$verbose" = true ]; then
            echo "[TraceRunner] Terminating bpftrace process (PID: $bpftrace_pid)"
        fi
        kill -TERM "$bpftrace_pid"
        wait "$bpftrace_pid"
    fi

    # Remove cgroup
    if [ -n "$memory_limit" ]; then
        cgdelete memory:trace_cgroup
    fi

    if [ "$use_bpftrace" = false ] && [ -n "$output_file" ] && [ -d "$output_file" ]; then 
        if [ "$exec_pid" = -1 ]; then 
            echo "Execution PID not set; unable to clean directory properly..."
        else
            for entry in "$output_file"/*; do
                entry_name=$(basename "$entry")
                if [ "$entry_name" != "$exec_pid" ]; then
                    if [ -d "$entry" ]; then
                        rm -rf "$entry"
                        if [ "$verbose" = true ]; then
                            echo "[TraceRunner] Deleted directory: $entry"
                        fi
                    elif [ -f "$entry" ]; then
                        rm -f "$entry"
                        if [ "$verbose" = true ]; then
                            echo "[TraceRunner] Deleted file: $entry"
                        fi
                    fi
                fi
            done
        fi
    fi

    exit
}

trap 'cleanup_and_exit' EXIT;

init_sleep_secs=1;
if [ "$verbose" = true ]; then
    echo "[TraceRunner] Sleeping $init_sleep_secs second(s) to let bpftrace correctly set up"
fi

sleep $init_sleep_secs;

if [ "$verbose" = true ]; then
    echo "[TraceRunner] Executing parsec benchmark: $parsec_bench_executable $@"
    echo "[TraceRunner] Started bpftrace with PID $bpftrace_pid. Now running the Parsec Benchmark."
fi
# Run parsec benchmark, see https://unix.stackexchange.com/questions/197792/joining-bash-arguments-into-single-string-with-spaces
# "'$*'" &
# Execute the parsec benchmark

shift # Remove the 1st argument (the exec path, since we already have it)

# Function to automatically choose the best unit based on the magnitude
auto_choose_unit() {
    local value="$1"

    if [ "$value" -lt $((1024)) ]; then
        echo "${value}B"
    elif [ "$value" -lt $((1024 * 1024)) ]; then
        echo "$((value / 1024))KB"
    elif [ "$value" -lt $((1024 * 1024 * 1024)) ]; then
        echo "$((value / 1024 / 1024))MB"
    else
        echo "$((value / 1024 / 1024 / 1024))GB"
    fi
}

# Create a cgroup and execute the parsec benchmark inside it if memory_limit is specified
if [ -n "$memory_limit" ]; then
    if [ "$verbose" = true ]; then
        human_readable_memory_limit="$(auto_choose_unit "$memory_limit")"
        echo "[TraceRunner] Using a $human_readable_memory_limit limited cgroup"
    fi
    cgcreate -g memory:trace_cgroup
    if [ "$verbose" = true ]; then
        echo "[TraceRunner] Created memory:trace_cgroup CGroup"
    fi    
    cgset -r memory.max=$memory_limit trace_cgroup
    if [ "$verbose" = true ]; then
        echo "[TraceRunner] Set memory.limit_in_bytes:  cgset -r memory.max=$memory_limit trace_cgroup"
        echo "[TraceRunner] Executing parsec benchmark in CGroup: cgexec -g memory:trace_cgroup $parsec_bench_executable $@"
    fi
    cgexec -g memory:trace_cgroup "$parsec_bench_executable" "$@" &
    exec_pid=$!
else
    if [ "$verbose" = true ]; then
        echo "[TraceRunner] Executing parsec benchmark: $parsec_bench_executable $@"
    fi
    # Execute the parsec benchmark without a cgroup if memory_limit is not specified
    "$parsec_bench_executable" "$@" &
    exec_pid=$!
fi
wait "$exec_pid"

end_sleep_secs=2

if [ "$verbose" = true ]; then
    echo "[TraceRunner] Parsec finished execution, sleeping $end_sleep_secs second(s) to let BPF finish execution"
fi

sleep $end_sleep_secs;

cleanup_and_exit
