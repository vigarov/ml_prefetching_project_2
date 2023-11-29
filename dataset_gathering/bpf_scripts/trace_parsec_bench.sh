#!/bin/bash
# To be used a submit command for parsecmgmt, a.k.a. run 
# $ sudo parsecmgmt -a run -p <parsec benchmark> -i <sim size> -s "`pwd`/trace_parsec_bench.sh <-v>"

# Initialize verbose flag
verbose=false
output_file="/home/garvalov/ml_prefetching_project_2/dataset_gathering/results/out.txt"

# Function to display usage information
display_usage() {
    echo "[TraceRunner] Usage: $0 [-v] [-o output_file] <parsec_bench_executable> [parsec executable's args...]"
    echo "Options:"
    echo "  -v             Enable verbose mode"
    echo "  -o output_file Specify the output file for bpftrace (default: $output_file)"
    exit 1
}

# Process command line options
while getopts ":vo:" opt; do
    case $opt in
        v)
            verbose=true
            ;;
        o)
            output_file="$OPTARG"
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

# Display command information if in verbose mode
if [ "$verbose" = true ]; then
    echo "[TraceRunner] Running bpftrace in the background to trace $parsec_bench_name"
    echo "[TraceRunner] Command: bpftrace -o $output_file /home/garvalov/ml_prefetching_project_2/dataset_gathering/bpf_scripts/get_data_on_pfaults.bt \"$parsec_bench_name\" &"
fi


/home/garvalov/bp_appImage/bpftrace -o $output_file /home/garvalov/ml_prefetching_project_2/dataset_gathering/bpf_scripts/get_data_on_pfaults.bt "$parsec_bench_name"  &
bpftrace_pid=$!



cleanup_and_exit() {
    if [ -n "$bpftrace_pid" ] && ps -p "$bpftrace_pid" > /dev/null; then
        if [ "$verbose" = true ]; then
            echo "[TraceRunner] Terminating bpftrace process (PID: $bpftrace_pid)"
        fi
        kill -TERM "$bpftrace_pid"
        wait "$bpftrace_pid"
    fi
    exit
}

trap 'cleanup_and_exit' EXIT;

sleep_secs=3;
if [ "$verbose" = true ]; then
    echo "[TraceRunner] Sleeping $sleep_secs second(s) to let bpftrace correctly set up"
fi

sleep $sleep_secs;

if [ "$verbose" = true ]; then
    echo "[TraceRunner] Executing parsec benchmark: $parsec_bench_executable $@"
    echo "[TraceRunner] Started bpftrace with PID $bpftrace_pid. Now running the Parsec Benchmark."
fi
# Run parsec benchmark, see https://unix.stackexchange.com/questions/197792/joining-bash-arguments-into-single-string-with-spaces
# "'$*'" &
# Execute the parsec benchmark

shift # Remove the 1st argument (the exec path, since we already have it)
if [ "$verbose" = true ]; then
    echo "[TraceRunner] Executing parsec benchmark: $parsec_bench_executable $@"
fi

sleep 0.5;# for debug
"$parsec_bench_executable" "$@"

end_sleep_secs=5

if [ "$verbose" = true ]; then
    echo "[TraceRunner] Parsec finished execution, sleeping $end_sleep_secs second(s) to let BPF finish execution"
fi

sleep $end_sleep_secs;

cleanup_and_exit
