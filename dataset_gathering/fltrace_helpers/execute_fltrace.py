import argparse
import subprocess
import shutil
from pathlib import Path
import re
import time

# /home/vigarov/ml_prefetching_project_2/data/raw/fltrace_out# strings=("canneal" "ferret" "facesim" "bodytrack" "dedup" "fluidanimate" "raytrace" "streamcluster"); values=(200 100 500 30 1000 500 200 50); for ((i=0; i<${#strings[@]}; i++)); do string="${strings[$i]}"; N="${values[$i]}"; for ((j=0; j<4; j++)); do M=$((N * (j + 1) / 4)); parsecmgmt -a run -i simlarge -p "$string" -s "python /home/vigarov/ml_prefetching_project_2/dataset_gathering/fltrace_helpers/execute_fltrace.py --output_dir /home/vigarov/ml_prefetching_project_2/data/raw/fltrace_out/\!BN\!/${N}_${M} --fltrace_path /home/vigarov/fltrace/fltrace $N $M "; done; done

def parse_arguments():
    parser = argparse.ArgumentParser(description="Execute mycommand and copy files with a specific prefix.")
    parser.add_argument('--output_dir', help='Output directory. Can contain !BN! to denote the current '
                                               '(paresec) benchmark name (will evaluate to empty string if not found, '
                                               'and !TST! for a timestamp.',default=".")
    parser.add_argument('--prefix', help='Prefix of outputs',default="fltrace-data-")
    parser.add_argument("--fltrace_path", help="Path to fltrace", default="~/fltrace/fltrace")
    parser.add_argument('total_max_memory', type=int, help='Total max memory used by subprocess')
    parser.add_argument('limiting_memory', type=int, help='Limiting memory for subprocess')
    parser.add_argument('parsec_exec', help='Parsec executable and its arguments', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.output_dir[-1] != '/':
        args.output_dir+='/'
    KNOWN_BENCHES = ["canneal","ferret","facesim","bodytrack","dedup","fluidanimate","raytrace","streamcluster"]
    args.benchmark_name = ''
    for b in KNOWN_BENCHES:
        if sum([int(b in arg_or_param) for arg_or_param in args.parsec_exec])>0:
            args.benchmark_name = b
            break
    args.output_dir = args.output_dir.replace('!BN!',args.benchmark_name).replace('!TST!',time.strftime("%Y%m%d-%H%M%S"))
    print(args.parsec_exec)
    p = Path(args.fltrace_path)
    assert p.exists() and p.is_file()
    args.fltrace_path = p.absolute().as_posix()
    return args

def run_fltrace_and_bench(args):
    command = f'sudo {args.fltrace_path} record -M={args.total_max_memory} -L={args.limiting_memory} -- {" ".join(args.parsec_exec)}'
    # Run mycommand and capture stdout
    try:
        result = subprocess.run([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT , text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[STDOUT]{e.stdout}\n[STDERR]{e.stderr if e.stderr else ''}\nError executing command: {e}")
        exit(1)


def extract_and_copy_files(args,output_dir, output):
    # Extract path from the last line of stdout
    lines = output.splitlines()
    assert lines and len(lines) > 1
    last_line = lines[-1]
    paths = re.findall(r'(\/.+\/.*)', last_line)
    assert len(paths) == 1
    fltrace_output_dir= Path(paths[0])
    assert fltrace_output_dir.exists()
    for file in fltrace_output_dir.glob('*'):
        if file.name.startswith(args.prefix):
            shutil.copy2(file.absolute().as_posix(), output_dir.absolute().as_posix())
            print(f"Copied {file.absolute().as_posix()} to {output_dir.absolute().as_posix()}. Deleting.")
            file.unlink()


def main():
    args = parse_arguments()

    stdout = run_fltrace_and_bench(args)
    print(stdout)
    # Create output directory if it doesn't exist
    p = Path(args.output_dir)
    p.mkdir(exist_ok=True, parents=True)

    # Extract and copy files with correct prefix
    extract_and_copy_files(args,p, stdout)


if __name__ == "__main__":
    main()
