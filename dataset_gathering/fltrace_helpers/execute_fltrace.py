import argparse
import subprocess
import shutil
from pathlib import Path
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="Execute mycommand and copy files with a specific prefix.")
    parser.add_argument('-o', '--output', help='Output directory',default=".")
    parser.add_argument('-p', '--prefix', help='Prefix of outputs',default="fltrace-data-")
    parser.add_argument("-f","--fltrace-path", help="Path to fltrace", default="~/fltrace/fltrace")
    parser.add_argument('total_max_memory', type=int, help='Total max memory used by subprocess')
    parser.add_argument('limiting_memory', type=int, help='Limiting memory for subprocess')
    parser.add_argument('parsec_exec', help='Parsec executable and its arguments', nargs='+')

    args = parser.parse_args()
    if args.output[-1] != '/':
        args.output+='/'
    print(args.parsec_exec)
    p = Path(args.fltrace_path)
    assert p.exists() and p.is_file()
    args.fltrace_path = p.absolute().as_posix()
    return args

def run_fltrace_and_bench(args):
    command = f'sudo {args.fltrace_path} record -M={args.total_max_memory} -L={args.limiting_memory} -- {" ".join(args.parsec_exec)}'
    # Run mycommand and capture stdout
    try:
        result = subprocess.run([command], shell=True, stdout=subprocess.PIPE, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
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
    p = Path(args.output)
    p.mkdir(exist_ok=True, parents=True)

    # Extract and copy files with correct prefix
    extract_and_copy_files(args,p, stdout)


if __name__ == "__main__":
    main()
