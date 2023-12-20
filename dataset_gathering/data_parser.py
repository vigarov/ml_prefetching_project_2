import json5
import pandas as pd
from pandas import DataFrame, read_csv
from pathlib import Path
import ast
import re

import swifter
import subprocess
from dataclasses import dataclass, field

from bisect import bisect_left

PAGE_SIZE = 4096

"""
Parses an objectdump-s output
"""

def get_page_address(addr:   int) -> int:
    return addr & ~(PAGE_SIZE - 1)


def load_bpfout_json(filename):
    print("Loading data")
    # FIXME: I need bleach for my eyes Alex
    with open(filename, 'r') as file:
        data = file.read()
        data = data.replace("\n", " ")
    with open(filename, 'w') as file:
        file.write(data)
    with open(filename, 'r') as f:
        data = json5.load(f)['a']
    print("Loaded data")
    return data


# change depending on which machine this is ran on
RAW_DATA_PATH = "/home/vigarov/ml_prefetching_project_2/data/raw/correct_out_3.txt"
BPF_PREPROCESS_SAVE_PATH = "/home/vigarov/ml_prefetching_project_2/data/prepro/"

DEFAULT_BENCH_NAME = "canneal"

PREPRO_VERSION = 1.1
PRO_VERSION = 1.5


def preprocess_bpftrace(path=RAW_DATA_PATH,
                        save: str | None = BPF_PREPROCESS_SAVE_PATH + DEFAULT_BENCH_NAME + "_v" + str(PREPRO_VERSION) + ".csv"):
    assert "raw" in Path(path).absolute().as_posix()
    df = DataFrame.from_dict(load_bpfout_json(RAW_DATA_PATH))
    if save is not None:
        df.to_csv(save, index=True)
    else:
        return df


def process_bpftrace(preprocessed_file :str, source_window:int, pred_window:int, page_mask:bool, save: str | None = None):
    p = Path(preprocessed_file)
    assert p.exists() and p.is_file() and p.suffix == ".csv"
    if "prepro" in p.absolute().as_posix():
        df = read_csv(preprocessed_file)
    else:
        df = preprocess_bpftrace(preprocessed_file, save=None)

    df = build_history(df, page_mask, pred_window, source_window)

    df["ip"] = df["ip"].swifter.apply(lambda ip_int: hex(ip_int))
    df["ustack"] = df["ustack"].swifter.apply(lambda ustack_str: ustack_str.replace('"', '').strip())
    df["regs"] = df["regs"].swifter.apply(lambda regs_array: ' '.join(["0x" + hex(reg)[2:].zfill(2) for reg in ast.literal_eval(
        regs_array.replace('"',
                           ''))]))  # have regs in hex, makes more sense and reduces input size/dimension post tokenization
    df["flags"] = df["flags"].swifter.apply(lambda flag: format(flag,
                                                        "016b"))  # flags are bitmaps,  might be easier to interpret if we spell them out as such

    # Drop first column = index added by preprocessor
    df = df.drop(columns=df.columns[[0]])

    if save is not None:
        df.to_csv(save,index=False)
    else:
        return df


def build_history(df, page_mask, pred_window_size, source_window_size,
                  address_column_name: str = "address", address_base:int = 10, output_column_name: str = "y"):
    y = []
    if page_mask:
        df[address_column_name] = df[address_column_name].swifter.apply(lambda address: get_page_address(int(address,address_base)))
    # Build history
    running_past_window = [hex(a) for a in df[address_column_name][:source_window_size]]
    running_future_window = [hex(a) for a in
                             df[address_column_name][source_window_size:source_window_size + pred_window_size]]
    for i in range(source_window_size, len(df) - pred_window_size):
        fault_address = hex(df[address_column_name][i])
        df[address_column_name][i] = " ".join(running_past_window.copy())
        n_th_next_fault = hex(df[address_column_name][i + pred_window_size])
        y.append(" ".join(running_future_window.copy()))
        running_past_window.pop(0)
        running_past_window.append(fault_address)
        running_future_window.pop(0)
        running_future_window.append(n_th_next_fault)
    # Drop unused/useless entries = first "source window" ones which don't have enough history,
    # and last "pred_window" ones, which have no predictions
    to_drop = list(range(source_window_size)) + list(range(len(df) - pred_window_size, len(df)))
    df = df.drop(to_drop)
    assert len(df) == len(y)  # Sanity check
    df[output_column_name] = y  # add output to our data frame == have one file (input+output) per trace

    # Rename to reflect our new data
    df = df.rename(columns={address_column_name: "prev_faults"})

    return df


############################################################################################################
# Below mainly adapted from fltrace's `parse.py` script
# https://github.com/anilkyelam/fltrace/blob/master/scripts/parse.py

def binary_search(a, x, key, lo=0, hi=None):
    if hi is None: hi = len(a)
    pos = bisect_left(a, x, lo, hi, key=key)  # find insertion position
    return pos if pos != hi and key(a[pos]) == x else -1  # don't walk off the end


# parse /proc/<pid>/maps
MAPS_LINE_RE = re.compile(r"""
    (?P<addr_start>[0-9a-f]+)-(?P<addr_end>[0-9a-f]+)\s+  # Address
    (?P<perms>\S+)\s+                                     # Permissions
    (?P<offset>[0-9a-f]+)\s+                              # Map offset
    (?P<dev>\S+)\s+                                       # Device node
    (?P<inode>\d+)\s+                                     # Inode
    (?P<path>.*)\s+                                   # path
""", re.VERBOSE)


@dataclass
class Record:
    """A line in /proc/<pid>/maps"""
    addr_start: int
    addr_end: int
    perms: str
    offset: int
    path: str

    @staticmethod
    def parse(filename):
        records = []
        with open(filename) as fd:
            for line in fd:
                m = MAPS_LINE_RE.match(line)
                if not m:
                    print("Skipping: %s" % line)
                    continue
                addr_start, addr_end, perms, offset, _, _, path = m.groups()
                r = Record(addr_start=int(addr_start, 16), addr_end=int(addr_end, 16), offset=int(offset, 16),
                           perms=perms, path=path)
                records.append(r)
        return records

    @staticmethod
    def find_record(records, addr):
        for r in records:
            if r.addr_start <= addr < r.addr_end:
                return r
        return None


@dataclass
class ObjDump_ASM_Instr:
    addr: int
    hex_repr: str  # space separate
    instr: str
    params: str

    def get_full_text_repr(self):
        return self.instr + ' ' + self.params

    def __str__(self):
        return self.get_full_text_repr()


@dataclass
class ObjDump_Section:
    name: str
    start_addr: int  # included
    end_addr: int = -1  # excluded
    asm_instructions: list[ObjDump_ASM_Instr] = field(default_factory=list)


@dataclass
class ObjDump:
    sections: list[ObjDump_Section] = field(default_factory=list)  # sorted by section start address


class LibOrExe:
    """A library or executable mapped into process memory"""
    records: list
    ips: list
    path: str
    base_addr: int
    codemap: dict
    objdump: ObjDump | None

    def __init__(self, records):
        """For libs collected from /proc/<pid>/maps"""
        self.records = records
        self.path = records[0].path
        self.base_addr = min([r.addr_start for r in records])
        self.ips = []
        self.codemap = {}
        self.objdump = None


def get_objdump_object(objdump_dir:str|None,binary_file):
    if objdump_dir is None:
        objdump_out = subprocess.run(['objdump', '-d', binary_file], stdout=subprocess.PIPE).stdout.decode("utf-8")
    else:
        od = Path(objdump_dir)
        assert od.exists() and od.is_dir()
        saught_file = Path(binary_file)
        od_file = None
        for file in od.glob('*'):
            if file.name == saught_file.name:
                od_file = file
                break
        assert od_file is not None
        with open(od_file.absolute().as_posix(),'r',encoding="utf-8") as f:
            objdump_out = f.read()
    objdump = ObjDump()
    current_section = None
    new_big_section = False
    for line in objdump_out.split("\n")[3:]:
        if line is None:
            continue
        elif line.startswith("Disassembly of section"):
            new_big_section = True
        elif line.strip() != '':
            if line[0].isnumeric():
                # We're starting a new section
                assert ':' in line
                splitted = line.split()
                assert len(splitted) == 2
                raw_start, raw_name = splitted[0], splitted[1]
                assert raw_start.isalnum() and '<' in raw_name and '>' in raw_name
                curr_add = int(raw_start, 16)
                if current_section is not None:
                    # End previous section
                    assert new_big_section or current_section.end_addr == curr_add
                    objdump.sections.append(current_section)
                new_big_section = False
                # Start the new one
                current_section = ObjDump_Section(start_addr=curr_add,
                                                  name=raw_name)  # .replace('<','').replace('>','')) ?
            else:
                elements = line.strip().split('\t')
                assert ':' == elements[0][-1]
                curr_add = elements[0][:-1]
                assert curr_add.isalnum()
                curr_add = int(curr_add, 16)
                hex_repr = elements[1].strip()
                if len(elements) == 2:
                    # nop, quick path
                    instr = "nop"
                    params = ""
                else:
                    assert len(elements) == 3
                    textual_repr = elements[2] if len(elements) == 3 else "nop"
                    tr_splitted = textual_repr.split()
                    # for everything but `bnd <instr>`, instruction is one word, rest is params
                    # `bnd` simply specifies CPU to check bounds, can ignore it, as it doesn't give semantical info abt input
                    if "bnd" in textual_repr:
                        assert tr_splitted[0] == "bnd"
                        tr_splitted = tr_splitted[1:]
                    instr = tr_splitted[0]
                    # restore params with spaces (e.g.: for `call`)
                    params = ' '.join(tr_splitted[1:])
                curr_line_asm = ObjDump_ASM_Instr(curr_add, hex_repr, instr, params)
                current_section.asm_instructions.append(curr_line_asm)
        elif current_section is not None:
            # End Section
            last_asm_instr = current_section.asm_instructions[-1]
            current_section.end_addr = last_asm_instr.addr + len(last_asm_instr.hex_repr.split())
    if objdump.sections[-1] != current_section:
        objdump.sections.append(current_section)
    objdump.sections.sort(
        key=lambda section: section.start_addr)  # Should essentially not change the order, but juuuust in case
    return objdump


def get_surrounding_assembly(loe: LibOrExe, ip: int, window:tuple[int,int]) -> (
        list[ObjDump_ASM_Instr], str):
    correct_rec = Record.find_record(loe.records, ip)
    assert correct_rec
    ip = correct_rec.offset + (ip - correct_rec.addr_start)
    # returns element after which we can insert such that we remain sorted
    objdump = loe.objdump
    address_section_idx = bisect_left(objdump.sections, ip, lo=0, hi=len(objdump.sections),
                                      key=lambda section: section.start_addr)
    if address_section_idx == len(objdump.sections):
        address_section_idx-=1
        assert objdump.sections[address_section_idx].start_addr <= ip < objdump.sections[address_section_idx].end_addr
    elif objdump.sections[address_section_idx].start_addr != ip:
        assert address_section_idx != 0 and objdump.sections[address_section_idx].start_addr >= ip
        address_section_idx -= 1
    ip_sect = objdump.sections[address_section_idx]
    assert ip_sect.start_addr <= ip < ip_sect.end_addr
    asms_in_sect = ip_sect.asm_instructions
    ip_idx_in_list = binary_search(asms_in_sect, ip, lambda asm_inst: asm_inst.addr)
    assert ip_idx_in_list != -1
    past_len,future_len = window
    min_past, max_future = max(0, ip_idx_in_list - past_len), min(len(asms_in_sect), ip_idx_in_list + future_len)
    return asms_in_sect[min_past:max_future], ip_sect.name


def preprocess_fltrace(path_faults, path_procmap,objdump_dir,code_window:tuple[int,int], save: str | None = None):
    df = read_csv(path_faults)  # tstamp,ip,addr,pages,flags,tid,trace
    # We don't need tid or tstamp
    # Also, "ip" is actually never set by fltrace
    # However, it simply corresponds to the top most record of the stacktrace
    # --> we can simply remove the "ip" column
    df = df.drop(columns=["tstamp", "tid", "ip", "pages"])
    # Since we only have anonymous page faults, flags only informs us of whether the access was a read, write, or writeprotect
    # we don't really care about the latter --> parse only if read write
    df["flags"] = df["flags"].swifter.apply(lambda flag: int(flag) & 0x1)
    df = df.rename(columns={"flags": "rW", "trace": "ips","addr":"address"})
    # get all unique ips
    iplists = df['ips'].str.split("|")
    ips = set().union(*[set(i) for i in iplists])
    ips.discard("")

    libmap = {}
    libs = {}
    records = Record.parse(path_procmap)
    for ip in ips:
        lib = Record.find_record(records, int(ip, 16))
        assert lib, "can't find lib for ip: {}".format(ip)
        assert lib.path, "no lib file path for ip: {}".format(ip)
        # Ignore fltrace.so, see comment after the for-loop for reasoning
        if "fltrace.so" in lib.path:
            continue
        if lib.path not in libs:
            librecs = [r for r in records if r.path == lib.path]
            libs[lib.path] = LibOrExe(librecs)
        libs[lib.path].ips.append(ip)
        libmap[ip] = lib.path

    # Remove all the ips linked to fltrace.so (addresses not representative of our application, but of allocations done
    # in userfaultd handler)
    # Note, since we already filter fltrace.so in the for loop, we can just the set difference
    ips = ips.intersection(set().union(*[set(lib.ips) for lib in libs.values()]))

    # Remove the now incorrect ips from the traces
    df["ips"] = df["ips"].swifter.apply(lambda ips_str: '|'.join([ip for ip in ips_str.split('|') if ip in ips]))

    for path, lib in libs.items():
        assert "fltrace.so" not in path
        # If there is an executable record (= memory region) for that lib/exec
        if sum([int('x' in record.perms) for record in lib.records]) > 0:
            lib.objdump = get_objdump_object(objdump_dir,lib.path)
    def instructions_lookup(ips):
        iplist = ips.split("|")
        if iplist[-1] == '':
            del iplist[-1]
        instrs = ';'.join(
            [' '.join(
                [asm_instr.get_full_text_repr() for asm_instr in
                 ip_to_windows_cache.setdefault(ip,
                                                get_surrounding_assembly(libs[libmap[ip]],
                                                                         int(ip, 16),
                                                                         window=code_window))[0]]
            ) for ip in iplist])
        return instrs


    ip_to_windows_cache = {i:' '.join([str(ins) for ins in get_surrounding_assembly(libs[libmap[i]],int(i, 16),window=code_window)[0]]) for i in ips}
    df['surr_insts'] = df['ips'].swifter.apply(lambda ipstr: ';'.join([ip_to_windows_cache[i] for i in ipstr.split('|')])) #instructions_lookup)

    # Final columns: address,rW,trace,surr_insts
    if save:
        df.to_csv()
    else:
        return df


def process_fltrace(preprocessed_file_or_dir :str|list, objdump_dir: str, source_window:int, pred_window:int, page_mask:bool, code_window:tuple[int,int] = (1, 2), multiple:bool = False ,save: str | None = None):
    # page_mask is technically useless for us as address off fltrace already come as pages, but still included just to make sure
    if not multiple:
        df = get_one_df(code_window, objdump_dir, preprocessed_file_or_dir)
        df = build_history(df, page_mask, pred_window, source_window, address_base=16)
    else:
        assert type(preprocessed_file_or_dir) == list
        all_dfs = []
        for df_file in preprocessed_file_or_dir:
            all_dfs.append(build_history(get_one_df(code_window,objdump_dir,df_file), page_mask, pred_window, source_window, address_base=16))
        df = pd.concat(all_dfs,ignore_index=True)

    df["ips"] = df["ips"].swifter.apply(lambda ip_str: ip_str.replace('|',' '))

    if save is not None:
        df.to_csv(save,index=False)
    else:
        return df


def get_one_df(code_window, objdump_dir, preprocessed_file_or_dir):
    assert type(preprocessed_file_or_dir) == str
    p = Path(preprocessed_file_or_dir)
    assert p.exists()
    if p.is_file():
        assert p.suffix == ".csv" and "prepro" in p.absolute().as_posix()
        df = read_csv(preprocessed_file_or_dir)
    else:
        assert p.is_dir()
        fault_file, procmap_file = "", ""
        for file in p.glob('*'):
            if "data-faults" in file.name:
                fault_file = file.absolute().as_posix()
            elif "procmaps" in file.name:
                procmap_file = file.absolute().as_posix()
        assert fault_file != '' and procmap_file != ''
        df = preprocess_fltrace(fault_file, procmap_file, objdump_dir, code_window, save=None)
    return df


BPFTRACE = False

if __name__ == "__main__":
    if BPFTRACE:
        p = "/home/vigarov/ml_prefetching_project_2/data/prepro/canneal_v1.1.csv"
        process_bpftrace(p, 10, 10, True,
                         save="/home/vigarov/ml_prefetching_project_2/data/processed/processed_" + re.sub(r"v\d+\.\d+", f"v{str(PRO_VERSION)}",  Path(p).name))
    else:
        p = "/home/vigarov/ml_prefetching_project_2/data/raw/fltrace_out/canneal/300_60"
        objdump_dir = "/home/vigarov/ml_prefetching_project_2/data/objdumps"
        process_fltrace(p,objdump_dir,10,10,True,
                        code_window=(-1, 2),
                        save="/home/vigarov/ml_prefetching_project_2/data/processed/processed_" + re.sub(r"v\d+\.\d+", f"v{str(PRO_VERSION)}",Path(p).name))
