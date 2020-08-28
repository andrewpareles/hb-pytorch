import json
import re
import csv
import sys

ROUTE_JSON = 'sinkhorn_wmd.json'
HB_STATS = 'run_{}/manycore_stats.log'
HB_LOG = 'run_{}/log.txt'
CPU_LOG = 'cpu_run/log.txt'

HB_FREQ = 10 ** 9  # 1 GHz.
HB_MACHINE_FRAC = 16  # Simulating 1/16th of the machine.
HB_DATA_FRAC = 16  # Used this fraction of the CPU's data.


def cycles_from_stats(stats):
    """Given the text contents of a `manycore_stats.log` file, extract the
    total number of cycles for the kernel execution.
    """
    lines = stats.splitlines()
    overview = lines[3]
    cycles = int(overview.split()[6])
    return cycles


def kernel_name(sig):
    """Given a C++ function signature, extract the base name, for human
    legibility.
    """
    return re.search(r'::(\w+)\(', sig).group(1)


def times_from_log(log):
    """Given a CPU execution log, look for the output from
    `hammerblade.profiler.stats` that breaks down the amount of wall-clock
    time spent in each kernel. Generate (kernel, time) pairs.
    """
    in_report = False
    for line in log.splitlines():
        if 'Kernel execution time' in line:
            in_report = True
            continue

        if in_report:
            kernel, tm, pct = line.strip().split()
            if kernel.startswith('aten::'):
                _, kernel = kernel.split('::')
            if 'time_in_roi' in kernel:
                break
            yield kernel, float(tm)


def parse_tree(log):
    """Given an execution log from any run, look for the "tree" output
    from `hammerblade.profiler.exec_time.raw_stack` and parse it into
    (level, function, time) tuples.
    """
    for line in log.splitlines():
        if line.strip().startswith('|- Node'):
            indent, rest = line.split('|-', 1)
            level = len(indent) // 2
            match = re.search(r'Node\((.*) : (\d+\.\d+)\)', line)
            sig, tm = match.groups()

            micros = int(tm.split('.')[0])  # Data reported in microseconds.

            yield level, sig, micros / 10**6


def total_times_from_tree(log):
    """Given an execution log from any run, use the `raw_stack` tree to
    get the total execution time for top-level kernel invocations.
    Generate (kernel, time) pairs.
    """
    for level, sig, secs in parse_tree(log):
        if level == 1:
            kernel = kernel_name(sig) if sig.startswith('at::') else sig
            yield kernel, secs


def trimmed_times_from_tree(log):
    last_level = None
    last_secs = 0.0
    cur_kernel = None
    cur_total = None

    for level, sig, secs in parse_tree(log):
        # If we are at an equal or lower level than the immediate
        # predecessor, then that predecessor was a leaf.
        if last_level is not None and level <= last_level:
            cur_total += last_secs
        last_level = level
        last_secs = secs

        # Check for a new top-level kernel.
        if level == 1:
            if cur_kernel is not None:
                yield cur_kernel, cur_total
            cur_kernel = kernel_name(sig) if sig.startswith('at::') else sig
            cur_total = 0.0

    # Last item was a leaf.
    cur_total += last_secs

    # Emit final kernel.
    yield cur_kernel, cur_total


def hb_cycles_to_time(cycles):
    """Compute seconds from simulation cycles, assuming weak scaling and the
    hardware frequence.
    """
    # The scale factor is the amount slower we would run if we used *all* the
    # data. If we were simulating the *full* machine, this would just be
    # `HB_DATA_FRAC`, i.e., how much *less* data HB had to process than the
    # CPU. However, we simulate only `HB_MACHINE_FRAC` of the machine and
    # assume that expanding the data by *that* amount would lead to the same
    # execution time.
    scale_factor = HB_DATA_FRAC / HB_MACHINE_FRAC

    # The amount of "real" time we simulated for, according to the machine
    # frequency.
    sim_secs = cycles / HB_FREQ

    return sim_secs * scale_factor


def collect():
    with open(ROUTE_JSON) as f:
        kernels = json.load(f)

    # Load results from every HB run (one per kernel).
    hb_cycles = {}
    for i, kernel in enumerate(kernels):
        # Load HB cycles from statistics dump.
        stats_fn = HB_STATS.format(i)
        with open(stats_fn) as f:
            stats_txt = f.read()
        hb_cycles[kernel_name(kernel['signature'])] = \
            cycles_from_stats(stats_txt)

    # Load CPU time breakdown.
    with open(CPU_LOG) as f:
        log_txt = f.read()
    cpu_times = dict(total_times_from_tree(log_txt))

    # Dump a CSV.
    writer = csv.DictWriter(
        sys.stdout,
        ['kernel', 'cpu_time', 'hb_cycles', 'hb_time']
    )
    writer.writeheader()
    for kernel in sorted(set(hb_cycles).union(cpu_times)):
        writer.writerow({
            'kernel': kernel,
            'cpu_time': cpu_times.get(kernel),
            'hb_cycles': hb_cycles.get(kernel),
            'hb_time': (hb_cycles_to_time(hb_cycles[kernel])
                        if kernel in hb_cycles else ''),
        })


if __name__ == '__main__':
    collect()
