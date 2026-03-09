import os
import datetime
import logging
from logging import DEBUG

def get_log_file():
    LOG_DIR = os.path.join(os.path.expanduser("~"), '.FLIKA', 'log')
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    existing_files = os.listdir(LOG_DIR)
    existing_files = [f for f in existing_files if os.path.splitext(f)[1] == '.log']
    existing_idxs = [int(f.split('.')[0]) for f in existing_files]
    log_idx = 0
    while log_idx in existing_idxs:
        log_idx += 1
    log_idx -= 1
    LOG_FILE = os.path.join(LOG_DIR, '{0:0>3}.log'.format(log_idx))
    return LOG_FILE

def flush_all_handlers():
    """Flush all logging handlers to ensure log file is complete."""
    for handler in logging.root.handlers:
        handler.flush()
    flika_logger = logging.getLogger("flika")
    for handler in flika_logger.handlers:
        handler.flush()

def get_log_steps():
    LOG_FILE = get_log_file()
    lines = []
    for line in reversed(list(open(LOG_FILE))):
        text = line.rstrip()
        if 'DEBUG' in text and ('Started' in text or 'Completed' in text):
            lines.append(text)
        if "Started 'reading __init__.py'" in text:
            break
    lines = lines[::-1]
    steps = get_steps(lines, 0)
    return steps

class Step(object):
    def __init__(self, name, time, children):
        self.name = name
        self.time = time
        self.children = children
    def __repr__(self):
        t = "{:.3f} s".format(self.time.seconds + self.time.microseconds/1000000)
        return "{} ({})".format(self.name, t)
    def repr_w_children(self, prefix=''):
        repr = prefix + self.__repr__() + '\n'
        for child in self.children:
            repr += child.repr_w_children(prefix=prefix + '- ')
        return repr

def get_steps(lines, idx, parent_step=None):
    steps = []
    while idx < len(lines):
        line = lines[idx]
        step_name = line.split("'")[1]
        msg = line.split(' - DEBUG - ')[1]
        if msg.startswith('Started'):
            t_i = datetime.datetime.strptime(line.split(' - DEBUG')[0], "%Y-%m-%d %H:%M:%S,%f")
            result = get_steps(lines, idx+1, parent_step=step_name)
            if isinstance(result, tuple) and len(result) == 3:
                substeps, t_f, idx = result
                steps.append(Step(step_name, t_f - t_i, substeps))
                continue  # idx already points to next line
            else:
                # Missing Completed entry — use last known time
                substeps = result if isinstance(result, list) else []
                t_last = datetime.datetime.strptime(lines[-1].split(' - DEBUG')[0], "%Y-%m-%d %H:%M:%S,%f")
                steps.append(Step(step_name + ' [incomplete]', t_last - t_i, substeps))
                break
        elif msg.startswith('Completed'):
            if step_name != parent_step:
                print("Warning: Step name '{}' != parent_step '{}'".format(step_name, parent_step))
            t_f = datetime.datetime.strptime(line.split(' - DEBUG')[0], "%Y-%m-%d %H:%M:%S,%f")
            return steps, t_f, idx+1
        idx += 1
    return steps

if __name__ == '__main__':
    import sys
    real_stdout = sys.__stdout__  # preserve original stdout before flika wraps it
    from flika import *
    start_flika()
    assert logger.level == DEBUG
    flush_all_handlers()
    steps = get_log_steps()
    for step in steps:
        real_stdout.write(step.repr_w_children())
    real_stdout.flush()
