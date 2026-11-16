import ctypes as c
import os
import sysconfig
from pathlib import Path
import numpy as np
SITE = Path(sysconfig.get_path('purelib'))
CUDA_LIB = os.environ.get('SEC_CUDA_LIB', str(SITE / 'nvidia/cuda_runtime/lib/libcudart.so.12'))
TRT_LIB = os.environ.get('SEC_TRT_LIB', str(SITE / 'tensorrt_libs/libnvinfer.so.11'))
class Plan(c.Structure):
    _fields_ = [(key, c.c_size_t) for key in ("start", "count", "width", "offset")]


class Native:
    def __init__(self, plan, nvtx_off=False, library=None):
        self.dll = c.CDLL(str(library))
        signatures = {
            "sec_error": ([], c.c_char_p),
            "sec_create": ([c.c_char_p] * 3, c.c_void_p),
            "sec_destroy": ([c.c_void_p], None),
            "sec_acquire": ([c.c_void_p, c.c_size_t, c.c_size_t], c.c_void_p),
            "sec_host": ([c.c_void_p, c.c_int], c.POINTER(c.c_int32)),
            "sec_upload": ([c.c_void_p, c.c_void_p, c.c_size_t], c.c_int),
            "sec_score": ([c.c_void_p, c.c_void_p, c.POINTER(Plan), c.c_size_t,
                           c.c_size_t, c.POINTER(c.c_float)], c.c_int),
            "sec_release": ([c.c_void_p, c.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes = args
            function.restype = result
        runtime = CUDA_LIB
        self.cuda = c.CDLL(str(runtime))
        self.context = self.dll.sec_create(os.fsencode(plan), os.fsencode(runtime), os.fsencode(TRT_LIB))
        self.check(self.context)
        if nvtx_off:
            self.dll.sec_no_annotations.argtypes = [c.c_void_p]
            self.dll.sec_no_annotations.restype = c.c_int
            self.check(self.dll.sec_no_annotations(self.context))

    def check(self, result):
        if not result:
            raise RuntimeError(self.dll.sec_error().decode())

    def close(self):
        if self.context:
            self.dll.sec_destroy(self.context)
            self.context = None

    def set_graph_widths(self, widths):
        self.dll.sec_graph_widths.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.c_size_t]
        self.dll.sec_graph_widths.restype = None
        values = (c.c_uint32 * len(widths))(*widths)
        self.dll.sec_graph_widths(self.context, values, len(values))

    def prepare(self, tokens, batch, width_multiple=8, token_budget=None):
        order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
        plans, elements = [], 0
        start = 0
        while start < len(order):
            count = min(batch, len(order) - start)
            width = (len(tokens[order[start + count - 1]]) + width_multiple - 1) // width_multiple * width_multiple
            if token_budget:
                while count > 8 and count * width > token_budget:
                    count = max(8, count - 8)
                    width = (len(tokens[order[start + count - 1]]) + width_multiple - 1) // width_multiple * width_multiple
            plans.append(Plan(start, count, width, elements))
            elements += count * width
            start += count
        buffer = self.dll.sec_acquire(self.context, elements, len(order))
        self.check(buffer)
        ids = np.ctypeslib.as_array(self.dll.sec_host(buffer, 0), shape=(elements,))
        masks = np.ctypeslib.as_array(self.dll.sec_host(buffer, 1), shape=(elements,))
        ids.fill(0)
        masks.fill(0)
        for p in plans:
            for row in range(p.count):
                value = tokens[order[p.start + row]]
                start = p.offset + row * p.width
                ids[start:start + len(value)] = value
                masks[start:start + len(value)] = 1
        self.check(self.dll.sec_upload(self.context, buffer, elements))
        return buffer, (Plan * len(plans))(*plans), order, elements

    def score(self, job):
        buffer, plans, order, _ = job
        result = np.empty((len(order), 4), dtype=np.float32)
        self.check(self.dll.sec_score(self.context, buffer, plans, len(plans), len(order),
                                    result.ctypes.data_as(c.POINTER(c.c_float))))
        output = np.empty_like(result)
        output[order] = result
        return output

    def release(self, job):
        self.dll.sec_release(self.context, job[0])

