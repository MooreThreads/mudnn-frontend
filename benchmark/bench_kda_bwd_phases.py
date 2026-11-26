import statistics
import torch
from mudnn.linear_attention.mutlass.kda_backward import chunk_kda_bwd
from mudnn.linear_attention.mutlass import kda_ops, kda_backward_scan_ops, kda_backward_grad_ops

torch.musa.set_device(0)
D, H, T = 128, 64, 2048
dev = torch.device('musa')
dtype = torch.bfloat16
gen = torch.Generator(device=dev); gen.manual_seed(123)
def rand(shape, dtype=dtype):
    return torch.randn(shape, device=dev, dtype=dtype, generator=gen)
q = rand((1,T,H,D)); k = rand(q.shape); v = rand(q.shape); g = rand(q.shape)
beta = rand((1,T,H)); dO = rand(v.shape)
A_log = torch.randn((H,), device=dev, dtype=torch.float32, generator=gen)
dt_bias = torch.randn((H,D), device=dev, dtype=torch.float32, generator=gen)
state = torch.zeros((1,H,D,D), device=dev, dtype=dtype)
d_final = torch.zeros((1,H,D,D), device=dev, dtype=torch.float32)
kw = dict(scale=D**-0.5, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
          use_qk_l2norm_in_kernel=True, initial_state=state, d_final_state=d_final)

times = {key: [] for key in ('k1','k2','k3','k4')}
def wrap(mod, attr, key):
    original = getattr(mod, attr)
    def get(config):
        module = original(config)
        original_fn = module.get_function
        class Proxy:
            def get_function(self, name):
                fn = original_fn(name)
                def timed(*args):
                    start = torch.musa.Event(enable_timing=True); end = torch.musa.Event(enable_timing=True)
                    start.record(); out = fn(*args); end.record(); end.synchronize()
                    times[key].append(start.elapsed_time(end)); return out
                return timed
        return Proxy()
    setattr(mod, attr, get)
wrap(kda_ops, 'get_kda_prepare_ops_module', 'k1')
wrap(kda_ops, 'get_kda_recurrence_ops_module', 'k2')
wrap(kda_backward_scan_ops, 'get_kda_backward_scan_ops_module', 'k3')
wrap(kda_backward_grad_ops, 'get_kda_backward_grad_ops_module', 'k4')

def run():
    return chunk_kda_bwd(dO, q, k, v, g, beta, **kw)
run(); torch.musa.synchronize()
for _ in range(3): run()
torch.musa.synchronize()
for _ in range(20): run()
for key, values in times.items():
    samples = values[-20:]
    print(key, 'samples_ms=', [round(x,4) for x in samples],
          'p50_ms=', round(statistics.median(samples),4),
          'p90_ms=', round(sorted(samples)[int(.9*len(samples))-1],4))

# Logical byte count for the K4 ABI.  All K4 input/output/workspace views are
# contiguous BF16 except dg/dbeta, A_log/dt_bias and dstate0-side metadata.
chunks = (T + 15) // 16
records = chunks * H
bf16 = 2; fp32 = 4
activation = T * H * D
workspace_elements = records * (2*16*D + 16*D + 16*16 + 16*16 + D)
bytes_read = (5*activation + T*H) * bf16  # q,k,v,g,dO,beta
bytes_read += (2*records*D*D + workspace_elements) * bf16
bytes_read += (H + H*D) * fp32  # A_log, dt_bias
bytes_written = (3*activation) * bf16 + activation * fp32 + (T*H) * fp32
logical = bytes_read + bytes_written
print('shape', {'B':1,'T':T,'H':H,'D':D,'chunks':chunks,'records':records})
print('bytes_read', bytes_read, 'bytes_written', bytes_written, 'logical_total', logical)
print('k4_logical_GB', logical/1e9, 'k4_logical_GiB', logical/2**30)
for x in (statistics.median(times['k4'][-20:]),):
    print('k4_effective_GBps_decimal', logical/(x*1e-3)/1e9,
          'k4_effective_GiBps', logical/(x*1e-3)/2**30)
