"""Op interface used by the timing harness.

An Op owns `n_copies` independent sets of operands so consecutive calls can rotate through them (L2 rotation).
Contract:
  footprint_bytes(): device bytes allocated per copy
  bytes(): per-call traffic {weights, metadata, act_in, out}
  build(n_copies): allocate operands
  call(i): launch the op on copy i (must be graph-capturable unless graph_ok is False)
  check(): (ok, detail) correctness of copy 0 against a reference; run before any timing
  free(): drop all device memory
"""


class Op:
    op = "?"
    impl = "?"
    precision = "?"
    graph_ok = True

    def __init__(self, M=None, N=None, K=None, model=None, layer_name=None):
        self.M, self.N, self.K = M, N, K
        self.model, self.layer_name = model, layer_name

    def config(self):
        """Fields that identify this op in the cell id."""
        return {"op": self.op, "impl": self.impl, "precision": self.precision,
                "shape": {"M": self.M, "N": self.N, "K": self.K}, "model": self.model, "layer_name": self.layer_name}

    def footprint_bytes(self):
        raise NotImplementedError

    def bytes(self):
        raise NotImplementedError

    def build(self, n_copies):
        raise NotImplementedError

    def call(self, i):
        raise NotImplementedError

    def check(self):
        raise NotImplementedError

    def free(self):
        import torch
        for k in list(vars(self)):
            if k.startswith("_t"):
                setattr(self, k, None)
        torch.cuda.empty_cache()
