import numpy as np

class Tensor:
    def __init__(self, data, _children=()):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other))

        def _backward():
            # A (self) is shape (m, n), B (other) is shape (n, p)
            # out is shape (m, p)
            
            # dA = dC @ B.T
            self.grad += out.grad @ other.data.T
            
            # dB = A.T @ dC
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def backward(self):
        # Topological sort to ensure correct order of gradient resolution
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        
        # Base case: gradient of the output scalar w.r.t itself is 1
        self.grad = np.ones_like(self.data)
        
        # Apply the chain rule backwards through the graph
        for node in reversed(topo):
            node._backward()