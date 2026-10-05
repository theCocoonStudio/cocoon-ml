class Autograd:
    def __init__(self, value, _constituents=(), _operation=""):
        # Autograd is itself a number (scalar for now) being operated on
        self.value = float(value)
        # default, to be assigned by operation
        self._derivative = 0.0
        # closure to remember how the number was made, assigned when it is made
        self._backward = lambda: None
        # unique nodes comprising this number
        self._prev = set(_constituents)
        # operational memory
        self._operation = _operation

    def __repr__(self):
        return f"Autograd(value={self.value}, _derivative={self._derivative})"

    # all methods are kept binary, keeping them consistent at the expense of n-ary operations like addition becoming syntactically verbose to call
    def add(self, other):
        other = other if isinstance(other, Autograd) else Autograd(other)
        # output is always a new node with self and other as constituents
        out = Autograd(self.value + other.value, (self, other), "add")

        # derivatives are assigned on the way back via binding closure after out._derivative has been assigned
        def _backward():
            # Accumulate gradients (+=) in case a node has multiple consumers
            self._derivative += 1.0 * out._derivative
            other._derivative += 1.0 * out._derivative

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Autograd) else Autograd(other)
        out = Autograd(self.value * other.value, (self, other), "*")

        def _backward():
            # The derivative of (self * other) with respect to self is other.value
            self._derivative += other.value * out._derivative
            other._derivative += self.value * out._derivative

        out._backward = _backward
        return out

    def relu(self, zero_derivative=0.0):
        out_value = self.value if self.value > 0 else 0.0
        out = Autograd(out_value, (self,), "relu")

        def _backward():
            if self.value > 0:
                self._derivative += 1.0 * out._derivative
            elif self.value < 0:
                self._derivative += 0.0 * out._derivative
            else:
                # User-defined behavior strictly at the x=0 threshold
                self._derivative += float(zero_derivative) * out._derivative

        out._backward = _backward
        return out

    def tanh(self):
        import math

        # Forward pass: compute the hyperbolic tangent
        out = Autograd(math.tanh(self.value), (self,), "tanh")

        def _backward():
            # Local physics: d/dx tanh(x) = 1 - tanh^2(x)
            # Chain rule: local_derivative * global_derivative
            self._derivative += (1.0 - out.value**2) * out._derivative

        out._backward = _backward
        return out

    def backward(self, _derivative=1.0):
        # 1. Build the topological sequence dynamically
        topo_list = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                # Dive all the way down to the atomic nodes first
                for child in node._prev:
                    build_topo(child)
                # Only append after all constituents are fully explored
                topo_list.append(node)

        # Start the top-down dive from the output node
        build_topo(self)

        # 2. Seed the global tension
        self._derivative = float(_derivative)

        # 3. Execute closures in reverse topological order
        for node in reversed(topo_list):
            node._backward()
