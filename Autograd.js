export const PATH_STRATEGIES = Object.freeze({
  RANDOM_LENGTH: 0,
  BROWN_NOISE: 1,
  PREDETERMINED: 2,
});
export const STATUSES = Object.freeze({
  READY: 0,
  RUNNING: 1,
  COMPUTATION_PAUSED: 2,
  UNCOMPUTATION_PAUSED: 3,
});

export class Autograd {
  #cacheOutputs;
  #options;
  #isIONode = false;
  #status = null;
  #pool = null;
  #operation = null;
  #cache = [];

  constructor(
    operationPool,
    {
      strategy = PATH_STRATEGIES.RANDOM_LENGTH,
      params: { fullComputation = true, ...params } = {},
    },
    cacheOutputs = false,
  ) {
    this.#cacheOutputs = cacheOutputs;
    this.#options = { strategy, params: { fullComputation, ...params } };
    if ((this.#isIONode = Array.isArray(operationPool))) {
      this.#pool = operationPool.map((operation) => ({
        operation,
        autograd: null,
      }));
      this.#status = STATUSES.READY;
    } else {
      this.#operation = operationPool;
    }
  }

  #getLocalDerivative(cachedIndex, f_x1, localDerivativeInput) {
    if (this.#cache.length < 1 || cachedIndex < 1) {
      return 1.0;
    }

    const { input: x1 } = this.#cache[cachedIndex];
    const { input: x0 } = this.#cache[cachedIndex - 1];
    const f_x0 = this.#operation(x0);
    return localDerivativeInput * ((f_x1 - f_x0) / (x1 - x0));
  }

  #compute(input, { strategy, params } = {}) {
    const oldStatus = this.#status;
    this.#status = STATUSES.RUNNING;

    const _strategy = strategy ?? this.#options.strategy;
    const _params = params ?? this.#options.params;

    const _fullComputation =
      _params?.fullComputation ?? this.#options.params.fullComputation;

    if (_strategy === PATH_STRATEGIES.RANDOM_LENGTH) {
      let randomLength;

      if (oldStatus === STATUSES.READY) {
        const startingPool = [...this.#pool];
        randomLength = Math.floor(Math.random() * this.#pool.length) + 1;

        const nextIndex = Math.floor(Math.random() * this.#pool.length);
        const nextNode = startingPool.splice(nextIndex, 1)[0];

        if (!(nextNode.autograd instanceof Autograd)) {
          nextNode.autograd = new Autograd(
            nextNode.operation,
            this.#options,
            this.#cacheOutputs,
          );
        }

        const output = nextNode.autograd._next(
          {
            input,
            localDerivativeInput: 1.0,
            nodeRef: this,
          },
          {
            strategy: _strategy,
            params: {
              ..._params,
              randomLength,
              startingPool,
              fullComputation: _fullComputation,
            },
          },
        );

        const cache = {
          options: {
            strategy: _strategy,
            params: {
              ..._params,
              fullComputation: _fullComputation,
            },
          },
          input,
          output,
        };
        this.#cache.push(cache);
      } else if (oldStatus === STATUSES.COMPUTATION_PAUSED) {
        const { output: operationCache, input: cachedInput } =
          this.#cache[this.#cache.length - 1];

        randomLength = operationCache._options.params.randomLength;
        const nextNode = operationCache._target;

        const { lastNode, localDerivative, _output } = nextNode.autograd._next(
          operationCache._inputs,
          {
            ...operationCache._options,
            params: {
              ...operationCache._options.params,
              fullComputation: _fullComputation,
            },
          },
        );
        const cache = {
          options: {
            strategy: _strategy,
            params: {
              ..._params,
              fullComputation: _fullComputation,
            },
          },
          input: cachedInput,
          output: {
            lastNode,
            localDerivative,
            _output: this.#cacheOutputs ? _output : null,
          },
        };
        this.#cache[this.#cache.length - 1] = cache;
      }

      this.#status =
        _fullComputation || randomLength < 2
          ? STATUSES.READY
          : STATUSES.COMPUTATION_PAUSED;
    } else if (_strategy === PATH_STRATEGIES.BROWN_NOISE) {
      //
    } else if (_strategy === PATH_STRATEGIES.PREDETERMINED) {
      //
    }

    return this.#cacheOutputs ? this : _output;
  }

  #uncompute(output, stepwise = false) {
    const { lastNode, localDerivative, _output, output: keyedOutput } = output;
    const resolvedOutput = keyedOutput || _output;
  }

  get options() {
    return this.#options;
  }

  set options({
    strategy = PATH_STRATEGIES.RANDOM_LENGTH,
    params: { fullComputation = true, ...params } = {},
  }) {
    this.#options = { strategy, params: { fullComputation, ...params } };
    return this;
  }

  get isIoNode() {
    return this.#isIONode;
  }

  get status() {
    return this.#status;
  }
  get pool() {
    return [...this.#pool];
  }

  get operation() {
    return this.#operation;
  }

  get cache() {
    return [...this.#cache];
  }

  resetCache() {
    if (this.#isIONode) {
      this.#cache = [];
      this.#pool.forEach(({ operation, autograd }) => {
        autograd.resetCache();
      });
      this.#status = STATUSES.READY;
    } else {
      this.#cache = [];
    }
    return this;
  }

  compute(input, options) {
    if (!this.#isIONode) {
      throw new error("Autograd.compute(): only the IO Node can compute()");
    }
    if (
      this.#status !== STATUSES.READY &&
      this.#status !== STATUSES.COMPUTATION_PAUSED
    ) {
      throw new Error(
        "Autograd.compute(): cannot compute() unless status=STATUSES.READY or status=STATUSES.COMPUTATION_PAUSED",
      );
    }
    return this.#compute(input, options);
  }

  uncompute(output, stepwise = false) {
    if (!this.#isIONode) {
      throw new error("Autograd.uncompute(): only the IO Node can uncompute()");
    }
    if (
      this.#status !== STATUSES.READY &&
      this.#status !== STATUSES.UNCOMPUTATION_PAUSED
    ) {
      throw new Error(
        "Autograd.uncompute(): cannot uncompute() unless status=STATUSES.READY or status=STATUSES.UNCOMPUTATION_PAUSED",
      );
    }
    return this.#uncompute(output, stepwise);
  }

  forward() {
    if (!this.#isIONode) {
      throw new error("Autograd.forward(): only the IO Node can forward()");
    }
    if (this.#status !== STATUSES.COMPUTATION_PAUSED) {
      throw new Error(
        "Autograd.forward(): cannot forward() unless status=STATUSES.COMPUTATION_PAUSED",
      );
    }
    return this.compute();
  }

  backward(output) {
    if (!this.#isIONode) {
      throw new error("Autograd.backward(): only the IO Node can backward()");
    }
    if (this.#status !== STATUSES.UNCOMPUTATION_PAUSED) {
      throw new Error(
        "Autograd.backward(): cannot backward() unless status=STATUSES.UNCOMPUTATION_PAUSED",
      );
    }
    return this.uncompute(output, true);
  }

  finish() {
    if (!this.#isIONode) {
      throw new error("Autograd.finish(): only the IO Node can finish()");
    }
    if (this.#status !== STATUSES.COMPUTATION_PAUSED) {
      throw new Error(
        "Autograd.finish(): cannot finish() unless status=STATUSES.COMPUTATION_PAUSED",
      );
    }
    const { input, options: oldOptions } = this.#cache[this.#cache.length - 1];

    const newOptions = {
      ...oldOptions,
      params: { ...oldOptions.params, fullComputation: true },
    };

    return this.compute(input, newOptions);
  }

  unfinish(output) {
    if (!this.#isIONode) {
      throw new error("Autograd.unfinish(): only the IO Node can unfinish()");
    }
    if (this.#status !== STATUSES.UNCOMPUTATION_PAUSED) {
      throw new Error(
        "Autograd.unfinish(): cannot unfinish() unless status=STATUSES.UNCOMPUTATION_PAUSED",
      );
    }

    return this.compute(output, false);
  }

  getLastOutput() {
    if (!this.#isIONode) {
      throw new error(
        "Autograd.getLastOutput(): only the IO Node can getLastOutput()",
      );
    }
    if (this.#cache.length < 1) {
      console.warn("Autograd.getLastOutput(): no outputs in cache");
      return null;
    }
    return this.#cache[this.#cache.length - 1].output._output;
  }

  getLastNode() {
    if (!this.#isIONode) {
      throw new error(
        "Autograd.getLastNode(): only the IO Node can getLastNode()",
      );
    }
    if (this.#cache.length < 1) {
      console.warn("Autograd.getLastNode(): no history in cache");
    }
    return this.#cache[this.#cache.length - 1].output.lastNode;
  }

  getLastDerivative() {
    if (!this.#isIONode) {
      throw new error(
        "Autograd.getLastDerivative(): only the IO Node can getLastDerivative()",
      );
    }
    if (this.#cache.length < 1) {
      console.warn("Autograd.getLastDerivative(): no history in cache");
    }
    return this.#cache[this.#cache.length - 1].output.localDerivative;
  }

  _next({ input, localDerivativeInput, nodeRef }, { strategy, params } = {}) {
    if (this.#isIONode) {
      throw new error("Autograd._next(): the IO Node cannot _next()");
    }
    const iterationCache = {
      input,
      localDerivativeInput,
      nodeRef,
      options: {
        strategy: strategy ?? this.#options.strategy,
        params: {
          ...(params ?? this.#options.params),
          fullComputation:
            params?.fullComputation ?? this.#options.params.fullComputation,
        },
      },
    };

    const { index: cachedIndex } = Autograd.insertSorted(
      this.#cache,
      iterationCache,
      ({ input: a }, { input: b }) => a - b,
    );

    const output = this.#operation(input);
    const localDerivative = this.#getLocalDerivative(
      cachedIndex,
      output,
      localDerivativeInput,
    );

    if (iterationCache.options.strategy === PATH_STRATEGIES.RANDOM_LENGTH) {
      const { randomLength, startingPool, fullComputation } =
        iterationCache.options.params;

      if (randomLength >= 2) {
        const _startingPool = [...startingPool];
        const nextIndex = Math.floor(Math.random() * _startingPool.length);
        const nextNode = _startingPool.splice(nextIndex, 1)[0];

        if (!(nextNode.autograd instanceof Autograd)) {
          nextNode.autograd = new Autograd(
            nextNode.operation,
            this.#options,
            this.#cacheOutputs,
          );
        }

        return fullComputation
          ? nextNode.autograd._next(
              {
                input: output,
                localDerivativeInput: localDerivative,
                nodeRef: this,
              },
              {
                strategy: iterationCache.options.strategy,
                params: {
                  ...iterationCache.options.params,
                  randomLength: randomLength - 1,
                  startingPool: _startingPool,
                },
              },
            )
          : {
              _target: nextNode,
              _inputs: {
                input: output,
                localDerivativeInput: localDerivative,
                nodeRef: this,
              },
              _options: {
                strategy: iterationCache.options.strategy,
                params: {
                  ...iterationCache.options.params,
                  randomLength: randomLength - 1,
                  startingPool: _startingPool,
                },
              },
              lastNode: this,
              localDerivative,
              _output: output,
            };
      } else {
        if (randomLength < 1) {
          throw new error(
            "Autograd._next(): non-IO nodes with options.strategy=PATH_STRATEGIES.RANDOM_LENGTH must receive options.params.randomLength >= 1",
          );
        }
        return {
          lastNode: this,
          localDerivative,
          _output: output,
        };
      }
    } else if (
      iterationCache.options.strategy === PATH_STRATEGIES.BROWN_NOISE
    ) {
      //
    } else if (
      iterationCache.options.strategy === PATH_STRATEGIES.PREDETERMINED
    ) {
      //
    }
  }
  /**
   * Inserts an item into a pre-sorted array in-place using a binary search algorithm.
   *
   * @template T - The type of elements in the array.
   * @param {T[]} arr - The target pre-sorted array to modify.
   * @param {T} item - The element to insert into the array.
   * @param {function(T, T): number} compareFn - A function used to determine the order of the elements.
   * @returns {{ array: T[], index: number }} An object containing the mutated array and the index where the item was inserted.
   */
  static insertSorted(arr, item, compareFn) {
    let low = 0;
    let high = arr.length;

    while (low < high) {
      let mid = (low + high) >> 1;

      if (compareFn(arr[mid], item) < 0) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }

    // Insert the item at the calculated index
    arr.splice(low, 0, item);

    // Return both the array reference and the insertion index
    return { array: arr, index: low };
  }
}
