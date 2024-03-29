
# A fast and easy-to-use library for asynchronous operations with continuations

There are many problems with using std::futures and std::promises when launching asynchronous operations.
They require a shared state allocated from the heap, as well as reference counting and synchronization. Hence, they are slow.
Eric Niebler proposes a fast stack-allocated replacement for futures and promises
[in his talk](https://www.youtube.com/watch?v=tF-Nz4aRWAM) at CppCon 2019 (the good bits start at 13:00).

_**Lazy**_ is an enhanced version of the core code he presents in the slides on the video.
The name stems from the future-like objects which employ lazy evaluation.
With Lazy you can run any number of tasks in parallel and attach any number of continuations.
Heap allocation is needed only if the the return value of a function is a vector.
It is an entirely header-only library and it should work with any C++17 compiler.

Should any parallel operation throw, the exception can be caught normally.
You can also use rudimentary stop tokens for letting a parallel task tell the other parallel tasks that their work is no longer needed.

## Three basic ways to use the library


### 1. Run any number of functions in parallel and get the results as a tuple

In this example, one of the three parallel functions may throw.

```c++
    int init = 10; // -1 will throw
    try {
        auto [i, d, s] = Lazy::runParallel(
                            [init](){ return 10*init; },
                            [init](){ if (init < 0) throw std::runtime_error("[[init < 0]]");
                                      return std::sqrt(init); },
                            [init](){ return std::to_string(init); });
        std::cout << "i = " << i << ", d = " << d << ", s = " << std::quoted(s) << '\n';
    }
    catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() <<"\n";
    }
```

The output will be `i = 100, d = 3.16228, s = "10"`.


There are more examples on how to use `Lazy::runParallel` in [example-1.cc](https://github.com/tirimatangi/Lazy/blob/main/examples/example-1.cc).
For an example on how to use stop tokens to communicate between the functions, see example 1.2 in [example-1.cc](https://github.com/tirimatangi/Lazy/blob/main/examples/example-1.cc).

### 2. Vector in, vector out using either disposable threads or a threadpool

#### 2.1. Disposable threads

You can run a function or a set of continuations in parallel for each element of the input vector and get the results as a vector.
By default, the maximum number of parallel threads is the number of cores in your machine.
You can set the number of threads manually with template argument, for example `auto vecOutput = Lazy::runForAll<128>(vecInput, ...);`
The threads will be disposable in the sense that the threads will die when the function returns.
Examples 2.1 to 2.6 in [example-2.cc](https://github.com/tirimatangi/Lazy/blob/main/examples/example-2.cc) show how to use
disposable threads with function `Lazy::runForAll`.

Here is an example on running a sequence of 3 continuations where the inputs are in std::vector<int> and the output is an std::vector<double>.

```c++
    try {
        std::vector<int> vecInput {0, 10, 20, 30};  // E.g. {0,-10,-20,-30} would throw
        // Calculates out = sqrt(10 * in + 1.5) split into 3 continuations for demonstration.
        auto vecOutput = Lazy::runForAll(vecInput,
                                         [](auto x) { return 10 * x; },
                                         [](auto x) { return x + 1.5; },
                                         [](auto x) { if (x < 0) throw std::runtime_error("[[negative sqrt!]]");
                                                      return std::sqrt(x); });
        for (int i = 0; i < vecInput.size(); ++i)
            std::cout << vecInput[i] << " --> " << vecOutput[i] << '\n';
    }
    catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() <<"\n";
    }
```

The output will be
```
0 --> 1.22474
10 --> 10.0747
20 --> 14.1951
30 --> 17.3638
```

Notice that if the input were an `std::array<int, 4>` instead of `std::vector<int>`,
the output would be an `std::array<double, 4>` instead of `std::vector<double>`.
There are no heap allocations if the input is an `std::array`.

If the return type of your function is `void`, the return type of `Lazy::runForAll` is also void.
For example, the input and output vectors may be preallocated and indexed with a vector-like range container `Lazy::Sequence`. It behaves as if it was `std::vector<size_t> v = {0...N-1}` but does not consume memory at all.

Here is an example which uses one preallocated input vector to calculate two preallocated output vectors.
```c++
    {
        const std::size_t N = 4;
        std::vector<double> vecIn = {0.1, 0.2, 1.2, 2.4};  // Input vector
        std::vector<double> vecFractionOut(N); // 1st output vector
        std::vector<int> vecExponentOut(N);    // 2nd output vector

        // Use Sequence{N} to index the vectors.
        Lazy::runForAll(Lazy::Sequence{N}, [&](std::size_t i)
          {
            vecFractionOut[i] = std::frexp(vecIn[i], &vecExponentOut[i]);
          });

        // Print the results
        for (auto i : Lazy::Sequence{N}) {
          std::cout << "Input # " << i << ": " << vecIn[i]
                    << " = "  << vecFractionOut[i] << " * 2^" << vecExponentOut[i] << "\n";
        }
    }
```

The output will be
```
Input # 0: 0.1 = 0.8 * 2^-3
Input # 1: 0.2 = 0.8 * 2^-2
Input # 2: 1.2 = 0.6 * 2^1
Input # 3: 2.4 = 0.6 * 2^2
```

For more information on `Lazy::Sequence` and functions with preallocated output vectors,
see examples 2.5 and 2.6 in [example-2.cc](https://github.com/tirimatangi/Lazy/blob/main/examples/example-2.cc).

Here is an example on how a function which gets ready first can abort the others by using `Lazy::StopToken`.
The task is to find a number from a vector using 4 parallel threads.

```c++
template <class Vec, class T = typename Vec::value_type>
std::size_t indexOf(const Vec& vec,
                   std::size_t from,
                   std::size_t to,
                   const T& value,
                   Lazy::StopToken *token)
{
  for (auto i = from; i < to; ++i) {
    if (*token)
        return std::size_t(-1); // Abort
    if (vec[i] == value) {
        token->setValue(1);
        return i;
    }
  }
  return std::size_t(-1); // Not found
}
//...
{
    std::vector<int> vec(1000);
    // Fill the vector with some values
    for (int i = 0; i < vec.size(); ++i)
        vec[i] = i;

    int iFindMe = 543;  // Find this value from vector vec.

    // Finder function which inputs a StopToken and an index pair (from, to)
    // and outputs the index of the searched value, or -1 if not found.
    auto finder = [value = iFindMe, &vec](Lazy::StopToken* token, auto indexPair)
    {
        return indexOf(vec, indexPair.first, indexPair.second, value, token);
    };

    // Run the finder in parallel using 4 threads.
    // A StopToken object is created automatically by the library
    // because the finder takes a pointer to a Stoptoken as the first argument.

    auto vecIndex = Lazy::runForAll({std::pair{0, 250}, {250, 500}, {500, 750}, {750, 1000}}, finder);

    for (int i = 0; i < 4; ++i)
      std::cout << "Quarter #" << i << " returns index " << int(vecIndex[i]) <<
                   (int(vecIndex[i]) >= 0 ? "-> FOUND !" : "-> not found") << '\n';
}

```

The output will be
```
Quarter #0 returns index -1 -> not found
Quarter #1 returns index -1 -> not found
Quarter #2 returns index 543 -> FOUND !
Quarter #3 returns index -1 -> not found
```

For other methods provided by `Lazy::StopToken`, see `class StopToken` in the beginning of [Lazy.h](https://github.com/tirimatangi/Lazy/blob/main/include/Lazy/Lazy.h).

For many more examples on how to use `Lazy::runForAll`, see [example-2.cc](https://github.com/tirimatangi/Lazy/blob/main/examples/example-2.cc).

#### 2.2. Threadpool

If you repeatedly call the same function with different input and output vectors, it may be faster to
launch a permanent threadpool instead of using disposable threads as described above in section 2.1.
However, disposable threads are lock-free whereas the threadpool must use a mutex and condition
variables to manage the states of the threads. Hence, you should measure which method gives
a better performance in your use case.
Example 2-9 in [example-2.cc](https://github.com/tirimatangi/Lazy/blob/main/examples/example-2.cc)
is an example of such a measurement.

Example 2-7 gives an example of a use case where the return type of the function is a vector
so a call to the threadpool maps a vector of integers into a vector of vectors of integers.

Example 2-8 demonstrates how to run a void function in the threadpool and how to deal with
exceptions and stop tokes for cancelling the run prematurely.

Here is a brief example on how to use the threadpool. In this case,
the same input vector is used for each `multiplier` value.

```c++
    {
        int multiplier = 1;
        auto func = [&multiplier](double x) {
            return multiplier * std::exp(x);
        };

        std::vector<double> vecIn(256), vecOut(256);
        // Fill in the input vector
        for (int i = 0; i < 256; ++i)
            vecIn[i] = std::cos((3.141592654 / 256) * i);

        // Start the threadpool
        auto thrPool = Lazy::ThreadPool(func);

        // Fill in the output vector several times using different parameters
        while (multiplier <= 1000000) {
            // Set vecOut[i] = func(vecIn[i]) for each i in parallel.
            thrPool.run(vecIn.data(), vecOut.data(), vecIn.size());
            // Do something with vecOut...
            std::cout << "multiplier = " << multiplier
                      << ", vecOut[first] = " << vecOut[0]
                      << ", vecOut[last] = " << vecOut[255] << "\n";
            multiplier *= 10;
        }
    }
```

The output will be
```
multiplier = 1, vecOut[first] = 2.71828, vecOut[last] = 0.367907
multiplier = 10, vecOut[first] = 27.1828, vecOut[last] = 3.67907
multiplier = 100, vecOut[first] = 271.828, vecOut[last] = 36.7907
multiplier = 1000, vecOut[first] = 2718.28, vecOut[last] = 367.907
multiplier = 10000, vecOut[first] = 27182.8, vecOut[last] = 3679.07
multiplier = 100000, vecOut[first] = 271828, vecOut[last] = 36790.7
multiplier = 1000000, vecOut[first] = 2.71828e+06, vecOut[last] = 367907
```

### 3. Use futures and continuations in manual mode

You can also define the futures and the continuations and launch them manually at proper places.
The chain of continuations is defined by calling `Lazy::future<ReturnType>(args...)`.
The given arguments are passed to the first continuation function.<br>
`ReturnType` is the return type of the last function in the chain of continuations (like in std::future).<br>
The thread running the continuation is started by calling method `run` and the result is collected with `get`.

Here is an example on launching the tasks and getting the results as they become available.

```c++
    // Define future #1:
    // The first function in the chain of continuations takes one parameter and
    // the last function returns a double.
    // The output of the last continuation will be sqrt(i^2 - 2*i - 1), where i is the input
    // to the first continuation.
    int iInput = 10;  // Set iInput = 0 to raise an exception.
    auto f = Lazy::future<double>(iInput).
                then([](auto i){ return std::vector{i * i, -2 * i, -1}; }).
                then([](const auto& vec) {return std::accumulate(vec.begin(), vec.end(), 0.0);}).
                then([](auto x) { if (x < 0) throw std::runtime_error("f: Error: negative value detected!"); return x;}).
                then([](auto x){ return std::sqrt(double(x)); }).
                finalize();

    // Define future #2:
    // The function takes two parameters (x, n) and
    // returns a vector which contains {x, x*x, ... x^n}.
    double dValue = 2.0;  // Set dValue=0 and iN = -1 to raise an exception.
    int iN = 10; // Negative iN means negative exponents.
    auto g = Lazy::future<std::vector<double>>(dValue, iN).
                then([](double x, int n) {
                    if (x == 0 && n < 0)
                        throw std::runtime_error("g: Error: n must be positive if x is zero!");
                    int iSign = (n < 0) ? -1 : 1;
                    std::vector<double> vec(iSign * n);
                    double xx = x;
                    for (auto& v : vec) {
                        v = xx;
                        xx = (iSign < 0) ? xx / x : xx * x;
                    }
                    return vec; }).
                finalize();

    // Launch the threads.
    std::cout << "Calling f.run...\n";
    auto f_run = f.run();
    std::cout << "Calling g.run...\n";
    auto g_run = g.run();

    // Do something else while f and g are running...

    // Now get the results
    try {
        double dResult = f.get(f_run);
        std::vector<double> vec = g.get(g_run);

        std::cout << "Future f returned " << dResult << "\n";
        std::cout << "Future g returned { ";
        for (auto v : vec)
            std::cout << v << " ";
        std::cout << "}\n";
    } // If both futures throw, the first one will be caught here.
      // The other one will be ignored silently and the exception object
      // released so there will be no memory leak.
    catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() <<"\n";
    }
```

The output will be
```
Calling f.run...
Calling g.run...
Future f returned 8.88819
Future g returned { 2 4 8 16 32 64 128 256 512 1024 }
```
For more examples on how to use `Lazy::future` manually, see [example-3.cc](https://github.com/tirimatangi/Lazy/blob/main/examples/example-3.cc) and [example-mergesort.cc](https://github.com/tirimatangi/Lazy/blob/main/examples/example-mergesort.cc).

Mergesort gives a practical example on how parallelism can help sort large arrays significantly faster.
You can copy and paste lines 1-99 in [example-mergesort.cc](https://github.com/tirimatangi/Lazy/blob/main/examples/example-mergesort.cc) to your own code for a fast parallel vector sorter with prototype:

```c++
// Sorts the vector in non-descending order.
// The order of equal elements is not guaranteed to be preserved.
template <class T>
void mergeSort(std::vector<T>& vec);
```

## Compilation

The easiest way to compile all examples is to do
`cmake -DCMAKE_BUILD_TYPE=Release examples` followed by `make`.
If you don't want to use cmake, the examples can be compiled manually one by one. For instance, <br>
`g++ examples/example-1.cc -std=c++17 -I include/ -O3 -pthread -o example-1`

The examples have been tested with g++ 11.2.0  and clang++ 13.0.0 but any compiler which complies with c++17 standard should do.
The compiler can be switched from gcc to clang by building the examples with `cmake examples -DCMAKE_CXX_COMPILER=clang++`.
