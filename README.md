
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

The examples have been tested with gcc 9.3.0 and compiled with `g++ example.cc -pthread --std=c++17`.

#### 1. Run any number of functions in parallel and get the results as a tuple

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

There are more examples on how to use `Lazy::runParallel` in [example-1.cc](example-1.cc).
For an example on how to use stop tokens to communicate between the functions, see example 1.2 in example-1.cc.

#### 2. Vector in, vector out

You can run a function or a set of continuations in parallel for each element of the input vector and get the results as a vector.
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

Notice that if the input were an `std::array<int, 4>` instead of `std::vector<int>`,
the output would be an `std::array<double, 4>` instead of `std::vector<double>`.
There are no heap allocations if the input is an `std::array`.

By default, the maximum number of parallel threads running at any time is the number of cores+1.
You can set the maximum manually with template argument, for example `auto vecOutput = Lazy::runForAll<128>(vecInput, ...);`

Here is an example on how a function which gets ready first can abort the others using `Lazy::StopToken`.
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
    // Fill in the vector with some values
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
    // because the finder takes one as the first argument.

    auto vecIndex = Lazy::runForAll({std::pair{0, 250}, {250, 500}, {500, 750}, {750, 1000}}, finder);

    for (int i = 0; i < 4; ++i)
      std::cout << "Quarter #" << i << " returns index " << int(vecIndex[i]) <<
                   (int(vecIndex[i]) >= 0 ? "-> FOUND !" : "-> not found") << '\n';
}

```
For other methods provided by `Lazy::StopToken`, see `class StopToken` in the beginning of `Lazy.h`.

For more examples on how to use `Lazy::runForAll`, see [example-2.cc](example-2.cc).



#### 3. Use futures and continuations in manual mode

You can define the futures and the continuations they will run manually.
The first function in the chain of continuations can have any number of input parameters which are passed to function `Lazy::future<ReturnType>(...)`.
`ReturnType` is the return type of the last function in the chain of continuations (like in std::future).

Here is an example on lauching the tasks and getting the results once they are available.

```c++
    // Define future #1:
    // The first function in the chain of continuations takes one parameter and
    // the last function returns a double.
    // The output of the last continuation will be sqrt(i^2 - 2*i - 1), where is the input
    // to the first continuation.

    int iInput = 10;  // Set iInput = 0 to raise an exception.
    auto f = Lazy::future<double>(iInput).
                then([](auto i){ return std::vector{i * i, -2 * i, -1}; }).
                then([](const auto& vec) {return std::accumulate(vec.begin(), vec.end(), 0.0);}).
                then([](auto x) { if (x < 0) throw std::runtime_error("f: Error: negative value detected!"); return x;}).
                then([](auto x){ return std::sqrt(double(x)); }).
                finalize();

    // Define future #2:
    // The function takes two parameters and
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
Future g returned {2 4 8 16 32 64 128 256 512 1024 }
```
For more examples on how to use `Lazy::future` manually, see [example-3.cc](example-3.cc).

