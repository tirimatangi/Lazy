#include <cstdint>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>

#include <Lazy/Lazy.h>

template <class... Args>
void atomic_print(Args&&... args)
{
    std::stringstream ss;
    (ss << ... << args) << '\n';
    std::cout << ss.str();
}

int main()
{
    // Example 3.1: Chains of continuations running in parallel.
    std::cout << "\n*** Example 3.1 *** : Chains of continuations running in parallel. Some may throw.\n";

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

    // The first function in the chain of continuations takes two parameters and
    // the last function returns a vector which contains {x, x*x, ... x^n}
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
                then([](auto vec) {
                    atomic_print("g: The vector has ", vec.size(), " elements.");
                    // Simulate an operation on the vector
                    std::this_thread::sleep_for(std::chrono::seconds(1));
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
        std::cout << "Future g returned {";
        for (auto v : vec)
            std::cout << v << ", ";
        std::cout << "}\n";
    } // If both futures throw, the first one will be caught here.
      // The other one will be ignored silently and the exception object
      // released so there will be no memory leak.
    catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() <<"\n";
    }
}
