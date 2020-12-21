#include <cstdint>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>

#include "Lazy.h"

// Compilation: g++ example-1.cc -pthread --std=c++17

template <class... Args>
void atomic_print(Args&&... args)
{
    std::stringstream ss;
    (ss << ... << args) << '\n';
    std::cout << ss.str();
}

// Sleeps x seconds and, if the token is not set, sets the token and
// returns sqrt(x). If the token is set by someone else, returns NAN.
double myFuncStopToken(Lazy::StopToken *token, int x)
{
  int timeLeft = x;
  atomic_print("myFuncStopToken: Token will be set in ", timeLeft, " secs.");

  while (timeLeft > 0) {
    if (*token) // Is token value non-zero?
    {
      atomic_print("x = ", x, ": Someone else has set stop token to value ");
      return NAN; // Someone else has already finished so give it up
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    --timeLeft;
  }
  int iTokenVal = token->setValue(x);
  atomic_print("myFuncStopToken: call with input ", x,
               " finished. Token was ", iTokenVal,
               ", now token = ", token->value());
  return std::sqrt(double(x));
}
// For other methods provided by Lazy::StopToken,
// see `class StopToken` in the beginning of Lazy.h.


// Calculates square root of a 32-bit integer
uint16_t intSqrt(uint32_t x)
{
  uint32_t uS = 0; // Initial guess for sqrt
  for (int iBit = 15; iBit >= 0; --iBit) {
    uint32_t uTest = uS + (1u << iBit);
    if (uTest * uTest <= x)
      uS = uTest;
  }
  return uint16_t(uS);
}


int main()
{
    // Example 1.1: run 4 functions in parallel and return the values as a tuple.
    std::cout << "\n*** Example 1.1 *** : Run functions in parallel and return the values as a tuple.\n";
    auto [s2int, s2dbl, s3int, s3dbl] = Lazy::runParallel(
        [init = 2]() { return intSqrt(1000000*init); }, // integer
        [init = 2]() { return std::sqrt(init); },       // double
        [init = 3]() { return intSqrt(1000000*init); }, // integer
        [init = 3]() { return std::sqrt(init); });      // double

    std::cout << "int vs double sqrt: "
            << "1000*sqrt2 = " << s2int
            << ", sqrt2 dbl = " << s2dbl
            << ", 1000*sqrt3 = " << s3int
            << ", sqrt3 dbl = " << s3dbl
            << "\n";

    // Example 1.2: Run 3 stoppable functions in parallel.
    //            If any of them sets the token, the others will voluntarily bail out.
    std::cout << "\n*** Example 1.2 *** : Functions share a stop token.\n";
    auto [a3, a2, a5] = Lazy::runParallel(
        [init = 3](Lazy::StopToken *st) { return myFuncStopToken(st, init); },
        [init = 2](Lazy::StopToken *st) { return myFuncStopToken(st, init); },
        [init = 5](Lazy::StopToken *st) { return myFuncStopToken(st, init); });
    std::cout << "Stop Token test:"
            << " a3 = " << a3 << ", a2 = " << a2 << ", a5 = " << a5 << "\n";


    // Example 1.3: Run 3 functions in parallel and catch an exception if one or more
    // functions throw.
    std::cout << "\n*** Example 1.3 *** : One or more functions may throw an exception.\n";
    int init = 3; // 0 or 1 mean no exception. 2, 3 or 4 mean an exception will be thrown

    try {
        auto [dd, ii, ss] = Lazy::runParallel(
            [init](){if (init > 1) throw std::runtime_error("[[Init > 1]]");
                        return init * 3.14;},  // double
            [init](){if (init > 2) throw std::runtime_error("[[Init > 2]]");
                        return init + 1;},    // integer
            [init](){if (init > 3) throw std::runtime_error("[[Init > 3]]");
                        return "init = " + std::to_string(init);});  // string
        std::cout << "dd = " << dd << ", ii = " << ii << ", ss = " << std::quoted(ss) << " (no exception was thrown)\n";
    }
    catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() <<"\n";
    }
}
