#include <cstdint>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>

#include "Lazy.h"

// Compilation: g++ example-2.cc -pthread --std=c++17

template <class... Args>
void atomic_print(Args&&... args)
{
    std::stringstream ss;
    (ss << ... << args) << '\n';
    std::cout << ss.str();
}

// Returns the index of the element which matches the searched
// value. Indices on range from...to-1 are searched.
// Sets the stop token when found.
// The search is aborted if someone else has found the value.
// size_t(-1) is returned if not found.
template <class Vec, class T = typename Vec::value_type>
std::size_t indexOf(const Vec& vec, std::size_t from, std::size_t to, const T& value, Lazy::StopToken *token)
{
  for (auto i = from; i < to; ++i) {
    // Some sleep to make sure all threads really are running in parallel.
    // This hack simulates a long operation.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (*token)
      {
        atomic_print("indexOf [", from, ",",to,"]: Bailing out because stop token = ", token->value());
        return std::size_t(-1);
      }
    if (vec[i] == value) {
      int iOldTokenVal = token->setValue(1);
      atomic_print("indexOf [",from,",",to,"]: Value found. Token was ", iOldTokenVal,
                   ", now token = ", token->value());
      return i;
    }
  }
  return std::size_t(-1);
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
    std::cout << "Hey! Your machine has " << std::thread::hardware_concurrency() << " cores!\n";

    int iVectorLength = 10 * std::thread::hardware_concurrency();

    // Example 2.1: Call a function concurrently once for each element of the input vector and
    //              store the results to the output vector.
    std::cout << "\n*** Example 2.1 *** : Call a function for each value in the input vector.\n";
    std::vector<int> vecInput(iVectorLength);
    for (int i = 0; i < iVectorLength; ++i)
        vecInput[i] = 100 * i;

    {
        // Set vecOutput[i] = func(vecInput[i]) for each i running in a separate thread.
        // The number of parallel threads will be limited to the number of cores + 1
        // to avoid running the system out of resources.
        auto vecOutput = Lazy::runForAll(vecInput, intSqrt);
        std::cout << "2.1.1: Input vector length = " << vecInput.size() << ", output vector length = " << vecOutput.size() << "\n";
    }
    {
        // The number of parallel threads can also be given as a template parameter. Use 10 in this example.
        std::vector<double> vecOutput = Lazy::runForAll<10>(vecInput, [](auto x) { return intSqrt(x * 100) * 0.1; });
        std::cout << "2.1.2: Input vector length = " << vecInput.size() << ", output vector length = " << vecOutput.size() << "\n";
    }
    {
        // The input can also be an array. There will be as many parallel threads as
        // there are elements in the array. There will be no heap allocations.
        std::array<int, 10> arrInput;
        for (int i = 0; i < arrInput.size(); ++i)
            arrInput[i] = 100 * i;
        std::array arrOutput = Lazy::runForAll(arrInput, intSqrt);
        std::cout << "2.1.3: Input array length = " << arrInput.size() << ", output array length = " << arrOutput.size() << "\n";
    }
    {
        // Initializer lists are also supported. The output is an std::vector.
        auto vecOutput = Lazy::runForAll({33,22,77,99,88}, [](auto x) { return x - 0.5; });
        std::cout << "2.1.4: input values are {33,22,77,99,88}, output vector is {" <<
                     vecOutput[0] << ", " << vecOutput[1] << ", " << vecOutput[2] << ", " << vecOutput[3] << ", " << vecOutput[4] <<"}\n";

        // If you want to avoid heap allocation, you can use initialized std:array
        std::array arrOutput = Lazy::runForAll(std::array{33,22,77,99,88}, [](auto x) { return x - 0.5; });
        std::cout << "2.1.5: input values are {33,22,77,99,88}, output array is  {" <<
                     arrOutput[0] << ", " << arrOutput[1] << ", " << arrOutput[2] << ", " << arrOutput[3] << ", " << arrOutput[4] <<"}\n";
    }

    // Example 2.2: You can attach as many continuation functions as needed.
    //              For instance, if there are 3 functions f1,f2,f3, the result will be
    //              vecOutput[i] = f3(f2(f1((vecInput[i])))
    std::cout << "\n*** Example 2.2 *** : Run a set of continuations for each value in the input vector.\n";
    {
        // out = sqrt((10 * in) + 1.5)
        auto vecOutput = Lazy::runForAll(vecInput,
                                         [](auto x) { return 10 * x; },
                                         [](auto x) { return x + 1.5; },
                                         [](auto x) { return std::sqrt(x); });

        std::cout << "2.2: Input vector length = " << vecInput.size() << ", output vector length = " << vecOutput.size() << "\n";
        std::cout << "    Last input value = " << vecInput.back() << ", last output value = " << vecOutput.back() << ".\n";
    }

    // Example 2.3: One or more functions in one or more threads throw. The exception can be caught normally.
    std::cout << "\n*** Example 2.3 *** : The function may throw for some values of the input vector.\n";
    {
        try {
            auto vecOutput = Lazy::runForAll({2,1,0,-1,2},
                                         [](auto x) { return 100 * x; },
                                         [](auto x) { if (x < 0)
                                                        throw std::runtime_error("[[Negative sqrt]]");
                                                      return std::sqrt(x); });
        std::cout << "2.3: Output vector is  {" <<
                     vecOutput[0] << ", " << vecOutput[1] << ", " << vecOutput[2] << ", " << vecOutput[3] << ", " << vecOutput[4] <<"}\n";
        }
        catch (const std::exception& e) {
            std::cout << "EXCEPTION: " << e.what() <<"\n";
        }
    }

    // Example 2.4: Threads communicate with shared StopToken.
    // In this example a value is searched from a vector
    // in parallel threads until one of the threads find it.
    // The other threads give up as soon as a thread finds
    // the value.
    std::cout << "\n*** Example 2.4 *** : Abort the other parallel function calls as soon as one becomes ready.\n";
    {
      std::vector<int> vec(1000);
      // Fill in the vector with some values
      for (int i = 0; i < vec.size(); ++i)
        vec[vec.size() - i - 1] = 10 * i;

      // Make an array of {from, to}-index pairs. Can be either an array or a vector.
#if 1
      const std::size_t N = 10; // Number of parallel threads
      std::vector<std::pair<std::size_t, std::size_t>> aPairs(N);
#else
      constexpr std::size_t N = 10; // Number of parallel threads
      std::array<std::pair<std::size_t, std::size_t>, N> aPairs;
#endif
      double dFrom = 0, dTo = 0;
      for (auto i = 0; i < N; ++i) {
        dFrom = dTo;
        dTo += vec.size() / double(N);
        aPairs[i] = {std::size_t(dFrom), std::min(std::size_t(dTo), vec.size())};
      }
      aPairs[N-1].second = vec.size();

      int iFindMe = 5500;  // Find this value from vector vec.
      // Finder function which inputs a StopToken and an index pair (from, to)
      // and outputs the index of the searched value, or -1 if not found.
      auto finder = [value = iFindMe, &vec](Lazy::StopToken* token, auto indexPair)
        {
          return indexOf(vec, indexPair.first, indexPair.second, value, token);
        };

      // Run the finder in parallel for all index pairs
      // A StopToken object is created automatically by the library
      // because finder takes one as the first argument.
      auto vecIndex = Lazy::runForAll(aPairs, finder);

      std::cout << "2.4: Finder results for value "<<iFindMe<<" were:  (-1 == not found)\n";
      for (auto i = 0; i < N; ++i)
        std::cout << i << ": index range [" << aPairs[i].first << ","<< aPairs[i].second << "] : found at index = " << int(vecIndex[i]) << "\n";
    }
}
