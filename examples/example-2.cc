#include <cstdint>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <cassert>

#include <Lazy/Lazy.h>

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

void myVoidFunction(int n)
{
  for(int i=0; i < n; ++i) {
    atomic_print("myVoidFunction says (", i, '/', n, ')');
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

int main()
{
    std::cout << "Hey! Your machine has " << std::thread::hardware_concurrency() << " cores!\n";

    // Make input vector and input array
    int iVectorLength = 10 * std::thread::hardware_concurrency();
    std::vector<int> vecInput(iVectorLength);
    for (int i = 0; i < iVectorLength; ++i)
        vecInput[i] = 100 * i;

    constexpr std::size_t szArrayLength = 100;
    std::array<int, szArrayLength> arrInput;
    for (std::size_t i = 0; i < szArrayLength; ++i)
        arrInput[i] = 100 * i;

    // Example 2.1: Call a function concurrently once for each element of the input vector and
    //              store the results to the output vector.
    std::cout << "\n*** Example 2.1 *** : Call a function for each value in the input vector.\n";
    {
        // Set vecOutput[i] = func(vecInput[i]) for each i running in a separate thread.
        // The number of parallel threads will be limited to the number of cores.
        // to avoid running the system out of resources.
        auto vecOutput = Lazy::runForAll(vecInput, intSqrt);
        std::cout << "2.1.1: Input vector length = " << vecInput.size() << ", output vector length = " << vecOutput.size() << "\n";
    }
    {
        // The number of parallel threads can also be given as a template parameter. Use 10 in this example.
        auto vecOutput = Lazy::runForAll<10>(vecInput, [](auto x) { return intSqrt(x * 100) * 0.1; });

        static_assert(std::is_same_v<decltype(vecOutput), std::vector<double>>,
                     "2.1.2: Output vector type does not match!");

        std::cout << "2.1.2: Input vector length = " << vecInput.size() << ", output vector length = " << vecOutput.size() << "\n";
    }
    {
        // The input can also be an array. There will be as many parallel threads as
        // there are elements in the array. There will be no heap allocations.
        auto arrOutput = Lazy::runForAll(arrInput, intSqrt);
        static_assert(std::is_same_v<decltype(arrOutput), std::array<uint16_t, szArrayLength>>,
                     "2.1.3: Output array type does not match!");
        std::cout << "2.1.3: Input array length = " << arrInput.size() << ", output array length = " << arrOutput.size() << "\n";
    }
    {
        // Initializer lists are also supported. The output is an std::vector.
        auto vecOutput = Lazy::runForAll({33,22,77,99,88}, [](auto x) { return x - 0.5; });
        auto vecOutput2 = Lazy::runForAll({33,22,77,99,88}, [](auto x) { return x - 0.5; }, [](auto x) { return 2*(x + 0.5); });
        static_assert(std::is_same_v<decltype(vecOutput), std::vector<double>>,
                     "2.1.4: Output vector type does not match!");
        static_assert(std::is_same_v<decltype(vecOutput2), std::vector<double>>,
                     "2.1.4: Output vector type does not match!");
        std::cout << "2.1.4: input values are {33,22,77,99,88}, output vector is {" <<
                     vecOutput[0] << ", " << vecOutput[1] << ", " << vecOutput[2] << ", " << vecOutput[3] << ", " << vecOutput[4] <<"}\n";

        // If you want to avoid heap allocation, you can use initialized std:array
        auto arrOutput = Lazy::runForAll(std::array{33,22,77,99,88}, [](auto x) { return x - 0.5; });
        auto arrOutput2 = Lazy::runForAll(std::array{33,22,77,99,88}, [](auto x) { return x - 0.5; }, [](auto x) { return 2*(x + 0.5); });
        static_assert(std::is_same_v<decltype(arrOutput), std::array<double, 5>>,
                     "2.1.5: Output array type does not match!");
        static_assert(std::is_same_v<decltype(arrOutput2), std::array<double, 5>>,
                     "2.1.5: Output array type does not match!");
        std::cout << "2.1.5: input values are {33,22,77,99,88}, output array is  {" <<
                     arrOutput[0] << ", " << arrOutput[1] << ", " << arrOutput[2] << ", " << arrOutput[3] << ", " << arrOutput[4] <<"}\n";
    }

    // Example 2.2: You can attach as many continuation functions as needed.
    //              For instance, if there are 3 functions f1,f2,f3, the result will be
    //              vecOutput[i] = f3(f2(f1((vecInput[i])))
    std::cout << "\n*** Example 2.2 *** : Run a set of continuations for each value in the input vector.\n";
    {
        // vector out = sqrt((10 * in) + 1.5).
        auto vecOutput = Lazy::runForAll(vecInput,
                                         [](auto x) { return 10 * x; },
                                         [](auto x) { return x + 1.5; },
                                         [](auto x) { return std::sqrt(x); });

        static_assert(std::is_same_v<decltype(vecOutput), std::vector<double>>,
                      "2.2: Output vector type does not match!");

        std::cout << "2.2: Input vector length = " << vecInput.size() << ", output vector length = " << vecOutput.size() << "\n";
        std::cout << "     Last input value = " << vecInput.back() << ", last output value = " << vecOutput.back() << ".\n";

        // array out = sqrt((10 * in) + 1.5)
        // Array input uses always as many threads as there are elements in the array.
        auto arrOutput = Lazy::runForAll(arrInput,
                                         [](auto x) { return 10 * x; },
                                         [](auto x) { return x + 1.5; },
                                         [](auto x) { return std::sqrt(x); });

        static_assert(std::is_same_v<decltype(arrOutput), std::array<double, szArrayLength>>,
                      "2.2: Output array type does not match!");

        std::cout << "     Input array length = " << arrInput.size() << ", output vector length = " << arrOutput.size() << "\n";
        std::cout << "     Last input value = " << arrInput.back() << ", last output value = " << arrOutput.back() << ".\n";
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
            std::cout << "EXCEPTION: " << e.what() << "\n";
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

      // Make an array of {from, to}-index pairs.
      // Can be either an array or a vector. Both are used for demonstration.
      constexpr std::size_t N = 10; // Number of parallel finder threads
      std::vector<std::pair<std::size_t, std::size_t>> vecPairs(N);
      std::array<std::pair<std::size_t, std::size_t>, N> arrPairs;

      double dFrom = 0, dTo = 0;
      for (auto i = 0; i < N; ++i) {
        dFrom = dTo;
        dTo += vec.size() / double(N);
        vecPairs[i] = {std::size_t(dFrom), std::min(std::size_t(dTo), vec.size())};
        arrPairs[i] = {std::size_t(dFrom), std::min(std::size_t(dTo), vec.size())};
      }
      vecPairs[N-1].second = vec.size();
      arrPairs[N-1].second = vec.size();

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
      // The job is done twice using both an array and a vector for demonstration.
      auto vecIndex = Lazy::runForAll(vecPairs, finder);
      auto arrIndex = Lazy::runForAll(arrPairs, finder);

      static_assert(std::is_same_v<decltype(arrIndex), std::array<std::size_t, N>>,
                    "2.4: Output array type does not match!");
      static_assert(std::is_same_v<decltype(vecIndex), std::vector<std::size_t>>,
                    "2.4: Output vector type does not match!");

      std::cout << "2.4: Finder results for value "<<iFindMe<<" were:  (-1 == not found)\n";
      for (auto i = 0; i < N; ++i) {
        std::cout << i << ": index range [" << arrPairs[i].first << ","<< arrPairs[i].second << "] : found at index = " << int(vecIndex[i]);
        if (int(vecIndex[i]) >= 0)
          std::cout << ", value is " << vec[vecIndex[i]] << ", should be " << iFindMe << "\n";
        else
          std::cout << "\n";
        if (arrPairs[i] != vecPairs[i])  // Should never go here
          std::cout << "2.4: Array vs. vector mismatch at index " << i << " !!\n";
      }
    }

    // Example 2.5: If the function return type is void, runForAll returns void.
    std::cout << "\n*** Example 2.5 *** : Functions with void return type can also be used. \n";
    {
        // Use nullptr_t as the dummy return type
        Lazy::runForAll({1,2,3,4}, [](auto n) { myVoidFunction(n); });
    }

    // Example 2.6: Input and output vectors may be preallocated and indexed by
    //              using Lazy::Sequence{N} as input vector.
    //              It looks as if it was std::vector<size_t> X = {0,1,..N-1} even though it has no data.
    std::cout << "\n*** Example 2.6 *** : Input and output vectors are preallocated and the function may return void.\n";
    {
        const std::size_t N = 5;
        std::vector<double> vecIn(N);          // Input vector
        std::vector<double> vecFractionOut(N); // 1st output vector
        std::vector<int> vecExponentOut(N);    // 2nd output vector

        // Prepare test input
        for (auto i : Lazy::Sequence{N})
          vecIn[i] = 0.1 * (i + 1) * std::pow(2.0, i);

        // Use Sequence{N} = {0,1,...N-1} as input to a lambda which returns void.
        Lazy::runForAll(Lazy::Sequence{N}, [&](std::size_t i)
          {
            vecFractionOut[i] = std::frexp(vecIn[i], &vecExponentOut[i]);
          });

        // Alternatively, one output can be returned as a vector and the other
        // as output parameter. The outcome is the same as above.
        auto vecFrac2 =  Lazy::runForAll(Lazy::Sequence{N}, [&](std::size_t i)
          {
            return std::frexp(vecIn[i], &vecExponentOut[i]);
          });

        // Sequence can also be used in range-based for-loops.
        for (auto i : Lazy::Sequence{N}) {
          std::cout << "Input # " << i << ": " << vecIn[i]
                    << " = "  << vecFractionOut[i] << " * 2^" << vecExponentOut[i] << "\n";
          assert(vecFrac2[i] == vecFractionOut[i]);
        }
    }
}
