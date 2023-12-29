#include <vector>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <thread>

#include <Lazy/Lazy.h>

using std::cout;
using std::size_t;

// Buffers smaller than this many elements are sorted with ordinary std::sort
static constexpr size_t smallBufferSize = 1024;

// Merges buffers pointed by pA and pB into the buffer pointed by pTarget.
// Returns the number of items in pTarget.
template <class T>
size_t mergeBuffers(const T* pA, size_t szA, const T* pB, size_t szB, T* pTarget)
{
    size_t indexA = 0;
    size_t indexB = 0;
    size_t indexTarget = 0;

    while ((indexA < szA) && (indexB < szB)) {
        if (pA[indexA] < pB[indexB])
            pTarget[indexTarget] = pA[indexA++];
        else
            pTarget[indexTarget] = pB[indexB++];
        indexTarget++;
    }

    // Copy the rest to pTarget. Only one of the while-loops will actually run.
    while (indexA < szA)
        pTarget[indexTarget++] = pA[indexA++];

    while (indexB < szB)
        pTarget[indexTarget++] = pB[indexB++];

    return indexTarget;
}


// Sorts the buffer of size sz in pA. The sorted values appear in either
// buffer pA or pB, depending on the parity of level.
//   Even level: input is in buffer pA, output will be also in buffer pA.
//   Odd level:  input is in buffer pA, output will be in buffer pB.
// Returns true if the sorting was done by std::sort and the recursion stops.
template <class T>
bool mergeSortImpl(T* pA, T* pB, size_t sz, int level = 0)
{
    // Stop the recursion if the array is to short or the recursion level is too deep.
    if (sz <= smallBufferSize || ((1<<level) >= std::thread::hardware_concurrency())) {
        if (level & 0x1) { // Store the result in pB because the caller wants so.
            // Sort in two halves and merge to pB.
            size_t mid = sz / 2;
            std::sort(pA, pA + mid);
            std::sort(pA + mid, pA + sz);
            mergeBuffers(pA, mid, pA + mid, sz - mid, pB);
        }
        else  // Sort in place at pA
            std::sort(pA, pA + sz);
        return true;
    }

    size_t mid = sz / 2;
    int nextLevel = level + 1;

    // Make a future for sorting the first half of the vector
    auto f = Lazy::future<bool>(pA, pB, mid, nextLevel).
        then([](auto* pA, auto* pB, size_t mid, int level){return mergeSortImpl(pA, pB, mid, level);}).
        finalize();
    // Start the sort
    auto f_run = f.run();

    // Sort the other half while the first half is being sorted in another thread.
    mergeSortImpl(pA + mid, pB + mid, sz - mid, nextLevel);

    // Get the result (even though the boolean output is ignored.)
    f.get(f_run);

    if (nextLevel & 0x1)  // Output is in B so merge to A
        mergeBuffers(pB, mid, pB + mid, sz - mid, pA);
    else  // Output is in A so merge to B
        mergeBuffers(pA, mid, pA + mid, sz - mid, pB);
    return false;
}

// Sorts the vector in non-descending order.
// The order of equal elements is not guaranteed to be preserved.
// Allocates a temporary buffer of vec.size()*sizeof(T) bytes.
template <class T>
void mergeSort(std::vector<T>& vec)
{
    std::vector<T> tmp(vec.size());
    mergeSortImpl(vec.data(), tmp.data(), vec.size());
}


int main()
{
    cout << "Merge sort. Max threads  = " << std::thread::hardware_concurrency() << "\n";

    for (int ii = 1; ii <= 10; ++ii) {
        // Make a test vector of random(ish) size.
        size_t sz = 2000511 - (rand() % 1024);
        std::vector<float> vec(sz);
        for (auto& x : vec)
            x = (rand() - (RAND_MAX/2)) / 65536.0;

        // Copy to compare the performance with std::sort
        auto vecCopy = vec;

        // Sort with std::sort and time.
        auto start1 = std::chrono::high_resolution_clock::now();
        std::sort(vecCopy.begin(), vecCopy.end());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration1 = end1 - start1;

        // Sort with merge sort and time.
        auto start2 = std::chrono::high_resolution_clock::now();
        mergeSort(vec);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration2 = end2 - start2;

        cout << "Round " << ii << ": vec.size = " << vec.size()
             <<", std::sort: " << duration1.count() << " sec, Merge sort: " << duration2.count() << " sec."
             << " Speed-up = " << 100*(1.0 - duration2.count() / duration1.count()) << " %\n";

        // Verify
        for (size_t jj = 0; jj < vec.size(); ++jj)
            assert(vec[jj] == vecCopy[jj]);
    }
    return 0;
}
