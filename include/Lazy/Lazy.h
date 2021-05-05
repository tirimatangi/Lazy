#ifndef LAZY_H
#define LAZY_H

#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <initializer_list>

// A library for running functions in parallel
// using future-like objects with continuation
// and lazy evaluation.
// This is work is partly based on ideas presented by Eric Niebler
// at CppCon 2019 in his talk “A Unifying Abstraction for Async in C++”
// https://www.youtube.com/watch?v=tF-Nz4aRWAM

namespace Lazy
{

// An std::stop_token - like class.
// All it does is to wrap an atomic integer.
class StopToken
{
public:
  // Sets token to a new value and returns the old value
  int setValue(int v) noexcept
  {
    return _value.exchange(v);
  }

  // Increments the value atomically and returns the new value.
  int increment() noexcept
  {
    return ++_value;
  }

  // Decrements the value atomically and returns the new value
  int decrement() noexcept
  {
    return --_value;
  }

  // Returns the current value
  int value() const noexcept
  {
    return _value.load();
  }

  // Returns true if the value is non-zero.
  operator bool() const noexcept
  {
    return bool(value());
  }

private:
  std::atomic<int> _value{0};
};

// An std::vector<std::size_t> - like object which behaves as if it was filled with std::iota,
// meaning that vec[i] = i for i = 0...size()-1.
// Also iterators vec.begin() and vec.end() work so range-based for-loops work.
class Sequence
{
public:
  using value_type = const std::size_t;
  using size_type = std::size_t;

  Sequence(std::size_t sz = 0) : N(sz)  {}

  value_type operator[](std::size_t i) const noexcept
  {
      return i;
  }

  bool empty() const noexcept
  {
      return !bool(N);
  }

  size_type size() const noexcept
  {
      return N;
  }

  void resize(size_type newSize)
  {
      N = newSize;
  }

  class Iterator {
  public:
    typedef std::bidirectional_iterator_tag iterator_category;
    typedef const std::size_t        value_type;
    typedef std::ptrdiff_t           difference_type;
    typedef value_type*              pointer;
    typedef value_type&              reference;

    Iterator() : _n(0), _maxN(0) {}
    Iterator(std::size_t n, std::size_t mx) : _n(n), _maxN(mx) {}

  reference operator*() const
  {
    return _n;
  }

  Iterator& operator++()
  {
    if (_n < _maxN)
      ++_n;
    return *this;
  }

  Iterator operator++(int)
  {
    Iterator temp = *this;
    ++*this;
    return temp;
  }

  Iterator& operator--()
  {
    if (_n > 0)
      --_n;
    return *this;
  }

  Iterator operator--(int)
  {
    Iterator temp = *this;
    --*this;
    return temp;
  }

  bool operator== (const Iterator& it) const
  {
      return _n == it._n;
  }

  bool operator!= (const Iterator& it) const
  {
      return _n != it._n;
  }

  private:
    std::size_t _n = 0;
    std::size_t _maxN = 0;
  }; // class Iterator

  Iterator begin() const
  {
      return Iterator(0, N);
  }

  Iterator end() const
  {
      return Iterator(N, N);
  }

private:
  std::size_t N;
};  // class Sequence

struct Empty
{
  void operator()() const noexcept {}
};

// A promise-like object which is always allocated from the stack.
// If holds an other promise and the function which is applied
// at set_value() before storing the result to the upper level promise.
template <class Prom, class Func>
class Promise
{
public:
  Promise(Prom p, Func f) : _prom(p), _fun(f){};

  // Overload for _fun with parameters
  template <class... V>
  void set_value(const V&... vs)
  {
    _prom.set_value(_fun(vs...));
  }

  // Overload for _fun with no parameters
  void set_value(const Empty&)
  {
    _prom.set_value(_fun());
  }

  template <class E>
  void set_exception(const E& e)
  {
    _prom.set_exception(e);
  }

protected:
  Prom _prom;
  Func _fun;
};

// Joins the given thread if it is joinable and returns true if success.
inline bool joinThread(std::thread* pThr)
{
  if (pThr && pThr->joinable())
    {
      pThr->join();
      return true;
    }
  return false;
}

// Non-movable, non-copyable part of SyncState (see below).
// Unlike SyncState, it does not depend on return type T.
struct SyncCore
{
  std::mutex mtx;
  std::condition_variable cv;
  std::thread thr;
  ~SyncCore() { joinThread(&thr); }
};

// State for holding the result of type T or an exception.
template <class T>
struct SyncState
{
  SyncCore* core = nullptr;
  // Exception pointer is assumed to be stored at index <1> and
  // the data at index <2>.
  std::variant<std::monostate, std::exception_ptr, T> data = std::monostate();
  std::thread* getThread() const
  {
    if (core)
      return &(core->thr);
    else
      return nullptr;
  }
  ~SyncState() { joinThread(getThread()); }
};

// Promise-like object which stores atomically
// the result of a task and notifies to whom it
// may concern that either the result is ready or
// an exception has been thrown.
// SyncPromise is a concrete object which does not
// depend on any lower level promises.
template <class T, class CallBack>
class SyncPromise
{
public:
  SyncState<T>* pst;
  CallBack* pValueIsSet;

  template <class... V>
  void set_value(const V&... vs)
  {
    _set<2>(vs...);
  }
  template <class E>
  void set_exception(const E& e)
  {
    _set<1>(e);
  }

private:
  template <int I, class... V>
  void _set(const V&... vs)
  {
    std::unique_lock lck(pst->core->mtx);
    pst->data.template emplace<I>(vs...);
    lck.unlock();
    pst->core->cv.notify_one();
    // Callback to notify that the work is done
    // and the result is available for Future::get().
    if (pValueIsSet != nullptr) {
      (*pValueIsSet)();
    }
  }
};

// Returns a lambda which inputs a promise-like
// object p and creates a thread which sets a value to the promise.
// The lambda returns the thread.
// Note that the parameter u the only copy made of the input.
// From now on, u will be moved, not copied.
// This function will be used to initialize the ".then"-chain
// (i.e. _task owned by Future class, see below.)
template <class U>
auto makeNewThread(U u)
{
  return [uu = std::move(u)](auto p) -> std::thread {
    std::thread t{[p = std::move(p), &uu]() mutable {
      try
        {
          p.set_value(uu);
        }
      catch (...)
        {
          // Store exception. It will be rethrown in Future::get().
          // Note: exception_ptr is a shared pointer. The exception object
          // will be destroyed when the pointer eventually goes out of scope.
          p.set_exception(std::current_exception());
        }
    }};
    return t;
  };
}

// Overload for two input arguments.
template <class U1, class U2>
auto makeNewThread(U1 u1, U2 u2)
{
  return [uu1 = std::move(u1), uu2 = std::move(u2)](auto p) -> std::thread {
    std::thread t{[p = std::move(p), &uu1, &uu2]() mutable {
      try
        {
          p.set_value(uu1, uu2);
        }
      catch (...)
        {
          p.set_exception(std::current_exception());
        }
    }};
    return t;
  };
}

// Overload for several input arguments.
// Note that now the arguments will be copied, not moved
// because the capture list can not contain move(u)... until C++20.
template <class... U>
auto makeNewThread(const U&... u)
{
  return [u...](auto p) -> std::thread { // Note: [u...] makes a copy!
    std::thread t{[p = std::move(p), &u...]() mutable {
      try
        {
          p.set_value(u...);
        }
      catch (...)
        {
          p.set_exception(std::current_exception());
        }
    }};
    return t;
  };
}

template <class Task, class T, class... U>
class StateFuture;
template <class Task, class T, class... U>
class StateFutureWithCore;

// Future-like object which stores a promise-like task.
// The task does not run until get() method is called by the future.
// More functions to the function chain can be added with method 'then'.
// T is the type of the return value of the last function
// in the ".then" - chain of functions called by the future.
// U is the type of the input parameter to the first function in .then chain.
//
// An example of running a chain of continuations. The first function
// in the chain takes 2 integer input and the second returns a double.
/*
  auto fut = Lazy::future<double>(42, 58).
              then([](int i, int j){return i + j;}).
              then([](int k){ return std::sqrt(k);}).
              finalize();
  auto futRun = fut.run();
  // ... do stuff while the future is running ...
  double result = fut.get(futRun);
*/
// Note! A Future should be instantiated only by calling function
// future(...) as shown in the example.
template <class Task, class T, class... U>
class Future
{
public:
  using Type = T;
  // U is input to the first function in the call chain.
  // _task must always be initialized with makeNewThread.
  // Otherwise the thread will not be created for ".then"- function chain.
  explicit Future(const U&... u) : _task(makeNewThread(u...)) {};

  // Task must be a function which inputs a promise-like
  // object p and returns an std::thread (which calls p.set_value).
  Future(Task&& task) : _task(std::forward<Task>(task)) {};

  // Makes a new Future from *this. The new future is otherwise
  // like the current one but func is added to the end
  // of the function chain.
  template <class Func>
  auto then(Func func)
  {
    // f is a lambda function which takes
    // a promise-like object p and calls _task
    // with another promise-like object Promise(...)
    // which has the function baked in.
    // So basically this function replaces the original
    // promise with a new promise.
    // The return type of f is the return type of _task, which is std::thread.
    // At the top level p will be a SyncPromise (called from run())
    auto f = [task = std::move(_task), funcCapture = std::move(func)](auto prom) {
      return task(Promise(prom, std::move(funcCapture)));
    };
    // Return a new future with the new promise associated to the input parameter
    // of the task.
    return Future<decltype(f), T, U...>(std::move(f));
  }

  // Adds final future to the ".then" - chain.
  // The final future will contain the state data which
  // stores the result.
  // This overload is used if the core is allocated outside the sync data.
  // Notice that the core own the work thread so, where-ever it was
  // allocated, it has to outlive the future.
  auto finalize(SyncCore* core)
  {
    return StateFuture<Task, T, U...>(std::move(_task), core);
  }

  // This overload uses a version of the final future which
  // contains a core of its own.
  auto finalize()
  {
    return StateFutureWithCore<Task, T, U...>(std::move(_task));
  }

  // Gets the result of the future by waiting for the result to appear
  // in the state data.
  T get(SyncState<T>* state) const
  {
    { // Wait for the task to finish.
      // It is finished when something (the result or an exception) has been
      // stored in the state data.
      std::unique_lock lck(state->core->mtx);
      state->core->cv.wait(lck, [state]() { return state->data.index() != 0; });
    }
    state->core->thr.join();
    // Throw or return the result
    if (state->data.index() == 1)
      std::rethrow_exception(std::get<1>(state->data));
    return std::move(std::get<2>(state->data));
  }

protected:
  // Task is a lambda function that takes a promise-like object p as
  // the parameter and returns another lambda function which takes
  // another promise-like object. The return value of the task is
  // the thread that the task is running on.
  Task _task;
}; // Future


// The last future in the chain of futures.
// This version uses an external core in SyncState data,
// meaning that the future can be moved and copied.
template <class Task, class T, class... U>
class StateFuture : public Future<Task, T, U...>
{
public:
  StateFuture(Task&& t, SyncCore* core) : Future<Task, T, U...>(std::forward<Task>(t))
  {
    _state.core = core;
  };

  // Launches the thread returned by _task().
  // Input parameter to _task is a concrete SynPromise object.
  // It initiates the chain of set_value() calls
  // in promises that werecreated while making Futures in then()-functions.
  // Returns pointer to SyncState which can later be used to get the result.
  template <class CallBack = Empty>
  SyncState<T>* run(CallBack* callBack = nullptr)
  {
    _state.core->thr = this->_task(SyncPromise<T, CallBack>{&_state, callBack}); // Launch task
    return &_state;
  }

protected:
  SyncState<T> _state;
};

// The last future in the chain of futures.
// This version uses build-in core in SyncState data,
// meaning that the future can not be moved and copied.
template <class Task, class T, class... U>
class StateFutureWithCore : public StateFuture<Task, T, U...>
{
public:
  StateFutureWithCore(Task&& t) : StateFuture<Task, T, U...>(std::forward<Task>(t), &_core) {}

private:
  SyncCore _core;
};

// Prepares a future with one or more input parameters.
// Note! A Future should be instantiated only by calling future(...).
template <class T, class... U>
auto future(const U&... u)
{
  return Future<decltype(makeNewThread(std::declval<U>()...)), T, U...>(u...);
}

// Prepares a future with no input parameters.
// Note! A Future should be instantiated only by calling future(...).
template <class T>
auto future()
{
  return Future<decltype(makeNewThread(std::declval<Empty>())), T, Empty>(Empty{});
}

// Calls get() on the given futures and returns the values as a tuple.
// The set of futures and their states are also tuples.
template <class Futures, class States, std::size_t... I>
auto getResults(Futures&& futs, States&& states, std::index_sequence<I...>)
{
  return std::make_tuple(std::get<I>(futs).get(std::get<I>(states))...);
}

// Executes the given futures in parallel and returns the values of as a tuple
template <class... Futs>
auto runFutures(Futs&&... fs)
{
  auto futures = std::make_tuple(fs...);
  auto states = std::make_tuple(fs.run()...);
  return getResults(futures, states, std::make_index_sequence<sizeof...(Futs)>{});
}

// Helper function for sorting out the index sequence for the tuple of futures.
template <class FutTuple, std::size_t... I>
auto runParallelAsTuple(FutTuple&& futs, std::index_sequence<I...>)
{
  static_assert(std::tuple_size_v<FutTuple> == sizeof...(I), "Index sequence doesn't match.");
  // Stack-allocated set of core structures. One for each function.
  std::array<SyncCore, std::tuple_size_v<FutTuple>> aCore;
  return runFutures(std::get<I>(futs).finalize(&aCore[I])...);
}

// Runs the given functions in parallel in a future and returns the values as a
// tuple. The functions either must take no parameters or take StopToken* as the
// only parameter.
// The functions which take more parameters can be wrapped to a lambda.
// The parameters can be places into the capture list of the lambda.
// Example:
/*
  double myFunc(int x, int y) { return std::sqrt(x * y); }
  auto [res_dbl, res_int] =
    Lazy::runParallel([x = 10, y = 15](){ return myFunc(x, y); },
                      [x = 10, y = 20](){ return int(myFunc(x, y)); });
*/
template <class... Funcs>
auto runParallel(Funcs&&... funcs)
{
  // The functions take stop token as the only parameter
  constexpr bool bParamStopToken = (std::is_invocable<Funcs, StopToken*>() && ...);
  // The functions don't take parameters
  constexpr bool bParamNone = (std::is_invocable<Funcs>() && ...);
  // Run the functions either with or without stop token
  if constexpr (bParamStopToken)
    {
      StopToken token;
      return runParallelAsTuple(std::make_tuple(future<decltype(funcs(std::declval<StopToken*>()))>(&token).then(
                                  std::forward<Funcs>(funcs))...),
                                std::make_index_sequence<sizeof...(Funcs)>{});
    }
  else if constexpr (bParamNone)
    return runParallelAsTuple(std::make_tuple(future<decltype(funcs())>().then(std::forward<Funcs>(funcs))...),
                              std::make_index_sequence<sizeof...(Funcs)>{});
  else
    {
      static_assert(bParamStopToken || bParamNone,
                    "The functions must either take no parameters or one parameter of type StopToken*");
      return 0;
    }
}

// Callback functor which triggers notification
// that the result is ready and can be retrived with Future::get().
// Slot and index which identify the result are appended
// atomically to vector *pVec.
struct ResultWatcher
{
  std::mutex* pMtx = nullptr;
  std::condition_variable* pCv = nullptr;
  std::vector<std::pair<int, int>>* pVec = nullptr;
  std::pair<int, int> slotIndex{-1,-1}; // first = slot, second = index

  // Callback from Future.run().
  // Adds slot and index to the end of the vector atomically.
  void operator()() {
    std::unique_lock lck(*pMtx);
    pVec->push_back(slotIndex);
    slotIndex = {-1,-1}; // Mark as done. Needed for debug only.
    lck.unlock();
    pCv->notify_one();
  }
};

// Debug helper for finding out types T...
template <class... T>
void pretty_function(const T&...)
{
#if 0 // enable if iostream works in your system
    std::stringstream ss;
    ss << "pretty_function = " << __PRETTY_FUNCTION__ << "\n";
    std::cout << ss.str();
#endif
}

// Executes "y = func(x)" for each x in vector vecX in a lock-free thread pool.
// Returns a vector of y's. The maxmum number of parallel threads is MaxThreads.
// If MaxThreads <= 0, use the number of cores.
template <int MaxThreads = 0, class Vec, class Func>
auto runForAll(const Vec& vecX, Func&& func)
{
  using U = typename Vec::value_type; // input type
  // The functions take stop token as the first parameter
  constexpr bool bStopTokenAndParam = std::is_invocable_v<Func, StopToken*, U>;
  // The functions don't take parameters
  constexpr bool bOneParam = std::is_invocable_v<Func, U>;
  if constexpr (bOneParam)
    {
      using T = decltype(func(std::declval<U>()));  // output type (must be default constructible)
      constexpr bool bVoid = std::is_same_v<T, void>;  // The function return type is void?
      using NonVoidT = std::conditional_t<bVoid, char, T>;  // Replace void with something else, like char.

      std::vector<NonVoidT> vecY; // Result vector
      if constexpr (!bVoid)
        vecY.resize(vecX.size());  // Allocate only if needed (i.e non-const return type)
      std::exception_ptr pException;
      std::atomic_size_t szExceptionCount {0}; // Number of tasks that have tried to raise an exception.
      std::atomic_size_t szNumStartedTasks {0}; // Number of tasks that have either finished or running.
      auto worker = [&]() {  // Worker to run in each thread in the thread pool.
        do {
          auto szIndex = szNumStartedTasks.fetch_add(1);
          if (szIndex < vecX.size()) { // There are vecX.size() tasks to run in total
            try {
              if constexpr (bVoid)
                func(vecX[szIndex]);
              else
                vecY[szIndex] = func(vecX[szIndex]);
            }
            catch (...) {
              auto szExceptionsSoFar = szExceptionCount.fetch_add(1);
              if (szExceptionsSoFar == 0) // Only one exception will be stored
                pException = std::current_exception();
            }
          } // if
        } while (szNumStartedTasks.load() <= vecX.size());
      };  // worker

      if constexpr (MaxThreads > 0) { // Threadpool is an array of threads living in stack.
        std::array<std::thread, MaxThreads> aThreadPool;
        for (auto& thr : aThreadPool)
            thr = std::thread(worker);

        for(std::thread& thr : aThreadPool)
            if (thr.joinable())
              thr.join();
      }
      else { // Threadpool is a vector of threads living in heap.
        auto uNumThreads = std::min(std::size_t(std::thread::hardware_concurrency()), vecX.size());
        std::vector<std::thread> vecThreadPool(uNumThreads);
        for (auto& thr : vecThreadPool)
            thr = std::thread(worker);

        for(std::thread& thr : vecThreadPool)
            if (thr.joinable())
              thr.join();
      }

      // Deal with possible exception
      if (pException)
        std::rethrow_exception(pException);

      if constexpr (bVoid)
        return;
      else
        return vecY;
    } // func takes one parameter
  else if constexpr (bStopTokenAndParam)
    { // func takes a stop token and a parameter.
      StopToken token;
      auto funcWithToken = [&token, f = std::forward<Func>(func)](const U& x) { return f(&token, x); };
      return runForAll<MaxThreads>(vecX, funcWithToken);
    }
  else
    static_assert(bOneParam || bStopTokenAndParam,
                  "Function must either take one parameter or (StopToken* and a parameter).");
}

// Helper for array overload of runForAll(...)
template <class Arr, std::size_t... I, class Func>
auto runForAllInArray(const Arr& arrX, Func&& func, std::index_sequence<I...>)
{
  constexpr std::size_t N = sizeof...(I);
  using U = typename Arr::value_type; // input type

  // The functions take stop token as the first parameter
  constexpr bool bStopTokenAndParam = std::is_invocable_v<Func, StopToken*, U>;
  // The functions don't take parameters
  constexpr bool bOneParam = std::is_invocable_v<Func, U>;
  if constexpr (bOneParam)
    {
      using T = decltype(func(std::declval<U>())); // output type

      std::array<SyncCore, N> aCores;
      std::array aFutures = {future<T>(arrX[I]).then(func).finalize(&aCores[I])...};
      std::array aStates = {aFutures[I].run()...};

      std::array arrY = {aFutures[I].get(aStates[I])...};
      return arrY;
    } // func takes one parameter
  else if constexpr (bStopTokenAndParam)
    { // func takes a stop token and a parameter.
      StopToken token;
      auto funcWithToken = [&token, f = std::forward<Func>(func)](const U& x) { return f(&token, x); };
      return runForAll(arrX, funcWithToken);
    }
  else
    static_assert(bOneParam || bStopTokenAndParam,
                  "Function must either take one parameter or (StopToken* and a parameter).");
}

// Executes "y = func(x)" for each x in array arrX in a separate thread.
// There will be as many parallel threads as there are elements in the array.
// Returns an array of y's.
template <int MaxThreads = 0, class U, std::size_t N, class Func>
auto runForAll(const std::array<U, N>& arrX, Func&& func)
{
  return runForAllInArray(arrX, std::forward<Func>(func), std::make_index_sequence<N>{});
}

// Executes "y = func(x)" for each x in the initializer_list in a separate thread.
// Returns a vector of y's.
template <int MaxThreads = 0, class U, class Func>
auto runForAll(std::initializer_list<U> lstX, Func&& func)
{
  return runForAll<MaxThreads>(std::vector<U>{lstX}, std::forward<Func>(func));
}

// Telescope function:  f2(f1(f0(t)))...
/* Example:
auto sqrt1 = Lazy::nested(1, // initial value
                          [](int i) {return i + 1.0;},
                          [](double d) {return std::sqrt(d);});
*/
template <class T, class F, class... Fs>
auto nested(T t, F&& f, Fs&&... fs)
{
  if constexpr (sizeof...(Fs) == 0)
    return f(t);
  else
    return nested(f(t), std::forward<Fs>(fs)...);
}

// Accepts several functions which will be run as nested (like continuations)
// for each element in the vector x. The results are returned in another vector.
// Example
/*
  auto vecY = Lazy::runForAll({1,3,0,2,4},  [](int x){return std::sqrt(x*x*x);},
                                            [numDecimals=4](double z) {
                                            double r = 1;
                                            for (int i = 0; i < numDecimals; ++i, r *= 10);
                                            return int(r * z);});
*/
template <int MaxThreads = 0, class U, class F1, class F2, class... Funcs>
auto runForAll(const std::vector<U>& x, F1&& f1, F2&& f2, Funcs&&... funcs)
{
  auto nestedFuncs = [&f1, &f2, &funcs...](auto t) { return nested(t, f1, f2, funcs...); };
  return runForAll<MaxThreads>(x, nestedFuncs);
}

// Note: MaxThreads template parameter is ignored.
// There are always as many threads as there are elements in the array.
template <int MaxThreads = 0, class U, std::size_t N, class F1, class F2, class... Funcs>
auto runForAll(const std::array<U, N>& arrX, F1&& f1, F2&& f2, Funcs&&... funcs)
{
  auto nestedFuncs = [&f1, &f2, &funcs...](auto t) { return nested(t, f1, f2, funcs...); };
  return runForAll(arrX, nestedFuncs);
}

// Overload of the above function for initializer list input.
template <int MaxThreads = 0, class U, class F1, class F2, class... Funcs>
auto runForAll(std::initializer_list<U> lstX, F1&& f1, F2&& f2, Funcs&&... funcs)
{
  auto nestedFuncs = [&f1, &f2, &funcs...](auto t) { return nested(t, f1, f2, funcs...); };
  return runForAll<MaxThreads>(lstX, nestedFuncs);
}

// Overload of the above function for Sequence (0,1,...N-1) and multiple nested functions.
template <int MaxThreads = 0, class F1, class F2, class... Funcs>
auto runForAll(Sequence seq, F1&& f1, F2&& f2, Funcs&&... funcs)
{
  auto nestedFuncs = [&f1, &f2, &funcs...](auto t) { return nested(t, f1, f2, funcs...); };
  return runForAll<MaxThreads>(seq, nestedFuncs);
}

} // namespace Lazy

#endif // LAZY_H
