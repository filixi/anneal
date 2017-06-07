#ifndef _ANNEAL_SIMULATED_ANNEALING_H_
#define _ANNEAL_SIMULATED_ANNEALING_H_

#include <cassert>
#include <cmath>

#include <atomic>
#include <list>
#include <random>
#include <thread>

namespace anneal {

class SimulatedAnnealing {
 public:
  template <class TemperaturePolicy, class Problem>
  auto operator()(TemperaturePolicy &&temperature, Problem &&problem) {
    using ProblemType = typename std::decay<Problem>::type;
    typename ProblemType::MutatorManagerType mutator_manager;
    auto solution = std::forward<Problem>(problem).NewSolution();

    while (!std::forward<TemperaturePolicy>(temperature).Quit(solution)) {
      mutator_manager.Premutate();
      mutator_manager.MutateFrom(solution);
      const auto accept = temperature.Accept(mutator_manager.DeltaQuality(
          std::forward<Problem>(problem), solution));

      if ( accept )
        mutator_manager.Mutate(solution);
    }

    return solution;
  }
  
  template <class TemperaturePolicy, class Problem>
  auto MultiThread(TemperaturePolicy &&temperature, Problem &&problem) {
    auto solution = std::forward<Problem>(problem).NewSolution();
    using SolutionType = typename std::decay<decltype(solution)>::type;
    
    std::list<std::thread> threads;
    for (size_t i=0; i<std::thread::hardware_concurrency(); ++i) {
      threads.emplace_back(
          [this, &temperature, &problem, &solution]() -> void {
            Worker(std::forward<TemperaturePolicy>(temperature),
                   std::forward<Problem>(problem),
                   solution);
          });
    }
    
    for (auto &thread : threads)
      thread.join();

    return solution;
  }
  
  template <class Problem>
  void Debug(Problem &&problem) {
    using ProblemType = typename std::decay<Problem>::type;
    typename ProblemType::MutatorManagerType mutator_manager;
    auto solution = std::forward<Problem>(problem).NewSolution();
    
    for (int i=0; i<1000; ++i) {
      const auto old_quality = solution.Quality();
      mutator_manager.Premutate();
      mutator_manager.MutateFrom(solution);
      const auto delta_quality = mutator_manager.DeltaQuality(
          std::forward<Problem>(problem), solution);
      mutator_manager.Mutate(solution);

      assert(old_quality+delta_quality ==
        std::forward<Problem>(problem).CostFunction(solution));
    }
  }
  
 private:
  template <class TemperaturePolicy, class Problem, class Solution>
  void Worker(TemperaturePolicy &&temperature, Problem &&problem,
              Solution &solution);

  std::atomic_flag lck_ = ATOMIC_FLAG_INIT;
  std::atomic_flag copy_lck_ = ATOMIC_FLAG_INIT;
  std::atomic<uint32_t> solution_version_{0};
  std::atomic<bool> quit_{false};
  
  static constexpr uint32_t kVersionCapacity = 4200000000;
};

template <class TemperaturePolicy, class Problem, class Solution>
void SimulatedAnnealing::Worker(TemperaturePolicy &&temperature,
                                Problem &&problem,
                                Solution &solution) {
  using ProblemType = typename std::decay<Problem>::type;
  typename ProblemType::MutatorManagerType mutator_manager;

  while (!quit_.load()) {
    mutator_manager.Premutate();
    // Thread may see half mutated solution in reading section, which will not
    // be a problem only if seeing half mutated solution could crash the
    // program.

    // cirtical section for reading
    const uint32_t old_version = solution_version_.load();
    mutator_manager.MutateFrom(solution);
    const auto accept = temperature.Accept(mutator_manager.DeltaQuality(
        std::forward<Problem>(problem), solution));
    // critical section end
    
    if (accept && !quit_.load()) {
      const uint32_t new_version = (old_version+1)%kVersionCapacity;
      while (lck_.test_and_set()) {}
      if (old_version == solution_version_.load()) {
        // critical section for writing
        mutator_manager.Mutate(solution);
        solution_version_.store(new_version);
        // critical section end
      }
      lck_.clear();
    }
    
    quit_.store(quit_.load() || temperature.Quit(solution));
  }
}

template <class Problem, class Solution>
class MutatorManagerInterface {
 public:
  virtual void Premutate() = 0;
  virtual void MutateFrom(const Solution &) = 0;
  virtual double DeltaQuality(const Problem &, const Solution &) = 0;
  virtual void Mutate(Solution &) = 0;
  virtual ~MutatorManagerInterface() = default;
};

template <class Solution>
class SolutionInterface {
 public:
  virtual double Quality() const = 0;
  virtual ~SolutionInterface() = default;
};

template <class Solution>
class ProblemInterface {
 public:
  virtual Solution NewSolution() = 0;
  virtual double CostFunction(Solution &) = 0;
  virtual ~ProblemInterface() = default;
};

template <class Solution>
class TemperatureInterface {
 public:
  virtual bool Quit(const Solution &) = 0;
  virtual bool Accept(const double delta) = 0;
  virtual ~TemperatureInterface() = default;
};

template <class Solution>
class TemperatureBasic : public TemperatureInterface<Solution> {
 public:
  template <class AnnealingDuration, class ClimbingDuration>
  TemperatureBasic(AnnealingDuration d1, ClimbingDuration d2) {
    thread_timer_ = std::thread(&TemperatureBasic::Timer, this,
      std::chrono::steady_clock::now() + d1,
      std::chrono::steady_clock::now() + d2);
  }
  
  TemperatureBasic(const TemperatureBasic &) = delete;
  TemperatureBasic &operator=(const TemperatureBasic &) = delete;
  
  ~TemperatureBasic() {
    thread_timer_.join();
  }
  
  bool Quit(const Solution &x) override {
    return quit_.load();
  }

  bool Accept(const double delta) override {
    static thread_local std::mt19937 e{std::random_device{}()};
    static thread_local std::uniform_real_distribution<double> d(0, 1);
    
    if (delta <= 1e-6)
      return true;
    return d(e) <= ::exp(-delta/rate_.load());
  }
  
  void Timer(std::chrono::steady_clock::time_point dead_line1,
             std::chrono::steady_clock::time_point dead_line2) {
    static const double T0 = rate_.load();
    static const double t =
        (dead_line1 - std::chrono::steady_clock::now()).count();

    for (;;) {
      auto duration = dead_line1 - std::chrono::steady_clock::now();
      if (duration.count() <= 0)
        break;
      
      const double x = t - duration.count();
      
      rate_.store(T0 - sqrt((t*t-(x-t)*(x-t))*T0*T0/t/t) + 0.01);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    rate_.store(0);
    for (;;) {
      auto duration = dead_line2 - std::chrono::steady_clock::now();
      if (duration.count() <= 0)
        break;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    quit_.store(true);
  }
  
 private:
  std::atomic<double> rate_{10};
  std::atomic<bool> quit_{false};
  std::thread thread_timer_;
  
  static thread_local std::mt19937 e;
  static thread_local std::uniform_real_distribution<double> d;
};

} // namespace anneal

#endif // _ANNEAL_SIMULATED_ANNEALING_H_
