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

    std::mt19937 random_engine(std::random_device{}());
    std::uniform_real_distribution<float> uniform(0,1);
    
    while (!std::forward<TemperaturePolicy>(temperature).Quit(solution)) {
      mutator_manager.Premutate();
      mutator_manager.MutateFrom(solution);
      const auto accept = temperature.Accept(mutator_manager.DeltaQuality(
          std::forward<Problem>(problem), solution), random_engine, uniform);

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
  
  std::mt19937 random_engine(std::random_device{}());
  std::uniform_real_distribution<float> uniform(0,1);
  
  while (!quit_.load()) {
    mutator_manager.Premutate();
    // Thread may see half mutated solution in reading section, which will not
    // be a problem only if seeing half mutated solution could crash the
    // program.

    // cirtical section for reading
    const uint32_t old_version = solution_version_.load();
    mutator_manager.MutateFrom(solution);
    const auto accept = temperature.Accept(mutator_manager.DeltaQuality(
        std::forward<Problem>(problem), solution), random_engine, uniform);
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
class TemperatureBasic {
 public:
  TemperatureBasic() = default;
  TemperatureBasic(const int step_interval, const double initial_temperature)
      : kStepInterval(step_interval), temperature_(initial_temperature) {}
  TemperatureBasic(const TemperatureBasic &) = default;
  TemperatureBasic &operator=(const TemperatureBasic &) = default;
  
  bool Quit(const Solution &x) {
    ++times_iteration_;
    if (times_iteration_.load()%kStepInterval==0)
      --temperature_;
    
    return (x.Quality()==0) || (temperature_.load()<=0);
  }
  
  template <class RandomEngine, class Distribution>
  bool Accept(const double delta, RandomEngine &e, Distribution &d) {
    return delta<0 || d(e) <= ::exp(-delta*1000.0/temperature_.load());
  }
  
 private:
  const int32_t kStepInterval = 10000;
  std::atomic<int32_t> temperature_{2000};
  std::atomic<int32_t> times_iteration_{0};
};

} // namespace anneal

#endif // _ANNEAL_SIMULATED_ANNEALING_H_

