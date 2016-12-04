#ifndef _ANNEAL_SIMULATED_ANNEALING_H_
#define _ANNEAL_SIMULATED_ANNEALING_H_

#include <cassert>
#include <cmath>

#include <random>

namespace anneal {

class SimulatedAnnealing {
 public:
  template <class TemperaturePolicy, class Problem>
  auto operator()(TemperaturePolicy &&temperature, Problem &&problem) {
    using ProblemType = typename std::decay<Problem>::type;
    typename ProblemType::MutatorManagerType mutator_manager;
    auto solution = std::forward<Problem>(problem).NewSolution();
    
    while (!std::forward<TemperaturePolicy>(temperature).Quit(solution)) {
      mutator_manager.MutateFrom(solution);
      const auto accept = temperature.Accept(mutator_manager.DeltaQuality(
          std::forward<Problem>(problem), solution));

      if ( accept )
        mutator_manager.Mutate(solution);
    }
    
    return solution;
  }
  
  template <class Problem>
  void Debug(Problem &&problem) {
    using ProblemType = typename std::decay<Problem>::type;
    typename ProblemType::MutatorManagerType mutator_manager;
    auto solution = std::forward<Problem>(problem).NewSolution();
    
    for (int i=0; i<1000; ++i) {
      const auto old_quality = solution.Quality();
      mutator_manager.MutateFrom(solution);
      const auto delta_quality = mutator_manager.DeltaQuality(
          std::forward<Problem>(problem), solution);
      mutator_manager.Mutate(solution);
      assert(old_quality+delta_quality == solution.Quality());
    }
  }
};

template <class Problem, class Solution>
class MutatorManagerInterface {
 public:
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
  TemperatureBasic() = default;
  TemperatureBasic(const int step_interval, const double initial_temperature)
      : kStepInterval(step_interval), temperature_(initial_temperature) {}
  TemperatureBasic(const TemperatureBasic &) = default;
  TemperatureBasic &operator=(const TemperatureBasic &) = default;
  
  bool Quit(const Solution &x) {
    ++times_iteration_;
    if (times_iteration_%kStepInterval==0)
      temperature_*=0.99;
    
    if (x.Quality()==0)
      return true;
    else if (temperature_<=0.00001)
      return true;
    else
      return false;
  }
  
  bool Accept(const double delta) override {
    if (delta<0 || random() <= ::exp(-delta/temperature_))
      return true;
    else
      return false;
  }
  
 private:
  const int32_t kStepInterval = 10000;
  double temperature_ = 2.0;
  int32_t times_iteration_ = 0;
  
  auto random() {
    return uniform_(random_engine_);
  }

  std::mt19937 random_engine_{std::random_device{}()};
  std::uniform_real_distribution<float> uniform_{0,1};
};

} // namespace anneal

#endif // _ANNEAL_SIMULATED_ANNEALING_H_

