#ifndef _ANNEAL_SIMULATED_ANNEALING_H_
#define _ANNEAL_SIMULATED_ANNEALING_H_

#include <cassert>
#include <cmath>

#include <random>

namespace anneal {

class SimulatedAnnealing {
 public:
  template <class TemperaturePolicy, class ProblemPackage>
  auto operator()(TemperaturePolicy &&temperature, ProblemPackage &&problem) {
    using SolutionType =
      typename std::decay<ProblemPackage>::type::SolutionType;
    
    SolutionType solution, alter;
    std::forward<ProblemPackage>(problem).NewSolution(solution);
    
    while (!std::forward<TemperaturePolicy>(temperature).Quit(solution)) {
      std::forward<ProblemPackage>(problem).Mutate(solution, alter);
      if (std::forward<TemperaturePolicy>(temperature).Accept(solution, alter))
        solution.Swap(alter);
    }
    
    return solution;
  }
  
  template <class ProblemPackage>
  void Debug(ProblemPackage &&problem) {
    using SolutionType =
      typename std::decay<ProblemPackage>::type::SolutionType;
    SolutionType solution, alter;
    std::forward<ProblemPackage>(problem).NewSolution(solution);
    
    for (int i=0; i<1000; ++i) {
      double quality_solution = solution.Quality();
      std::forward<ProblemPackage>(problem).Mutate(solution, alter);
      assert(solution.Quality() == quality_solution);
      
      double quality_alter = alter.Quality();
      solution.Swap(alter);
      assert(solution.Quality() == quality_alter);
      auto temp = solution;
      assert(std::forward<ProblemPackage>(problem).CostFunction(temp) ==
        quality_alter);
    }
  }
};

template <class SolutionType>
class SolutionInterface {
 public:
  virtual double Quality() const = 0;
  virtual void Swap(SolutionType &) = 0;
  virtual ~SolutionInterface() = default;
};

template <class SolutionType>
class ProblemInterface {
 public:
  virtual void NewSolution(SolutionType &) = 0;
  virtual void Mutate(SolutionType &from, SolutionType &to) = 0;
  virtual double CostFunction(SolutionType &) = 0;
  virtual ~ProblemInterface() = default;
};

template <class SolutionType>
class TemperatureInterface {
 public:
  virtual bool Quit(const SolutionType &) = 0;
  virtual bool Accept(const SolutionType &from, const SolutionType &to) = 0;
  virtual ~TemperatureInterface() = default;
};

template <class SolutionType>
class TemperatureBasic : public TemperatureInterface<SolutionType> {
 public:
  TemperatureBasic() = default;
  TemperatureBasic(const int step_interval, const double initial_temperature)
      : kStepInterval(step_interval), temperature_(initial_temperature) {}
  TemperatureBasic(const TemperatureBasic &) = default;
  TemperatureBasic &operator=(const TemperatureBasic &) = default;
  
  bool Quit(const SolutionType &x) {
    ++times_iteration_;
    if (times_iteration_%kStepInterval==0)
      temperature_*=0.99;
    
    if (x.Quality()==0)
      return true;
    else if (temperature_<=0.01)
      return true;
    else
      return false;
  }
  
  bool Accept(const SolutionType &from, const SolutionType &to) {
    if (to.Quality() <= from.Quality() ||
      random() <= ::exp((from.Quality()-to.Quality())/temperature_))
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

