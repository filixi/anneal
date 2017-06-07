#ifndef _ANNEAL_N_QUEENS_H_
#define _ANNEAL_N_QUEENS_H_

#include <cassert>
#include <cmath>

#include <array>
#include <iostream>
#include <memory>
#include <random>

#include "simulated-annealing.h"

namespace nqueens {

// N Queens problem solution holder
template <size_t kN>
class NQueensSolution
    : public anneal::SolutionInterface<NQueensSolution<kN> > {
 public:
  NQueensSolution() : solution_(kN) {}
  NQueensSolution(const NQueensSolution &) = default;
    
  int &operator[](const size_t pos) {
    return solution_[pos];
  }
  
  int operator[](const size_t pos) const {
    return solution_[pos];
  }
  
  double Quality() const override {
    return quality_;
  }
  
  double &Quality() {
    return quality_;
  }
  
  static constexpr auto n = kN;
  
 private:
  std::vector<int> solution_;
  double quality_ = 0.0;
};

template <class Solution, class Problem>
class MutatorManager;

// Annealing Package specialized for N queens problem
// After debugging, inheritance can be removed for better performance.
template <size_t kN, class Solution = NQueensSolution<kN> >
class NQueensProblem : public anneal::ProblemInterface<Solution> {
 public:
  using SolutionType = Solution;
  using MutatorManagerType = MutatorManager<NQueensProblem, Solution>;
  
  Solution NewSolution() override {
    Solution solution;
    std::mt19937 random_engine{std::random_device{}()};
    std::uniform_int_distribution<int> uniform(0,kN-1);
    
    for (int i=0; i<kN; ++i)
      solution[i] = uniform(random_engine);
    solution.Quality() = CostFunction(solution);
    
    return solution;
  }
  
  double CostFunction(SolutionType &solution) override {
    double cost = 0;
    for (int i=0; i<kN; ++i)
      for (int j=i+1; j<kN; ++j)
        if (solution[i]==solution[j] || abs(solution[i]-solution[j])==abs(i-j))
          ++cost;
    return cost;
  }
  
  static constexpr auto n = kN;
};

template <class Solution, class Problem>
class AlterMutator {
 public:
  template <class Distribution, class Engine>
  void Premutate(Distribution &d, Engine &e) {
    mutate_pos_ = d(e);
    mutate_value_ = d(e);
  }
 
  template <class Distribution, class Engine>
  void MutateFrom(const Solution &from, Distribution &d, Engine &e) {
    //mutate_pos_ = d(e);
    //mutate_value_ = d(e);
  }
  
  double DeltaQuality(const Problem &problem, const Solution &from) {
    delta_quality_ = 0.0;
    
    for (int i=0; i<problem.n; ++i)
      if (from[i]==from[mutate_pos_] ||
          abs(from[i]-from[mutate_pos_])==abs(i-mutate_pos_))
      --delta_quality_;
    ++delta_quality_; // adjust for i==to.mutate_pos
    
    for (int i=0; i<problem.n; ++i)
      if (from[i]==mutate_value_ ||
          abs(from[i]-mutate_value_)==abs(i-mutate_pos_))
      ++delta_quality_;
      
    if (from[mutate_pos_] == mutate_value_)
      --delta_quality_;
    
    return delta_quality_;
  }
  
  void Mutate(Solution &from) {
    from[mutate_pos_] = mutate_value_;
    from.Quality() += delta_quality_;
  }
  
 private:
  int mutate_pos_ = 0;
  int mutate_value_ = 0;
  
  double delta_quality_ = 0.0;
};

template <class Problem, class Solution>
class MutatorManager :
    public anneal::MutatorManagerInterface<Problem, Solution> {
 public:
  void Premutate() override {
    mutator_.Premutate(uniform_, random_engine_);
    mutate_ready_ = true;
  }
 
  void MutateFrom(const Solution &from) override {
    //mutator.MutateFrom(from, uniform_, random_engine_);
    //mutate_ready_ = true;
  }
  
  double DeltaQuality(const Problem &problem, const Solution &x) override {
    assert(mutate_ready_);
    return mutator_.DeltaQuality(problem, x);
  }
  
  void Mutate(Solution &x) override {
    assert(mutate_ready_);
    mutator_.Mutate(x);
    mutate_ready_ = false;
  }
  
 private:
  bool mutate_ready_ = false;
  AlterMutator<Solution, Problem> mutator_;
  
  auto Random() {
    return uniform_(random_engine_);
  }
  
  std::uniform_int_distribution<int> uniform_{0,Solution::n-1};
  std::mt19937 random_engine_{std::random_device{}()};
};

} // namespace nqueens

#endif // _ANNEAL_N_QUEENS_H_
