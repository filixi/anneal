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
struct NQueensSolution
    : public anneal::SolutionInterface<NQueensSolution<kN> > {
  NQueensSolution() : solution(kN) {}
  NQueensSolution(const NQueensSolution &) = default;
    
  int &operator[](const size_t pos) {
    return solution[pos];
  }
  
  int operator[](const size_t pos) const {
    return solution[pos];
  }
  
  void Swap(NQueensSolution<kN> &y) {
    assert(y.mutate_ready);
    
    NQueensSolution<kN> &x = *this;
    x[y.mutate_pos] = y.mutate_value;
    x.quality = y.quality;
    y.mutate_ready = false;
  }
  
  double Quality() const override {
    return quality;
  }
  
  std::vector<int> solution;
  double quality = 0.0;
  
  bool mutate_ready = false;
  int mutate_pos = 0;
  int mutate_value = 0;
  
  static constexpr auto n = kN;
};

// Annealing Package specialized for N queens problem
// After debugging, inheritance can be removed for better performance.
template <size_t kN, class Solution = NQueensSolution<kN> >
class NQueensProblem : public anneal::ProblemInterface<Solution> {
 public:
  using SolutionType = Solution;
  
  void NewSolution(SolutionType &solution) override {
    for (int i=0; i<kN; ++i)
      solution[i] = uniform_(random_engine_);
    solution.quality = CostFunction(solution);
  }
  
  void Mutate(SolutionType &from, SolutionType &to) override {
    to.mutate_pos = uniform_(random_engine_);
    to.mutate_value = uniform_(random_engine_);
    to.mutate_ready = true;
    CostFunctionDelta(from,to);
  }
  
  double CostFunction(SolutionType &solution) override {
    double cost = 0;
    for (int i=0; i<kN; ++i)
      for (int j=i+1; j<kN; ++j)
        if (solution[i]==solution[j] || abs(solution[i]-solution[j])==abs(i-j))
          ++cost;
    return cost;
  }
  
 private:
  void CostFunctionDelta(const SolutionType &from, SolutionType &to) {
    assert(to.mutate_ready);
    
    to.quality = from.quality;
    for (int i=0; i<kN; ++i)
      if (from[i]==from[to.mutate_pos] ||
        abs(from[i]-from[to.mutate_pos])==abs(i-to.mutate_pos))
      --to.quality;
    ++to.quality; // adjust for i==to.mutate_pos
    
    for (int i=0; i<kN; ++i)
      if (from[i]==to.mutate_value ||
        abs(from[i]-to.mutate_value)==abs(i-to.mutate_pos))
      ++to.quality;
      
    if (from[to.mutate_pos] == to.mutate_value)
      --to.quality;
  }

  std::mt19937 random_engine_{std::random_device{}()};
  std::uniform_int_distribution<int> uniform_{0,kN-1};
};

} // namespace nqueens

#endif // _ANNEAL_N_QUEENS_H_

