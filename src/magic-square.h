#ifndef _ANNEAL_MAGIC_SQUARE_H_
#define _ANNEAL_MAGIC_SQUARE_H_

#include "cassert"

#include <algorithm>
#include <iostream>
#include <memory>

#include "simulated-annealing.h"

namespace magicsquare {

template <size_t kN, class Solution>
class MagicSquareProblem;
template <class Problem, class Solution>
class MutatorManager;

template <size_t kN>
class MagicSquareSolution
    : public anneal::SolutionInterface<MagicSquareSolution<kN> > {
 public:
  friend class MagicSquareProblem<kN, MagicSquareSolution<kN> >;
  template <class Problem, class Solution>
  friend class SwapMutator;
  
  MagicSquareSolution() : sum_row_(kN, 0.0), sum_col_(kN, 0.0) {
    matrix_.resize(kN);
    for (auto &line : matrix_)
      line.resize(kN);
  }
  
  MagicSquareSolution(const MagicSquareSolution &) = default;
  MagicSquareSolution(MagicSquareSolution &&) = default;
  
  MagicSquareSolution &operator=(const MagicSquareSolution &) = default;
  MagicSquareSolution &operator=(MagicSquareSolution &&) = default;
  
  std::vector<int> &operator[](const size_t pos) {
    return matrix_[pos];
  }
  const std::vector<int> &operator[](const size_t pos) const {
    return matrix_[pos];
  }
  
  double Quality() const override {
    return quality_;
  }
  
  double &Quality() {
    return quality_;
  }
  
  static const auto n = kN;
  
 private:
  double quality_ = 0.0;
  std::vector<std::vector<int> > matrix_;
  
  std::vector<double> sum_row_;
  std::vector<double> sum_col_;
  double sum_main_diagonal_ = 0.0;
  double sum_secondary_diagonal_ = 0.0;
};

template <size_t kN, class Solution = MagicSquareSolution<kN> >
class MagicSquareProblem : public anneal::ProblemInterface<Solution> {
 public:
  using SolutionType = MagicSquareSolution<kN>;
  using MutatorManagerType = MutatorManager<MagicSquareProblem, SolutionType>;
  
  virtual SolutionType NewSolution() override {
    SolutionType solution;
    std::unique_ptr<int[]> shuffle(new int[kN*kN]);
    
    for (size_t i=0; i<kN*kN; ++i)
      shuffle[i] = i+1;
    std::random_shuffle(shuffle.get(), shuffle.get()+kN*kN);
    
    for (size_t i=0; i<kN*kN; ++i)
      solution[i/kN][i%kN] = shuffle[i];
      
    CostFunction(solution);
    return solution;
  }
  
  double CostFunction(SolutionType &x) override {
    double quality = 0.0;
    
    // every row
    for (size_t i=0; i<kN; ++i) {
      double sum = 0.0;
      for (size_t j=0; j<kN; ++j)
        sum += x[i][j];
      x.sum_row_[i] = sum;
      quality += abs(sum-kM);
    }
    
    // every col
    for (size_t i=0; i<kN; ++i) {
      double sum = 0.0;
      for (size_t j=0; j<kN; ++j)
        sum += x[j][i];
      x.sum_col_[i] = sum;
      quality += abs(sum-kM);
    }
    
    // main diagonals
    double sum = 0.0;
    for (size_t i=0; i<kN; ++i)
      sum += x[i][i];
    x.sum_main_diagonal_ = sum;
    quality += abs(sum-kM);
    
    // secondary diagonals
    sum = 0.0;
    for (size_t i=0; i<kN; ++i)
      sum += x[i][kN-i-1];
    x.sum_secondary_diagonal_ = sum;
    quality += abs(sum-kM);
    
    return x.Quality() = quality;
  }
  
  static constexpr auto n = kN;
  static constexpr double kM = (kN*(kN*kN+1))/2;
};

template <class Problem, class Solution>
class SwapMutator {
 public:
  SwapMutator() : sum_row_(kN), sum_col_(kN) {}
  
  template <class RandomEngine, class Distribution>
  void Premutate(RandomEngine &e, Distribution &d) {
    mutate_pos_b_ = mutate_pos_a_ = d(e);
    while (mutate_pos_b_ == mutate_pos_a_)
      mutate_pos_b_ = d(e);
  }
 
  template <class RandomEngine, class Distribution>
  void MutateFrom(const Solution &from, RandomEngine &e, Distribution &d) {
    
  }
  
  double DeltaQuality(const Problem &problem, const Solution &from) {
    const size_t row_a = mutate_pos_a_/kN;
    const size_t col_a = mutate_pos_a_%kN;
    const size_t row_b = mutate_pos_b_/kN;
    const size_t col_b = mutate_pos_b_%kN;
    
    sum_row_[row_a] = from.sum_row_[row_a];
    sum_col_[col_a] = from.sum_col_[col_a];
    sum_row_[row_b] = from.sum_row_[row_b];
    sum_col_[col_b] = from.sum_col_[col_b];
    sum_main_diagonal_ = from.sum_main_diagonal_;
    sum_secondary_diagonal_ = from.sum_secondary_diagonal_;
    
    sum_row_[row_a] += -from[row_a][col_a] + from[row_b][col_b];
    sum_col_[col_a] += -from[row_a][col_a] + from[row_b][col_b];
    sum_row_[row_b] += -from[row_b][col_b] + from[row_a][col_a];
    sum_col_[col_b] += -from[row_b][col_b] + from[row_a][col_a];
    
    if (row_a == col_a)
      sum_main_diagonal_ += -from[row_a][col_a] + from[row_b][col_b];
    if (row_a == (kN-col_a-1))
      sum_secondary_diagonal_ += -from[row_a][col_a] + from[row_b][col_b];
    if (row_b == col_b)
      sum_main_diagonal_ += -from[row_b][col_b] + from[row_a][col_a];
    if (row_b == (kN-col_b-1))
      sum_secondary_diagonal_ += -from[row_b][col_b] + from[row_a][col_a];
      
    delta_quality_ =
      abs(sum_row_[row_a]-kM) - abs(from.sum_row_[row_a]-kM) +
      abs(sum_col_[col_a]-kM) - abs(from.sum_col_[col_a]-kM) +
      abs(sum_row_[row_b]-kM) - abs(from.sum_row_[row_b]-kM) +
      abs(sum_col_[col_b]-kM) - abs(from.sum_col_[col_b]-kM) +
      abs(sum_main_diagonal_-kM) - abs(from.sum_main_diagonal_-kM) +
      abs(sum_secondary_diagonal_-kM) -
      abs(from.sum_secondary_diagonal_-kM);
    return delta_quality_;
  }
  
  void Mutate(Solution &x) const {
    const size_t row_a = mutate_pos_a_/kN;
    const size_t col_a = mutate_pos_a_%kN;
    const size_t row_b = mutate_pos_b_/kN;
    const size_t col_b = mutate_pos_b_%kN;
    
    x.sum_row_[row_a] = sum_row_[row_a];
    x.sum_col_[col_a] = sum_col_[col_a];
    x.sum_row_[row_b] = sum_row_[row_b];
    x.sum_col_[col_b] = sum_col_[col_b];
    x.sum_main_diagonal_ = sum_main_diagonal_;
    x.sum_secondary_diagonal_ = sum_secondary_diagonal_;
    
    std::swap(x[row_a][col_a], x[row_b][col_b]);
    x.quality_ += delta_quality_;
  }
  
  static constexpr int kN = Problem::n;
  static constexpr double kM = Problem::kM;
  
 private:
  double delta_quality_ = 0.0;
 
  std::vector<double> sum_row_;
  std::vector<double> sum_col_;
  double sum_main_diagonal_ = 0.0;
  double sum_secondary_diagonal_ = 0.0;
  
  size_t mutate_pos_a_ = 0;
  size_t mutate_pos_b_ = 0;
};

template <class Problem, class Solution>
class MutatorManager
    : public anneal::MutatorManagerInterface<Problem, Solution> {
 public:
  void Premutate() override {
    swap_mutator_.Premutate(random_engine_, uniform_);
  }
  
  void MutateFrom(const Solution &from) {
    //swap_mutator_.MutateFrom(from, random_engine_, uniform_);
    mutate_ready_ = true;
  }
  
  double DeltaQuality(const Problem &problem, const Solution &x) {
    assert(mutate_ready_);
    return swap_mutator_.DeltaQuality(problem,x);
  }
  
  void Mutate(Solution &x) {
    assert(mutate_ready_);
    swap_mutator_.Mutate(x);
    mutate_ready_ = false;
  }
  
 private:
  SwapMutator<Problem, Solution> swap_mutator_;
  bool mutate_ready_ = false;
  
  std::mt19937 random_engine_{std::random_device{}()};
  std::uniform_int_distribution<int> uniform_{0,Problem::n*Problem::n-1};
};

} // magicsquare


#endif // _ANNEAL_MAGIC_SQUARE_H_

