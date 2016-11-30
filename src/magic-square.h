#ifndef _ANNEAL_MAGIC_SQUARE_H_
#define _ANNEAL_MAGIC_SQUARE_H_

#include "cassert"

#include <algorithm>
#include <memory>

#include "simulated-annealing.h"

namespace magicsquare {

template <size_t kN, class Solution>
class MagicSquareProblem;

template <size_t kN>
class MagicSquareSolution
    : public anneal::SolutionInterface<MagicSquareSolution<kN> > {
 public:
  friend class MagicSquareProblem<kN, MagicSquareSolution<kN> >;
  
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
  
  void Swap(MagicSquareSolution &x) override {
    assert(x.mutate_ready_);
    const size_t row_a = x.mutate_pos_a_/kN;
    const size_t col_a = x.mutate_pos_a_%kN;
    const size_t row_b = x.mutate_pos_b_/kN;
    const size_t col_b = x.mutate_pos_b_%kN;
    
    sum_row_[row_a] = x.sum_row_[row_a];
    sum_col_[col_a] = x.sum_col_[col_a];
    sum_row_[row_b] = x.sum_row_[row_b];
    sum_col_[col_b] = x.sum_col_[col_b];
    sum_main_diagonal_ = x.sum_main_diagonal_;
    sum_secondary_diagonal_ = x.sum_secondary_diagonal_;
    
    std::swap(matrix_[row_a][col_a], matrix_[row_b][col_b]);
    quality_ = x.quality_;
    x.mutate_ready_ = false;
  }
  
  static const size_t n = kN;
 private:
  double quality_ = 0.0;
  std::vector<std::vector<int> > matrix_;
  
  std::vector<double> sum_row_;
  std::vector<double> sum_col_;
  double sum_main_diagonal_ = 0.0;
  double sum_secondary_diagonal_ = 0.0;
  
  bool mutate_ready_ = false;
  size_t mutate_pos_a_ = 0;
  size_t mutate_pos_b_ = 0;
};

template <size_t kN, class Solution = MagicSquareSolution<kN> >
class MagicSquareProblem : public anneal::ProblemInterface<Solution> {
 public:
  using SolutionType = MagicSquareSolution<kN>;
  
  virtual void NewSolution(SolutionType &x) override {
    std::unique_ptr<int[]> shuffle(new int[kN*kN]);
    
    for (size_t i=0; i<kN*kN; ++i)
      shuffle[i] = i+1;
    std::random_shuffle(shuffle.get(), shuffle.get()+kN*kN);
    
    for (size_t i=0; i<kN*kN; ++i)
      x[i/kN][i%kN] = shuffle[i];
      
    CostFunction(x);
  }
  
  void Mutate(SolutionType &from, SolutionType &to) override {
    to.mutate_pos_b_ = to.mutate_pos_a_ = Random();
    while (to.mutate_pos_b_ == to.mutate_pos_a_)
      to.mutate_pos_b_ = Random();
    CostFunctionDelta(from,to);
  }
  
  double CostFunction(SolutionType &x) override {
    double quality = 0.0;
    
    // every row
    for (size_t i=0; i<kN; ++i) {
      double sum = 0.0;
      for (size_t j=0; j<kN; ++j)
        sum += x[i][j];
      x.sum_row_[i] = sum;
      quality += abs(sum-kM_);
    }
    
    // every col
    for (size_t i=0; i<kN; ++i) {
      double sum = 0.0;
      for (size_t j=0; j<kN; ++j)
        sum += x[j][i];
      x.sum_col_[i] = sum;
      quality += abs(sum-kM_);
    }
    
    // main diagonals
    double sum = 0.0;
    for (size_t i=0; i<kN; ++i)
      sum += x[i][i];
    x.sum_main_diagonal_ = sum;
    quality += abs(sum-kM_);
    
    // secondary diagonals
    sum = 0.0;
    for (size_t i=0; i<kN; ++i)
      sum += x[i][kN-i-1];
    x.sum_secondary_diagonal_ = sum;
    quality += abs(sum-kM_);
    
    return x.quality_ = quality;
  }

 private:
  double CostFunctionDelta(const SolutionType &from, SolutionType &to) {
    const size_t row_a = to.mutate_pos_a_/kN;
    const size_t col_a = to.mutate_pos_a_%kN;
    const size_t row_b = to.mutate_pos_b_/kN;
    const size_t col_b = to.mutate_pos_b_%kN;
    
    to.sum_row_[row_a] = from.sum_row_[row_a];
    to.sum_col_[col_a] = from.sum_col_[col_a];
    to.sum_row_[row_b] = from.sum_row_[row_b];
    to.sum_col_[col_b] = from.sum_col_[col_b];
    to.sum_main_diagonal_ = from.sum_main_diagonal_;
    to.sum_secondary_diagonal_ = from.sum_secondary_diagonal_;
    
    to.sum_row_[row_a] += -from[row_a][col_a] + from[row_b][col_b];
    to.sum_col_[col_a] += -from[row_a][col_a] + from[row_b][col_b];
    to.sum_row_[row_b] += -from[row_b][col_b] + from[row_a][col_a];
    to.sum_col_[col_b] += -from[row_b][col_b] + from[row_a][col_a];
    
    if (row_a == col_a)
      to.sum_main_diagonal_ += -from[row_a][col_a] + from[row_b][col_b];
    if (row_a == (kN-col_a-1))
      to.sum_secondary_diagonal_ += -from[row_a][col_a] + from[row_b][col_b];
    if (row_b == col_b)
      to.sum_main_diagonal_ += -from[row_b][col_b] + from[row_a][col_a];
    if (row_b == (kN-col_b-1))
      to.sum_secondary_diagonal_ += -from[row_b][col_b] + from[row_a][col_a];
      
    to.quality_ = from.quality_ +
      abs(to.sum_row_[row_a]-kM_) - abs(from.sum_row_[row_a]-kM_) +
      abs(to.sum_col_[col_a]-kM_) - abs(from.sum_col_[col_a]-kM_) +
      abs(to.sum_row_[row_b]-kM_) - abs(from.sum_row_[row_b]-kM_) +
      abs(to.sum_col_[col_b]-kM_) - abs(from.sum_col_[col_b]-kM_) +
      abs(to.sum_main_diagonal_-kM_) - abs(from.sum_main_diagonal_-kM_) +
      abs(to.sum_secondary_diagonal_-kM_) -
      abs(from.sum_secondary_diagonal_-kM_);
    to.mutate_ready_ = true;
    return to.quality_;
  }

  auto Random() {
    return uniform_(random_engine_);
  }
  
  static constexpr double kM_ = (kN*(kN*kN+1))/2;
 
  std::mt19937 random_engine_{std::random_device{}()};
  std::uniform_int_distribution<int> uniform_{0,kN*kN-1};
};

} // magicsquare


#endif // _ANNEAL_MAGIC_SQUARE_H_

