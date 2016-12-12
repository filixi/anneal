#ifndef _ANNEAL_TSP_H_
#define _ANNEAL_TSP_H_

#include <cassert>
#include <cstdio>

#include <iostream>
#include <random>
#include <utility>

#include "simulated-annealing.h"

namespace tsp {

template <size_t kN, class Solution>
class TspProblem;

template <size_t kN>
class TspSolution : public anneal::SolutionInterface<TspSolution<kN> > {
 public:
  friend class TspProblem<kN, TspSolution<kN> >;
  
  TspSolution() : tour_(kN) {}
  
  TspSolution(const TspSolution &) = default;
  TspSolution(TspSolution &&) = default;
  
  TspSolution &operator=(const TspSolution &) = default;
  TspSolution &operator=(TspSolution &&) = default;
  
  int &operator[](const size_t pos) {
    return tour_[pos]; 
  }
  int operator[](const size_t pos) const {
    return tour_[pos]; 
  }
  
  int &At(const int pos) {
    return tour_[(pos+kN)%kN]; 
  }
  int At(const int pos) const {
    return tour_[(pos+kN)%kN]; 
  }
  
  double &Quality() {
    return quality_;
  }
  double Quality() const override {
    return quality_;
  }
  
  static constexpr auto n = kN;
 private:
  std::vector<int> tour_;
  double quality_;
};

template <class Problem, class Solution>
class InsertMutator;
template <class Problem, class Solution>
class InverseMutator;
template <class Problem, class Solution>
class MutatorManager;

template <size_t kN, class Solution = TspSolution<kN> >
class TspProblem : public anneal::ProblemInterface<Solution> {
 public:
  friend class InsertMutator<TspProblem, Solution>;
  friend class InverseMutator<TspProblem, Solution>;
  
  using SolutionType = Solution;
  using MutatorManagerType = MutatorManager<TspProblem, Solution>;
 
  template <class Adjacency>
  TspProblem(const Adjacency &adj) : adjacency_matrix_(kN) {
    for (size_t i=0; i<kN; ++i)
      adjacency_matrix_[i].resize(kN);
      
    for (size_t i=0; i<kN; ++i)
      for (size_t j=0; j<kN; ++j)
        adjacency_matrix_[i][j] = adj[i][j];
  }
  
  Solution NewSolution() {
    Solution solution;
    
    for (size_t i=0; i<kN; ++i)
      solution[i] = i;
    std::random_shuffle(solution.tour_.begin(), solution.tour_.end());

    CostFunction(solution);
    
    return solution;
  }
  
  double CostFunction(Solution &solution) {
    double cost = 0.0;
    for (size_t i=0; i<kN-1; ++i)
      cost += adjacency_matrix_[solution[i]][solution[i+1]];
    solution.Quality() = cost + adjacency_matrix_[solution[kN-1]][solution[0]];
    return solution.Quality();
  }
  
  static constexpr auto n = kN;
 private:
  std::vector<std::vector<double> > adjacency_matrix_;
};

class Random {
 public:
  template <class IntType>
  Random(const IntType min, const IntType max) : uniform(min, max) {}
  
  auto operator()() {
    return uniform(random_engine);
  }
 private:
  std::mt19937 random_engine{std::random_device{}()};
  std::uniform_int_distribution<int> uniform;
};

template <class Problem, class Solution>
class Mutator {
 public:
  virtual void Premutate(Random &random) = 0;
  virtual void MutateFrom(const Solution &, Random &) = 0;
  virtual double DeltaQuality(const Problem &, const Solution &) = 0;
  virtual void Mutate(Solution &) = 0;
  
  virtual ~Mutator() {}
};

// insert to left
template <class Problem, class Solution>
class InsertMutator : public Mutator<Problem, Solution> {
 public:
  void Premutate(Random &random) override {
    to_pos_ = from_pos_ = random();
    while (to_pos_ == from_pos_)
      to_pos_ = random();
  }
  
  void MutateFrom(const Solution &solution, Random &random) override {}
  
  double DeltaQuality(const Problem &problem,
                      const Solution &solution) override {
    delta_ = 0.0;
    const auto &adj = problem.adjacency_matrix_;
    
    if (solution.At(from_pos_+1) != solution[to_pos_]) {
      delta_ -= adj[solution.At(from_pos_-1)][solution[from_pos_]] +
        adj[solution[from_pos_]][solution.At(from_pos_+1)] + 
        adj[solution.At(to_pos_-1)][solution[to_pos_]];
        
      delta_ += adj[solution.At(from_pos_-1)][solution.At(from_pos_+1)] +
        adj[solution.At(to_pos_-1)][solution[from_pos_]] + 
        adj[solution[from_pos_]][solution[to_pos_]];
    }

    return delta_;
  }
  
  void Mutate(Solution &solution) override {
    if (to_pos_ < from_pos_) {
      auto temp = solution[from_pos_];
      for (int i=from_pos_; i>to_pos_; --i) // to_pos_ is moved backward
        solution[i] = solution[i-1];
      solution[to_pos_] = temp; 
    } else {
      auto temp = solution[from_pos_];
      for (int i=from_pos_; i<to_pos_-1; ++i) // to_pos_ isn't moved forward
        solution[i] = solution[i+1];
      solution[to_pos_-1] = temp;
    }
    
    solution.Quality() += delta_;
  }
 
 private:
  int from_pos_ = 0;
  int to_pos_ = 0;
  
  double delta_ = 0.0;
};

template <class Problem, class Solution>
class InverseMutator : public Mutator<Problem, Solution> {
 public:
  void Premutate(Random &random) override {
    left_pos_ = right_pos_ = random();
    while (left_pos_ == right_pos_)
      left_pos_ = random();
    
    if (left_pos_ > right_pos_)
      std::swap(left_pos_, right_pos_);
  }
 
  void MutateFrom(const Solution &solution, Random &random) {}
  
  double DeltaQuality(const Problem &problem,
                      const Solution &solution) override {
    delta_ = 0.0;
    const auto &adj = problem.adjacency_matrix_;
    
    if (left_pos_-1+solution.n != right_pos_) {
      delta_ -= adj[solution.At(left_pos_-1)][solution[left_pos_]] +
        adj[solution[right_pos_]][solution.At(right_pos_+1)];
      
      delta_ += adj[solution.At(left_pos_-1)][solution[right_pos_]] +
        adj[solution[left_pos_]][solution.At(right_pos_+1)];
    }
    
    return delta_;
  }
  
  void Mutate(Solution &solution) override {
    int i = left_pos_, j = right_pos_;
    while (i<j)
      std::swap(solution[i++],solution[j--]);

    solution.Quality() += delta_;
  }
 
 private:
  int left_pos_ = 0;
  int right_pos_ = 0;
  
  double delta_ = 0.0;
};

template <class Problem, class Solution>
class MutatorManager
    : public anneal::MutatorManagerInterface<Problem, Solution> {
 public:
  void Premutate() {
    if (uniform_(random_engine_))
      current_mutator_ = &insert_mutator_;
    else
      current_mutator_ = &inverse_mutator_;
      
    current_mutator_->Premutate(random_);
  }
  
  void MutateFrom(const Solution &) {
    mutate_ready_ = true;
  }
  
  double DeltaQuality(const Problem &problem, const Solution &solution) {
    assert(mutate_ready_);
    return current_mutator_->DeltaQuality(problem, solution);
  }
  
  virtual void Mutate(Solution &solution) {
    assert(mutate_ready_);
    current_mutator_->Mutate(solution);
    mutate_ready_ = false;
  }
  
 private:
  Mutator<Problem, Solution> *current_mutator_ = nullptr;
  InsertMutator<Problem, Solution> insert_mutator_;
  InverseMutator<Problem, Solution> inverse_mutator_;
  
  bool mutate_ready_ = false;
  
  std::mt19937 random_engine_{std::random_device{}()};
  std::uniform_int_distribution<int> uniform_{0, 1};
 
  Random random_{0, static_cast<int>(Solution::n-1)};
};


} // tsp

#endif // _ANNEAL_TSP_H_
