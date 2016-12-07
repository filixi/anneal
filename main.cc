#include "src/magic-square.h"
#include "src/n-queens.h"
#include "src/simulated-annealing.h"

#include <chrono>
#include <exception>
#include <iostream>

void NQueensExemple() {
  nqueens::NQueensProblem<200> n_queens;
  anneal::TemperatureBasic<decltype(n_queens)::SolutionType> policy;
  anneal::SimulatedAnnealing simulated_annealing;

  auto solution = simulated_annealing.MultiThread(policy, n_queens);
  
  std::cout << solution.Quality() << std::endl;
  std::cout << std::endl;
  for (int i=0; i<solution.n; ++i)
    std::cout << solution[i] << " ";
  std::cout << std::endl;
}

void MagicSquareExemple() {
  magicsquare::MagicSquareProblem<200> magic_square;
  anneal::TemperatureBasic<decltype(magic_square)::SolutionType> policy;
  anneal::SimulatedAnnealing simulated_annealing;

  auto solution = simulated_annealing.MultiThread(policy, magic_square);
  
  std::cout << solution.Quality() << std::endl;
  
  for (int i=0; i<solution.n; ++i) {
    for (int j=0; j<solution.n; ++j)
      std::cout << solution[i][j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  const auto time = std::chrono::steady_clock::now();
  
  NQueensExemple();
  
  const auto duration = std::chrono::steady_clock::now() - time;
  std::cout << duration.count() << std::endl;
  
  return 0;
}

