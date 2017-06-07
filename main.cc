#include "src/magic-square.h"
#include "src/n-queens.h"
#include "src/simulated-annealing.h"
#include "src/tsp.h"

#include <chrono>
#include <exception>
#include <iostream>

void NQueensExemple() {
  nqueens::NQueensProblem<400> n_queens;
  anneal::TemperatureBasic<decltype(n_queens)::SolutionType> policy(
      std::chrono::seconds(20), std::chrono::seconds(10));
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
  anneal::TemperatureBasic<decltype(magic_square)::SolutionType> policy(
      std::chrono::seconds(20), std::chrono::seconds(10));
  anneal::SimulatedAnnealing simulated_annealing;

  auto solution = simulated_annealing.MultiThread(policy, magic_square);

  for (int i=0; i<solution.n; ++i) {
    for (int j=0; j<solution.n; ++j)
      std::cout << solution[i][j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
  
  std::cout << solution.Quality() << std::endl;
}

void TspExemple() {
  const size_t kN = 10;
  double adj[kN][kN] = {0.0};
  
  std::mt19937 random_engine{std::random_device{}()};
  std::uniform_real_distribution<float> uniform{0,1};
  
  for (size_t i=0; i<kN; ++i)
    for (size_t j=i+1; j<kN; ++j)
      adj[i][j] = adj[j][i] = uniform(random_engine);
      
  tsp::TspProblem<kN> problem(adj);
  anneal::TemperatureBasic<decltype(problem)::SolutionType> policy(
      std::chrono::seconds(20), std::chrono::seconds(10));
  anneal::SimulatedAnnealing simulated_annealing;
  
  simulated_annealing.Debug(problem);
  auto solution = simulated_annealing(policy, problem);
  
  std::cout << solution.Quality() << std::endl;
  for (size_t i=0; i<solution.n; ++i)
    std::cout << solution[i] << " ";
  std::cout << std::endl;
}

int main() {
  NQueensExemple();
  
  return 0;
}

