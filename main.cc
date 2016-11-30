#include "src/magic-square.h"
#include "src/n-queens.h"
#include "src/simulated-annealing.h"

#include <exception>

void n_queen_exemple() {
  nqueens::NQueensProblem<200> n_queens;
  anneal::TemperatureBasic<decltype(n_queens)::SolutionType> policy;
  anneal::SimulatedAnnealing simulated_annealing;

  auto solution = simulated_annealing(policy, n_queens);
  
  for (int i=0; i<solution.n; ++i)
    std::cout << solution[i] << " ";
  std::cout << std::endl;
}

void MagicSquareExemple() {
  magicsquare::MagicSquareProblem<4> magic_square;
  anneal::TemperatureBasic<decltype(magic_square)::SolutionType> policy;
  anneal::SimulatedAnnealing simulated_annealing;

  auto solution = simulated_annealing(policy, magic_square);
  
  for (int i=0; i<solution.n; ++i) {
    for (int j=0; j<solution.n; ++j)
      std::cout << solution[i][j] << " ";
    std::cout << std::endl;
  }
}

int main() {
  MagicSquareExemple();
  
  return 0;
}

