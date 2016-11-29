#include "src/n-queens.h"
#include "src/simulated-annealing.h"

int main() {
  nqueens::NQueensProblem<400> n_queens;
  anneal::TemperatureBasic<decltype(n_queens)::SolutionType> policy;
  anneal::SimulatedAnnealing simulated_annealing;

  auto solution = simulated_annealing.operator()(policy, n_queens);
  
  for (int i=0; i<solution.N; ++i)
    std::cout << solution[i] << " ";
  std::cout << std::endl;
  
  return 0;
}

