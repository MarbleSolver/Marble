#include "solver.h"

Solver::Solver() {

};

void Solver::set_problem(Problem& prob) {
    this->prob = std::make_shared<Problem>(std::move(prob));
}