#include "problem.h"

Problem::Problem(SMat cost_hessian, Vec cost_gradient,
                 SMat J_eq, Vec c_eq, SMat J_ineq, Vec c_ineq,
                 SMat J_comp, Vec c_comp)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(J_comp.rows()/2),
      cost_hessian(std::move(cost_hessian)), cost_gradient(std::move(cost_gradient)),
      J_eq(std::move(J_eq)), c_eq(std::move(c_eq)),
      J_ineq(std::move(J_ineq)), c_ineq(std::move(c_ineq)),
      J_comp(std::move(J_comp)), c_comp(std::move(c_comp)) {};

Problem::Problem(Mat cost_hessian, Vec cost_gradient,
                 Mat J_eq, Vec c_eq, Mat J_ineq, Vec c_ineq,
                 Mat J_comp, Vec c_comp)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(J_comp.rows()/2),
      cost_hessian(cost_hessian.sparseView()), cost_gradient(std::move(cost_gradient)),
      J_eq(J_eq.sparseView()), c_eq(std::move(c_eq)),
      J_ineq(J_ineq.sparseView()), c_ineq(std::move(c_ineq)),
      J_comp(J_comp.sparseView()), c_comp(std::move(c_comp)) {};