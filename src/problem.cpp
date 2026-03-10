#include "problem.h"

Problem::Problem(SMat cost_hessian, Vec cost_gradient,
                 SMat J_eq, Vec c_eq, SMat J_ineq, Vec c_ineq,
                 SMat J_comp_l, Vec c_comp_l, SMat J_comp_r, Vec c_comp_r)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(J_comp_l.rows()),
      cost_hessian(std::move(cost_hessian)), cost_gradient(std::move(cost_gradient)),
      J_eq(std::move(J_eq)), c_eq(std::move(c_eq)),
      J_ineq(std::move(J_ineq)), c_ineq(std::move(c_ineq)),
      J_comp_l(std::move(J_comp_l)), c_comp_l(std::move(c_comp_l)),
      J_comp_r(std::move(J_comp_r)), c_comp_r(std::move(c_comp_r)) {};

Problem::Problem(Mat cost_hessian, Vec cost_gradient,
                 Mat J_eq, Vec c_eq, Mat J_ineq, Vec c_ineq,
                 Mat J_comp_l, Vec c_comp_l, Mat J_comp_r, Vec c_comp_r)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(J_comp_l.rows()),
      cost_hessian(cost_hessian.sparseView()), cost_gradient(std::move(cost_gradient)),
      J_eq(J_eq.sparseView()), c_eq(std::move(c_eq)),
      J_ineq(J_ineq.sparseView()), c_ineq(std::move(c_ineq)),
      J_comp_l(J_comp_l.sparseView()), c_comp_l(std::move(c_comp_l)),
      J_comp_r(J_comp_r.sparseView()), c_comp_r(std::move(c_comp_r)) {};