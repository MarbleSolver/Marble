#include "workspace.h"

Workspace::Workspace() {

};

void Workspace::appendBlockTriplets(std::vector<Eigen::Triplet<double>>& triplets, const SMat& block, int row_start, int col_start) {
    for (int k=0; k<block.outerSize(); ++k) {
        for (SMat::InnerIterator it(block, k); it; ++it) {
            int row = row_start + it.row();
            int col = col_start + it.col();
            double value = it.value();
            triplets.emplace_back(row, col, value);
        }
    }
}