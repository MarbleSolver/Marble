#include "workspace.h"
#include <algorithm>

Workspace::Workspace() {
    // Empty constructor
}

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

int Workspace::findValuePtrIndex(int row, int col) {
    // Get indices for the start and end of the column
    int col_start = kkt_system.outerIndexPtr()[col];
    int col_end = kkt_system.outerIndexPtr()[col + 1];

    // Search over the row for the row index
    const QDLDL_int* inner = kkt_system.innerIndexPtr(); // This is like rowval in CSC format
    const QDLDL_int* it = std::lower_bound(inner + col_start, inner + col_end, row);
    if (it != inner + col_end && *it == row) {
        return it - inner;
    }
    return -1;
}