#pragma once
#include <algorithm>
#include <vector>

class Filter {
public:
    struct FilterOptions {
        double gamma_objective{1e-5};
        double gamma_constraint{1e-5};
    };

    struct FilterEntry {
        double objective_value{__DBL_MAX__};
        double constraint_violation{__DBL_MAX__};

        FilterEntry() = default;
        FilterEntry(double objective_value, double constraint_violation)
            : objective_value(objective_value),
              constraint_violation(constraint_violation) {}
    };

    Filter() = default;
    Filter(const FilterOptions& options) : options(options) {}

    std::pair<double, double> sufficient_progress(const FilterEntry& candidate, const FilterEntry& entry) const;

    bool candidate_acceptable(const FilterEntry& candidate, const FilterEntry& entry) const;
    bool candidate_dominated(const FilterEntry& candidate, const FilterEntry& entry) const;

    bool acceptable(const FilterEntry& candidate);
    void update(const FilterEntry& new_entry);
private:
    std::vector<FilterEntry> entries{};
    FilterOptions options{};
};