#pragma once
#include <algorithm>
#include <vector>
#include <problem.h>

class Filter {
public:
    struct Entry {
        double feas;
        double merit;
    };

    // Entries maintained so no stored entry is dominated by another
    std::vector<Entry> entries;

    /**
     * Construct a filter with default progress parameters
     */
    Filter() : Filter(1e-5, 1e-5) {}

    /**
     * Construct a filter with explicit progress parameters
     *
     * @param gamma_objective Sufficient merit decrease parameter
     * @param gamma_constraint Sufficient feasibility decrease parameter
     */
    Filter(const double gamma_objective, const double gamma_constraint) : gamma_objective(gamma_objective), 
                                                gamma_constraint(gamma_constraint), entries(std::vector<Entry>()) {}

    /**
     * Check sufficient progress against one filter entry
     *
     * @param candidate Candidate filter entry 
     * @param entry Existing filter entry to compare against
     * @return Pair for feasibility progress and merit progress
     */
    std::pair<bool, bool> sufficient_progress(const Entry& candidate, const Entry& entry) const;

    /**
     * Check whether a candidate is acceptable against one entry
     *
     * @param candidate Candidate filter entry
     * @param entry Existing filter entry to compare against
     * @return True when candidate makes feasibility or merit progress
     */
    bool candidate_acceptable(const Entry& candidate, const Entry& entry) const;

    /**
     * Check whether a candidate is dominated by one entry
     *
     * @param candidate Candidate filter entry
     * @param entry Existing filter entry to compare against
     * @return True when candidate makes neither feasibility nor merit progress
     */
    bool candidate_dominated(const Entry& candidate, const Entry& entry) const;

    /**
     * Check whether a candidate is acceptable to the whole filter
     *
     * @param candidate Candidate filter entry
     * @return True when no existing entry rejects the candidate
     */
    bool acceptable(const Entry& candidate);

    /**
     * Add an entry and remove entries it dominates
     *
     * @param new_entry New filter entry to add
     */
    void update(const Entry& new_entry);

    /**
     * Remove every filter entry
     */
    void clear();
private:
    double gamma_objective{1e-5};
    double gamma_constraint{1e-5};
};
