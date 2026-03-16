#pragma once
#include <algorithm>
#include <vector>

class Filter {
public:
    struct Options {
        // Sufficient progress parameter for objective value decrease
        double gamma_objective{1e-5};
        // Sufficient progress parameter for constraint violation decrease
        double gamma_constraint{1e-5};
    };

    struct FilterEntry {
        double objective_value{__DBL_MAX__};
        double constraint_violation{__DBL_MAX__};

        /**
         * @brief Construct a new Filter Entry object
         */
        FilterEntry() = default;
        FilterEntry(double objective_value, double constraint_violation)
            : objective_value(objective_value),
              constraint_violation(constraint_violation) {}
    };

    /**
     * @brief Construct a new Filter object
     */
    Filter() = default;
    Filter(const Options& options) : options(options) {}

    /**
     * @brief Determine if a candidate point makes sufficient progress with respect to constraint violation, objective value
     * decrease compared to an entry in the filter
     * 
     * @param candidate Candidate filter entry 
     * @param entry Existing filter entry to compare against
     * @return std::pair<bool, bool> Pair indicating whether sufficient progress is made in constraint violation, objective value decrease, respectively
     */
    std::pair<bool, bool> sufficient_progress(const Filter::FilterEntry& candidate, const Filter::FilterEntry& entry) const;

    /**
     * @brief Determine if a candidate point is acceptable compared to an existing entry in the filter. Defined as making either
     * sufficient constraint violation progress or sufficient objective value decrease compared to any entry in the filter.
     * 
     * @param candidate Candidate filter entry
     * @param entry Existing filter entry to compare against
     * @return true Candidate is acceptable
     * @return false Candidate is not acceptable
     */
    bool candidate_acceptable(const FilterEntry& candidate, const FilterEntry& entry) const;

    /**
     * @brief Determine if a candidate point is dominated by an existing entry in the filter. Defined as not making sufficient
     * progress in either constraint violation or objective value decrease compared to the existing entry.
     * 
     * @param candidate Candidate filter entry
     * @param entry Existing filter entry to compare against
     * @return true Candidate is dominated
     * @return false Candidate is not dominated
     */
    bool candidate_dominated(const FilterEntry& candidate, const FilterEntry& entry) const;

    /**
     * @brief Determine if a candidate point is acceptable compared to any entry in the filter. Defined as making either
     * sufficient constraint violation progress or sufficient objective value decrease compared to any entry in the filter.
     * 
     * @param candidate Candidate filter entry
     * @return true Candidate is acceptable
     * @return false Candidate is not acceptable
     */
    bool acceptable(const FilterEntry& candidate);

    /**
     * @brief Add a new entry to the filter and remove any entries that are dominated by the new entry
     * 
     * @param new_entry New filter entry to add
     */
    void update(const FilterEntry& new_entry);

    /**
     * @brief Clear all entries from the filter
     */
    void clear() {
        entries.clear();
    }
private:
    // Filter options
    const Options options{};

    // List of entries in the filter, maintained such that no entry is dominated by any other entry
    std::vector<FilterEntry> entries{};
};