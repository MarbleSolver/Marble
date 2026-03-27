#pragma once
#include <algorithm>
#include <vector>
#include <problem.h>

class Filter {
public:
    typedef std::pair<double, double> Entry;

    /**
     * @brief Construct a new Filter object
     */
    Filter() : Filter(1e-5, 1e-5) {}
    Filter(const double gamma_objective, const double gamma_constraint) : gamma_objective(gamma_objective), 
                                                gamma_constraint(gamma_constraint), entries(std::vector<Entry>()) {}

    /**
     * @brief Determine if a candidate point makes sufficient progress with respect to constraint violation, objective value
     * decrease compared to an entry in the filter
     * 
     * @param candidate Candidate filter entry 
     * @param entry Existing filter entry to compare against
     * @return std::pair<bool, bool> Pair indicating whether sufficient progress is made in constraint violation, objective value decrease, respectively
     */
    std::pair<bool, bool> sufficient_progress(const Entry& candidate, const Entry& entry) const;

    /**
     * @brief Determine if a candidate point is acceptable compared to an existing entry in the filter. Defined as making either
     * sufficient constraint violation progress or sufficient objective value decrease compared to any entry in the filter.
     * 
     * @param candidate Candidate filter entry
     * @param entry Existing filter entry to compare against
     * @return true Candidate is acceptable
     * @return false Candidate is not acceptable
     */
    bool candidate_acceptable(const Entry& candidate, const Entry& entry) const;

    /**
     * @brief Determine if a candidate point is dominated by an existing entry in the filter. Defined as not making sufficient
     * progress in either constraint violation or objective value decrease compared to the existing entry.
     * 
     * @param candidate Candidate filter entry
     * @param entry Existing filter entry to compare against
     * @return true Candidate is dominated
     * @return false Candidate is not dominated
     */
    bool candidate_dominated(const Entry& candidate, const Entry& entry) const;

    /**
     * @brief Determine if a candidate point is acceptable compared to any entry in the filter. Defined as making either
     * sufficient constraint violation progress or sufficient objective value decrease compared to any entry in the filter.
     * 
     * @param candidate Candidate filter entry
     * @return true Candidate is acceptable
     * @return false Candidate is not acceptable
     */
    bool acceptable(const Entry& candidate);

    /**
     * @brief Add a new entry to the filter and remove any entries that are dominated by the new entry
     * 
     * @param new_entry New filter entry to add
     */
    void update(const Entry& new_entry);

    /**
     * @brief Clear all entries from the filter
     */
    void clear();
private:
    // Filter options
    const double gamma_objective{1e-5};
    const double gamma_constraint{1e-5};

    // List of entries in the filter, maintained such that no entry is dominated by any other entry
    std::vector<Entry> entries;
};