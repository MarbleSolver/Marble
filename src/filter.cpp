#include <filter.h>

std::pair<double, double> Filter::sufficient_progress(const Filter::FilterEntry& candidate, const Filter::FilterEntry& entry) const {
    bool violation_progress = candidate.constraint_violation <= (1 - options.gamma_constraint) * entry.constraint_violation;
    bool objective_progress = candidate.objective_value <= entry.objective_value - options.gamma_objective * entry.constraint_violation;

    return {violation_progress, objective_progress};
}

bool Filter::candidate_acceptable(const Filter::FilterEntry& candidate, const Filter::FilterEntry& entry) const {
    auto [violation_progress, objective_progress] = sufficient_progress(candidate, entry);
    return violation_progress || objective_progress;
}

bool Filter::candidate_dominated(const Filter::FilterEntry& candidate, const Filter::FilterEntry& entry) const {
    auto [violation_progress, objective_progress] = sufficient_progress(candidate, entry);
    return !violation_progress && !objective_progress;
}

// TODO: the `sufficient_progress` function is currently being called twice per entry in the filter, which is "inefficient"
// can just call it once and use the results for both acceptable and dominated checks
bool Filter::acceptable(const Filter::FilterEntry& candidate) {
    return std::any_of(entries.begin(), entries.end(), [&](const Filter::FilterEntry& entry) { return candidate_acceptable(candidate, entry); });
}

void Filter::update(const Filter::FilterEntry& new_entry) {
    // Remove any entries that are dominated by the new entry
    entries.erase(std::remove_if(entries.begin(), entries.end(),
        [&](const Filter::FilterEntry& entry) { return candidate_dominated(entry, new_entry); }), entries.end());

    // Add new entry to filter
    entries.push_back(new_entry);
}