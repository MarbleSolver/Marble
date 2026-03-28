#include <filter.h>

std::pair<bool, bool> Filter::sufficient_progress(const Filter::Entry& candidate, const Filter::Entry& entry) const {
    auto feas = candidate.feas;
    auto merit = candidate.merit;
    
    bool violation_progress = feas <= (1 - gamma_constraint) * entry.feas;
    bool objective_progress = merit <= entry.merit - gamma_objective * entry.feas;

    return {violation_progress, objective_progress};
}

bool Filter::candidate_acceptable(const Filter::Entry& candidate, const Filter::Entry& entry) const {
    auto [violation_progress, objective_progress] = sufficient_progress(candidate, entry);
    return violation_progress || objective_progress;
}

bool Filter::candidate_dominated(const Filter::Entry& candidate, const Filter::Entry& entry) const {
    auto [violation_progress, objective_progress] = sufficient_progress(candidate, entry);
    return !violation_progress && !objective_progress;
}

// TODO: the `sufficient_progress` function is currently being called twice per entry in the filter, which is "inefficient"
// can just call it once and use the results for both acceptable and dominated checks
bool Filter::acceptable(const Filter::Entry& candidate) {
    return entries.empty() || std::all_of(entries.begin(), entries.end(), [&](const Filter::Entry& entry) { return candidate_acceptable(candidate, entry); });
}

void Filter::update(const Filter::Entry& new_entry) {
    // Remove any entries that are dominated by the new entry
    entries.erase(
        std::remove_if(
            entries.begin(), entries.end(),
            [&](const Filter::Entry& entry) {
                return candidate_dominated(entry, new_entry); 
            }
        ), entries.end()
    );

    // Add new entry to filter
    entries.push_back(new_entry);
}

void Filter::clear() {
    entries.clear();
}