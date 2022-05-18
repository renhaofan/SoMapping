

#include "Associator.h"

//
Associator::Associator() {}
Associator::~Associator() {}

//
void Associator::update_matches(const std::vector<std::pair<int, int>> &matches,
                                const std::pair<int, int> &submap_id_pair,
                                const Matches_type match_type) {
  updated_without_optimization = true;

  if (match_type == Matches_type::Tracked_Matches) {
    this->tracked_matches.push_back(matches);
    this->tracked_submap_id_pair.push_back(submap_id_pair);
  } else if (match_type == Matches_type::Relocalization_Matches) {
    this->relocalization_matches.push_back(matches);
    this->relocalization_submap_id_pair.push_back(submap_id_pair);
  }
}

//
void Associator::prepare_for_optimization() {
  this->all_submap_id_pair.clear();
  this->all_matches.clear();

  //
  this->all_submap_id_pair.insert(this->all_submap_id_pair.end(),
                                  this->tracked_submap_id_pair.begin(),
                                  this->tracked_submap_id_pair.end());
  this->all_submap_id_pair.insert(this->all_submap_id_pair.end(),
                                  this->relocalization_submap_id_pair.begin(),
                                  this->relocalization_submap_id_pair.end());
  this->all_matches.insert(this->all_matches.end(),
                           this->tracked_matches.begin(),
                           this->tracked_matches.end());
  this->all_matches.insert(this->all_matches.end(),
                           this->relocalization_matches.begin(),
                           this->relocalization_matches.end());
}
