#pragma once

// std io
#include <iostream>
#include <vector>

typedef enum enum_Matches_type {
  Tracked_Matches = 0,
  Relocalization_Matches = 1,
} Matches_type;

//!
/*!

*/
class Associator {
public:
  //
  bool updated_without_optimization = false;

  //
  Associator();
  ~Associator();

  //
  std::vector<std::pair<int, int>> tracked_submap_id_pair;
  std::vector<std::vector<std::pair<int, int>>> tracked_matches;
  //
  std::vector<std::pair<int, int>> relocalization_submap_id_pair;
  std::vector<std::vector<std::pair<int, int>>> relocalization_matches;
  //
  std::vector<std::pair<int, int>> all_submap_id_pair;
  std::vector<std::vector<std::pair<int, int>>> all_matches;

  //
  void
  update_matches(const std::vector<std::pair<int, int>> &matches,
                 const std::pair<int, int> &submap_id_pair,
                 const Matches_type match_type = Matches_type::Tracked_Matches);

  //
  void prepare_for_optimization();
};
