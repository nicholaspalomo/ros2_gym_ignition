// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

namespace gym_ignition{

struct reward {
    reward() { }

    std::unordered_map<std::string, float> rewardTerms_;

    void setZero() {
      for (auto&[key, val]: rewardTerms_) val = 0.0f;
    }

    float sum() {
      float sum = 0.0f;
      for (auto&[key, val]: rewardTerms_) sum += val;
      return sum;
    }

    void readFromYaml(YAML::Node config) {
        for(auto it = config.begin(); it != config.end(); it++) rewardTerms_[it->first.as<std::string>()] = it->second.as<float>();
    }

    float &operator[](std::string name) {
      return rewardTerms_[name];
    }

    void getNames(std::vector<std::string> &names) {
      names.clear();
      for (auto&[key, val]: rewardTerms_) {
        names.push_back(key);
      }

    }
};

}