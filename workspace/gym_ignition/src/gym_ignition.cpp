// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

#include "Environment.hpp"

int main(int argc, char* argv[]) {

    std::string resourceDir = argv[1];
    std::stringstream cfg;
    cfg << argv[2];
    YAML::Node config = YAML::Load(cfg.str());
    int envIndex = atoi(argv[3]);
    bool render = strcmp(argv[4], "True") == 0;
    std::cout << argv[1] << std::endl;
    std::cout << argv[2] << std::endl;

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<gym_ignition::ENVIRONMENT>(resourceDir, config, envIndex, envIndex==0 && render));
    rclcpp::shutdown();
    
    return 0;
}