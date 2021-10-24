// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

#include "Environment.hpp"

int main(int argc, char* argv[]) {

    std::string resourceDir = argv[1];
    YAML::Node yaml = YAML::LoadFile(resourceDir);
    std::stringstream cfg;
    cfg << yaml["environment"];
    YAML::Node config = YAML::Load(cfg.str());
    int envIndex = 0;

    rclcpp::init(argc, argv);

    auto node = std::make_shared<gym_ignition::ENVIRONMENT>(resourceDir, config, envIndex);

    auto obsDim = node->getObsDim();
    auto actionDim = node->getActionDim();

    Eigen::VectorXd observation;
    observation.setZero(obsDim);
    Eigen::VectorXd action;
    action.setZero(actionDim);
    float reward;
    bool isDone;

    node->reset();

    for(int i = 0; i < 10000; i++) {
        rclcpp::spin_some(node);

        reward = node->step(action);

        node->observe(observation);

        // TODO: Get the camera observation (if a camera is present)
        // TODO: Get extra_info map

        float terminalReward = 0.0f;
        isDone = node->isTerminalState(terminalReward);

        if(isDone)
            node->reset();

        std::cout << "Total reward: " << reward + terminalReward << std::endl;
    }

    rclcpp::shutdown();
    
    return 0;
}