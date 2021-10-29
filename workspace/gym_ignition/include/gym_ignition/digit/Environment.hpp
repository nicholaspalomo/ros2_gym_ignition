// Copyright (C) Nicholas Palomo. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

/*
This environment implements a simple locomotion example for Digit - the humanoid robot from Agility Robotics
*/

#include "GymIgnitionEnv.hpp"

namespace gym_ignition {

    class ENVIRONMENT : public GymIgnitionEnv {

        public:

            ENVIRONMENT(
                const std::string& resource_dir,
                const YAML::Node& cfg,
                int env_index,
                bool render) :
                GymIgnitionEnv(resource_dir, cfg, env_index, render),
                distribution_(-1.0, 1.0),
                uniform_dist_(-1.0, 1.0),
                rand_num_gen_(std::chrono::system_clock::now().time_since_epoch().count()) {

                // IMPORTANT! Set the simulation timestep and the RL controller timestep
                setSimulationTimeStep(cfg_["step_size"].template as<double>());
                setControlTimeStep(cfg_["control_dt"].template as<double>());

                world_->insertWorld(get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/worlds/empty.world", "empty" + std::to_string(envIndex));

                // Insert ground plane
                world_->insertGround(get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/gym_ignition_models/ground_plane/ground_plane.sdf", "ground" + std::to_string(envIndex), /*enable contacts=*/true);

                // Open the GUI, if you enabled visualizing the training when launching the simulation
                if(visualizable_)
                    world_->openGazeboGui(5);

                digit_ = world_->insertRobot(
                get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/gym_ignition_models/digit/digit.urdf",
                "digit" + std::to_string(envIndex),
                MODEL::readJointSerializationFromYaml(cfg["joint_serialization"]),
                "floating",
                "torso");

                num_joints_ = robot_->numJoints();

                robot_->enableContacts(true);

                // Reset the initial joint positions and velocities
                // TODO: Finish setting up MDP for Digit

            }

            void init() final { }

            void reset() final {

            }

            float step(
                const Eigen::VectorXd& action) final {

            }

            bool isTerminal(
                float& terminalReward) final {


            }

            void getExtraInfo(
                std::unordered_map<std::string, double>& extraInfo,
                gym_ignition::msg::Rgb& rgbImg,
                gym_ignition::msg::Depth& depthImg,
                gym_ignition::msg::Thermal& thermalImg) {

                extraInfo = extraInfo_;
            }

            void setSeed(
                int seed) final {

            }

            void observe(
                Eigen::VectorXd& ob) final {
                
                ob = ob_scaled_;
            }

        private:

            std::mt19937 rand_num_gen_;
            std::normal_distribution<double> distribution_;
            std::uniform_real_distribution<double> uniform_dist_;

            std::unordered_map<std::string, double> extraInfo_;

            std::unique_ptr<KINEMATICS> digit_;

            Eigen::VectorXd ob_double_, ob_scaled_, action_unscaled_, action_mean_, action_std_, ob_mean_, ob_std_, p_target_;

            int num_joints_;

            std::vector<double> initial_joint_positions_, initial_joint_velocities_;
        
    };

}