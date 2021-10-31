// Copyright (C) Nicholas Palomo. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

/*
This environment implements a simple locomotion example for the Cassie robot
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

                cassie_ = world_->insertRobot(
                get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/gym_ignition_models/cassie/cassie.urdf",
                "cassie" + std::to_string(envIndex),
                MODEL::readJointSerializationFromYaml(cfg["joint_serialization"]),
                "floating",
                "torso");

                num_joints_ = cassie_->numJoints();

                cassie_->enableContacts(true);

                // Reset the initial joint positions and velocities
                initial_joint_positions_ = cfg_["initial_joint_positions"].template as<std::vector<double>>();
                initial_joint_velocities_ = std::vector<double>(initial_joint_positions_.size(), 0.);
                initial_joint_positions_vec_ = stdVec2EigenVec<double>(initial_joint_positions_);

                cassie_->resetJointStates(initial_joint_positions_, initial_joint_velocities_);

                // controller period for joint PIDs
                cassie_->setJointControllerPeriod(cfg_["joint_control_dt"].template as<double>());

                // Set the control mode for the joints
                cassie_->setJointControlMode(scenario::core::JointControlMode::Force);

                // Populate the P, D gains from the YAML config
                P_.setZero(num_joints_, num_joints_);
                D_.setZero(num_joints_, num_joints_);
                int i = 0;
                for(auto joint_name : robot_->jointNames()) {
                    P_(i, i) = cfg_["pid_gains"][joint_name]["p"].template as<double>();
                    D_(i, i) = cfg_["pid_gains"][joint_name]["d"].template as<double>();
                    i++;
                }

                // Set the control callback
                setRlControlCallback(cfg_["control_target_type"]);

                // Initialize the GymIgnitionEnv private members
                actionDim_ = 4;
                obsDim_ = 4;

                ob_scaled_.setZero(obsDim_);
                ob_double_.setZero(obsDim_);
                ob_mean_.setZero(obsDim_);
                ob_std_.setZero(obsDim_);
                action_mean_.setZero(actionDim_);
                action_std_.setZero(actionDim_);
                p_target_.setZero(num_joints_);
                action_unscaled_.setZero(actionDim_);
                final_joint_targets_.setZero(num_joints_-2);
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

            void setRlControlCallback(
                const YAML::Node node) {
                
                // In this method, user can specify what type of target to be applied at the joints - for example: joint torques, joint positions, joint velocities, etc.
            
                std::string control_target_type = node.template as<std::string>();

                // specify joint POSITION targets for the actuators
                if((control_target_type.find("joint") != std::string::npos) && (control_target_type.find("angle") != std::string::npos)) {
                    rlControlCallbackPtr_ = [this]() { robot_->setJointPositionTargets(p_target_); };
                }

                // specify joint EFFORT targets for actuators
                if((control_target_type.find("joint") != std::string::npos) && (control_target_type.find("effort") != std::string::npos)) {
                    rlControlCallbackPtr_ = [this]() { robot_->setJointEffortTargets(p_target_); };
                }
            }

        private:

            std::mt19937 rand_num_gen_;
            std::normal_distribution<double> distribution_;
            std::uniform_real_distribution<double> uniform_dist_;

            std::unordered_map<std::string, double> extraInfo_;

            std::unique_ptr<KINEMATICS> cassie_;

            Eigen::VectorXd ob_double_, 
                            ob_scaled_, 
                            action_unscaled_, 
                            action_mean_, 
                            action_std_,
                            ob_mean_,
                            ob_std_, 
                            p_target_, 
                            initial_joint_positions_vec_,
                            final_joint_targets_;

            Eigen::MatrixXd P_, D_;

            int num_joints_;

            std::vector<double> initial_joint_positions_, initial_joint_velocities_;
        
    };

}