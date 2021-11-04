// Copyright (C) Nico Palomo. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

/*
This environment implements a simple locomotion example for ANYmal.
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

                world_->insertWorld(get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/worlds/empty.world", "empty" + std::to_string(env_index));

                // Insert ground plane
                world_->insertGround(get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/gym_ignition_models/ground_plane/ground_plane.sdf", "ground" + std::to_string(env_index), /*enable contacts=*/true);

                // Open the GUI, if you enabled visualizing the training when launching the simulation
                if(visualizable_)
                    world_->openGazeboGui(5);

                anymal_ = world_->insertRobot(
                    get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/gym_ignition_models/anymal/anymal.urdf",
                    "anymal" + std::to_string(env_index),
                    MODEL::readJointSerializationFromYaml(cfg["joint_serialization"]),
                    "floating",
                    "base");

                num_joints_ = anymal_->numJoints();
                anymal_->printJointNames();

                anymal_->enableContacts(true);

                gc_.setZero(spot_->generalizedCoordinateDim());
                gc_init_.setZero(spot_->generalizedCoordinateDim());
                gv_.setZero(spot_->generalizedVelocityDim());
                gv_init_.setZero(spot_->generalizedCoordinateDim());

                anymal_->setJointControllerPeriod(cfg_["joint_control_dt"].template as<double>());

                anymal_->setJointControlMode(scenario::core::JointControlMode::Force);

                p_gain_.setZero(num_joints_);
                i_gain_.setZero(num_joints_);
                d_gain_.setZero(num_joints_);
                int i = 0;
                for(auto joint_name : anymal_->jointNames()) {
                    p_gain_(i) = cfg_["pid_gains"][joint_name]["p"].template as<double>();
                    i_gain_(i) = cfg_["pid_gains"][joint_name]["i"].template as<double>();
                    d_gain_(i) = cfg_["pid_gains"][joint_name]["d"].template as<double>();

                    gc_init_(i) = cfg_["initial_joint_positions"].template as<std::vector<double>>()[i];
                }

                obsDim_ =   1 + /* body height */
                            3 + /* body z-axis */
                            num_joints + /* joint angles */
                            num_joints + /* joint velocities */
                            3 + /* body linear velocities */
                            3; /* body angular velocities */

                actionDim_ = num_joints_; /* joint torques */

                action_mean_.setZero(actionDim_);
                action_std_.setZero(actionDim_);
                ob_mean_.setZero(obsDim_);
                ob_std_.setZero(obsDim_);

                action_mean_ = gc_init_.tail(num_joints_);
                action_std_.setConstant(0.6);

                ob_mean_ << 0.44,
                            0., 0., 0.,
                            gc_init_.tail(num_joints_),
                            Eigen::VectorXd::Constant(num_joints_, 0.),
                            Eigen::VectorXd::Constant(6, 0.);

                ob_std_  << 0.12,
                            Eigen::VectorXd::Constant(3, 0.7),
                            Eigen::VectorXd::Constant(num_joints_, 1.0),
                            Eigen::VectorXd::Constant(num_joints_, 10.0),
                            Eigen::VectorXd::Constant(3, 2.0),
                            Eigen::VectorXd::Constant(3, 4.0);

            }

        private:

            std::mt19937 rand_num_gen_;
            std::normal_distribution<double> distribution_;
            std::uniform_real_distribution<double> uniform_dist_;

            std::unordered_map<std::string, double> extraInfo_;

            std::unique_ptr<Kinematics> anymal_;

            Eigen::VectorXd gc_,
                            gc_init_,
                            gv_,
                            gv_init_,
                            v_target_,
                            ob_double_, 
                            ob_scaled_, 
                            action_unscaled_, 
                            action_mean_, 
                            action_std_,
                            ob_mean_,
                            ob_std_, 
                            p_target_, 
                            initial_joint_positions_vec_,
                            final_joint_targets_,
                            action_history_,
                            gen_force_,
                            p_gain_,
                            i_gain_,
                            d_gain_,
                            err_curr_,
                            err_prev_,
                            err_int_;

            int num_joints_;
    };

}