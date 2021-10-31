// Copyright (C) Nico Palomo. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

/*
This environment implements a simple locomotion example for Spot from Boston Dynamics.
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

                spot_ = world_->insertRobot(
                get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/gym_ignition_models/spot/spot.urdf",
                "spot" + std::to_string(env_index),
                MODEL::readJointSerializationFromYaml(cfg["joint_serialization"]),
                "floating",
                "torso");

                num_joints_ = spot_->numJoints();

                spot_->enableContacts(true);

                // Reset the initial joint positions and velocities
                initial_joint_positions_ = cfg_["initial_joint_positions"].template as<std::vector<double>>();
                initial_joint_velocities_ = std::vector<double>(initial_joint_positions_.size(), 0.);
                initial_joint_positions_vec_ = stdVec2EigenVec<double>(initial_joint_positions_);

                spot_->resetJointStates(initial_joint_positions_, initial_joint_velocities_);

                // controller period for joint PIDs
                spot_->setJointControllerPeriod(cfg_["joint_control_dt"].template as<double>());

                // Set the control mode for the joints
                spot_->setJointControlMode(scenario::core::JointControlMode::Force);

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

                buffer_length_ = cfg_["buffer_length"].template as<double>();
                nominal_body_height_ = cfg_["body_height_target"].template as<double>();

                // Initialize the GymIgnitionEnv private members
                actionDim_ = num_joints_ + 1;
                obsDim_ = 1 +                               /* base height */
                          3 +                               /* z-axis in world frame expressed in body frame */
                          12 +                              /* joint angles */ 
                          3 +                               /* body linear velocities */
                          3 +                               /* body angular velocities */
                          12 +                              /* joint velocities */
                          3 +                               /* target velocity (x, y, yaw) */
                          3 * actionDim_ +                  /* action history (3 control steps) */
                          1 +                               /* stride length */     
                          4 +                               /* swing start */
                          4 +                               /* swing duration */
                          4 +                               /* phase time left */
                          4 +                               /* target contact state */
                          4 +                               /* target foot clearance */
                          4 +                               /* actual foot clearance */
                          buffer_length_ * 12 +             /* joint position error */
                          buffer_length_ * 12 +             /* joint velocitiy */
                          3 * buffer_length_;               /* body velocity error (x, y, yaw) */

                ob_scaled_.setZero(obsDim_);
                ob_double_.setZero(obsDim_);
                ob_mean_.setZero(obsDim_);
                ob_std_.setZero(obsDim_);
                action_mean_.setZero(actionDim_);
                action_std_.setZero(actionDim_);
                p_target_.setZero(num_joints_);
                action_unscaled_.setZero(actionDim_);
                final_joint_targets_.setZero(num_joints_-2);
                action_history_.setZero(3 * actionDim_);
                actionHistory_ << initial_joint_positions_vec_, 0.,
                                  initial_joint_positions_vec_, 0.,
                                  initial_joint_positions_vec_, 0.;

                // Body velocity command
                command_params_["x"] = cfg_["target"]["x"].template as<std::vector<double>>();
                command_params_["y"] = cfg_["target"]["y"].template as<std::vector<double>>();
                command_params_["yaw"] = cfg_["target"]["yaw"].template as<std::vector<double>>();

                // Action scaling
                action_mean_ << initial_joint_positions_vec_, 0.;
                action_std_.setConstant(0.6);
                action_std_(num_joints_) = 0.1;
                
                // Observation scaling
                ob_mean_ << nominal_body_height_,
                            0., 0., 0.,
                            initial_joint_positions_vec_,
                            Eigen::VectorXd::Constant(6, 0.0),
                            Eigen::VectorXd::Constant(12, 0.0),
                            Eigen::VectorXd::Constant(3, 0.0),
                            Eigen::VectorXd::Constant(3 * actionDim_, 0.0),
                            0.0,
                            Eigen::VectorXd::Constant(24, 0.0),
                            Eigen::VectorXd::Constant(2 * num_joints_ * buffer_length_, 0.0),
                            Eigen::VectorXd::Constant(3 * buffer_length_);

                ob_std_ <<  0.12,
                            Eigen::VectorXd::Constant(3, 0.7),
                            Eigen::VectorXd::Constant(12, 1.0),
                            Eigen::VectorXd::Constant(3, 2.0),
                            Eigen::VectorXd::Constant(3, 4.0),
                            Eigen::VectorXd::Constant(12, 10.0),
                            command_params_["x"][1], command_params_["y"][1], command_params_["yaw"][1],
                            Eigen::VectorXd::Constant(3 * actionDim_, 1.0),
                            1.0,
                            Eigen::VectorXd::Constant(16, 1.0),
                            Eigen::VectorXd::Constant(8, 0.1),
                            Eigen::VectorXd::Constant(num_joints_ * buffer_length_, 1.0),
                            Eigen::VectorXd::Constant(num_joints_ * buffer_length_, 10.0),
                            Eigen::VectorXd::Constant(3 * buffer_length_, 1.);

                

                stance_gait_params_.is_stance_gait = true;
                for(size_t i = 0; i < 4; i++) {
                    default_gait_params_.swing_start[i] = cfg_["gait_params"]["default"]["swing_start"].template as<std::vector<double>>()[i];

                    default_gait_params_.swing_duration[i] = cfg_["gait_params"]["default"]["swing_duration"].template as<std::vector<double>>()[i];

                    default_gait_params_.foot_target[i] = cfg_["gait_params"]["foot_target"].template as<double>();

                    stance_gait_params_.swing_start[i] = 0.;
                    stance_gait_params_.swing_duration[i] = 0.;
                    stance_gait_params_.foot_target[i] = 0.;
                }

                default_gait_params_.stride = cfg_["gait_params"]["default"]["stride"].template as<double>();
                default_gait_params_.max_foot_height = cfg_["gait_params"]["foot_target"].template as<double>();

                // read the foot indices from the model

            }

            void reset() final {

                spot_->resetBasePose(
                    0,
                    0,
                    nominal_body_height_,
                    0,
                    0,
                    0);

            }

        private:

            std::mt19937 rand_num_gen_;
            std::normal_distribution<double> distribution_;
            std::uniform_real_distribution<double> uniform_dist_;

            std::unordered_map<std::string, double> extraInfo_;

            std::unique_ptr<Kinematics> spot_;

            Eigen::VectorXd ob_double_, 
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

            std::vector<double> initial_joint_positions_,
                                initial_joint_velocities_;

            int buffer_length_ = 1, step = -1;

            double  nominal_body_height_, 
                    body_height_target_,
                    gait_freq_;

            std::map<std::string, std::vector<double>> command_params_;

            typedef struct GaitParams {
                double stride = 0.8;
                double max_foot_height = 0.17;
                double phase = 0.;
                double swing_start[4] = {0., 0.5, 0.5, 0.};
                double swing_duration[4] = {0.5, 0.5, 0.5, 0.5};
                double phase_time_left[4] = {1.0, 1.0, 1.0, 1.0};
                double foot_target[4] = {0.17, 0.17, 0.17, 0.17};
                double foot_position[4] = {0., 0., 0., 0.};
                bool is_stance_gait = false;
                std::array<bool, 4> foot_contact_states = {true, true, true, true};
                std::array<bool, 4> desired_contact_states = {true, true, true, true};
                std::string ee_frame_names[4] = {"front_left_foot", "front_right_foot", "rear_left_foot", "rear_right_foot"};
            };

            GaitParams  gait_params_,
                        default_gait_params_,
                        stance_gait_params_;

            std::set<size_t> foot_indices_;
    };

}