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

                // controller period for joint PIDs
                spot_->setJointControllerPeriod(cfg_["joint_control_dt"].template as<double>());

                // Set the control mode for the joints
                spot_->setJointControlMode(scenario::core::JointControlMode::Force);

                // Populate the P, D gains from the YAML config
                p_gain_.setZero(num_joints_);
                i_gain_.setZero(num_joints_);
                d_gain_.setZero(num_joints_);
                int i = 0;
                for(auto joint_name : robot_->jointNames()) {
                    p_gain_(i) = cfg_["pid_gains"][joint_name]["p"].template as<double>();
                    d_gain_(i) = cfg_["pid_gains"][joint_name]["d"].template as<double>();
                    i++;
                }

                // Set the control callback
                setRlControlCallback(cfg_["control_target_type"]);

                buffer_length_ = cfg_["buffer_length"].template as<double>();
                body_height_ = cfg_["body_height_target"].template as<double>();

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
                final_joint_targets_.setZero(num_joints_);
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
                ob_mean_ << body_height_,
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

                    auto foot_name = cfg_["end_effector_frame_names"].template as<std::vector<double>>()[i];

                    default_gait_params_.ee_frame_names[i] = foot_name;
                    stance_gait_params_.ee_frame_names[i] = foot_name;

                    // read the foot indices from the model
                    foot_indices_.insert(spot_->getLinkIndexFromName(foot_name));
                }

                default_gait_params_.stride = cfg_["gait_params"]["default"]["stride"].template as<double>();
                default_gait_params_.max_foot_height = cfg_["gait_params"]["foot_target"].template as<double>();

            }

            void setRlControlCallback(
                const YAML::Node node) {
                // In this method, user can specify what type of target to be applied at the joints - for example: joint torques, joint positions, joint velocities, etc.

                std::string controlTargetType = node.template as<std::string>();

                // specify joint POSITION targets for the actuators
                if((controlTargetType.find("joint") != std::string::npos) && (controlTargetType.find("angle") != std::string::npos)) {
                    rlControlCallbackPtr_ = [this]() { robot_->setJointPositionTargets(final_joint_targets_); };
                }

                // specify joint EFFORT targets for actuators
                if((controlTargetType.find("joint") != std::string::npos) && (controlTargetType.find("effort") != std::string::npos)) {
                    rlControlCallbackPtr_ = [this]() { robot_->setJointEffortTargets(final_joint_targets_); };
                }                

            }

            void init() final {

            }

            void reset() final {
                step_ = -1;
                gen_force_.setZero(spot_->generalizedVelocityDim());
                err_curr_.setZero(num_joints_);
                err_prev_.setZero(num_joints_);
                err_int_.setZero(num_joints_);

                spot_->resetBasePose(
                    0,
                    0,
                    body_height_,
                    0,
                    0,
                    0);

                spot_->resetJointStates(initial_joint_positions_, initial_joint_velocities_);

                world_->pauseGazebo();

                sampleVelocity_();
                updateGaitParameters_();
                updateObservation_();

            }

            inline double sampleUniform(double lower, double upper) {
                return lower + uniform_dist_(rand_num_gen_) * (upper - lower);
            }

            inline double wrap01(double a) {
                return a - fastFloor(a);
            }

            inline int fastFloor(double a) {
                int i = int(a);
                if(i > a) i--;
                return i;
            }

            float step(
                const Eigen::VectorXd& action) final {

                action_unscaled_ = action;
                action_unscaled_ = action_unscaled_.cwiseProduct(action_std_);
                action_unscaled_ += action_mean_;

                p_target_.tail(num_joints_) = action_unscaled_.head(num_joints_);
                gait_freq_ = action_unscaled_(num_joints_);

                // update the action history buffer
                Eigen::VectorXd temp;
                temp.setZero(2 * actionDim_);
                temp = action_history_.tail(2 * actionDim_);
                action_history_.tail(actionDim_) = action_unscaled_;
                action_history_.head(2 * actionDim_) = temp;

                auto loop_count = int(getControlTimeStep() / getSimulationTimeStep() + 1e-10);

                for(int i = 0; i < loop_count; i++) {
                    Eigen::VectorXd gc = spot_->generalizedCoordinates();
                    Eigen::VectorXd gv = spot_->generalizedVelocities();        

                    err_curr_ = p_target_.tail(num_joints_) - gc.tail(num_joints_);

                    gen_force_.tail(num_joints_) = p_gain_.cwiseProduct(err_curr_) + i_gain_.cwiseProduct(err_int_) + d_gain_.cwiseProduct(err_curr_ - err_prev_) / getSimulationTimeStep();

                    final_joint_targets_ = gen_force_.tail(num_joints_);

                    world_->integrate(1, rlControlCallbackPtr_);
                }
                
                updateObservation_();

                return getReward_();
            }

        private:

            void sampleVelocity_() {
                default_gait_params_.is_stance_gait = false;
                gait_params_ = default_gait_params_;

                target_velocity_[0] = sampleUniform(command_params_["x"][0], command_params_["x"][1]);
                target_velocity_[1] = sampleUniform(command_params_["y"][0], command_params_["y"][1]);
                target_velocity_[2] = sampleUniform(command_params_["yaw"][0], command_params_["yaw"][1]);

                if(uniform_dist_(rand_num_gen_) < 0.1)
                    target_velocity_[0] = 0.;
                
                if(uniform_dist_(rand_num_gen_) < 0.1)
                    target_velocity_[1] = 0.;
                
                if(uniform_dist_(rand_num_gen_) < 0.1)
                    target_velocity_[2] = 0.;
                

            }

            void updateGaitParameters_() {

                if(!gait_params_.is_stance_gait) {
                    // update phase
                    double freq = (1.0 / gait_params_.stride + gait_freq_) * getControlTimeStep();
                    gait_params_.phase = wrap01(gait_params_.phase + freq);

                    // Get the current gait parameters
                    for(int i = 0; i < 4; i++) {
                        double swing_end = wrap01(gait_params_.swing_start[i] + gait_params_.swing_duration[i]);
                        double phase_shifted = wrap01(gait_params_.phase - swing_end);
                        double swing_start_shifted = 1.0 - gait_params_.swing_duration[i];

                        if(phase_shifted < swing_start_shifted) { // stance phase
                            gait_params_.desired_contact_state[i] = 1.0;
                            gait_params_.phase_time_left[i] = (swing_start_shifted - phase_shifted) * gait_params_.stride;
                            gait_params_.foot_target[i] = 0.0;
                        } else {
                            gait_params_.desired_contact_states[i] = 0.0;
                            gait_params_.phase_time_left[i] = (1.0 - phase_shifted) * gait_params_.stride;
                            gait_params_.foot_target[i] = gait_params_.max_foot_height * ( -std::sin(2 * M_PI * phase_shifted) < 0. ? 0. : -std::sin(2 * M_PI * phase_shifted) );
                        }
                    }
                }

            }

            void getFootContacts() {

                // Get the foot contact states
                std::fill(std::begin(gait_params_.foot_contact_states, std::end(gait_params_.foot_contact_states)), false);

                int i = 0;
                for(auto ee_link_name : gait_params_.ee_frame_names) {
                    auto ee_contacts = robot_->getLinkContacts(ee_link_name);

                    // If the end effector is in contact with the ground, set the contact state to true for the end effector
                    for(auto ee_contact : ee_contacts) {
                        if((ee_contact.bodyA).find("ground") != std::string::npos || (ee_contact.bodyA).find("ground") != std::string::npos) {
                            gait_params_.foot_contact_states[i] = true;
                        }
                    }

                    i++;
                }

            }

            void getReward_() {


            }

            void updateObservation_() {
                
                Eigen::VectorXd gc = spot_->generalizedCoordinates();
                Eigen::VectorXd gv = spot_->generalizedVelocities();

                getFootContacts_();

                body_height_ = gc[2]; // Assuming flat ground!

                int pos = 0;

                // body height
                ob_double_(pos) = body_height_; pos++;

                // body orientation
                Eigen::Quaterniond quat;
                quat.w() = gc[3];
                quat.x() = gc[4];
                quat.y() = gc[5];
                quat.z() = gc[6];

                Eigen::Matrix<double, 3, 3> rot_mat_body2world = quat2RotMat<double>(quat);

                ob_double_.segment(pos, 3) = rot_mat_body2world.row(2); pos += 3;

                // joint angles
                ob_double_.segment(pos, num_joints_) = gc.tail(num_joints_); pos += num_joints_;

                // body linear velocity
                body_linear_vel_ = rot_mat_body2world.transpose() * gv.segment(0, 3);
                ob_double_.segment(pos, 3) = body_linear_vel_; pos += 3;

                // body angular velocity
                body_angular_vel_ = rot_mat_body2world.transpose() * gv.segment(3, 3);
                ob_double_.segment(pos, 3) = body_angular_vel_; pos += 3;

                // joint velocities
                ob_double_.segment(pos, num_joints_) = gv.tail(num_joints_); pos += num_joints_;

                // target velocity
                ob_double_.segment(pos, 3) = target_velocity_; pos += 3;

                // action history
                ob_double_.segment(pos, 3 * actionDim_) = action_history_; pos += 3 * actionDim_;

                // gait parameters
                ob_double_(pos) = gait_params_.stride; pos++;

                for(int i = 0; i < 4; i++) {

                    ob_double_(pos) = gait_params_.swing_start[i];
                    ob_double_(pos + 4) = gait_params_.swing_duration[i];
                    ob_double_(pos + 8) = gait_params_.phase_time_left[i];
                    ob_double_(pos + 12) = gait_params_.desired_contact_states[i];
                    ob_double_(pos + 16) = gait_params_.foot_target[i];
                    ob_double_(pos + 20) = gait_params_.foot_position[i];
                    pos++;
                }
                pos += 20;

                if(step_ % buffer_stride_ == 0) {
                    step_ = 0;
                    Eigen::VectorXd temp;

                    // joint position error history
                    temp.setZero((buffer_length_ - 1) * num_joints_);
                    temp = ob_double_.segment(pos + num_joints_, num_joints_);
                    ob_double_.segment(pos + num_joints_, num_joints_) = p_target_.tail(num_joints_) - gc.tail(num_joints_);
                    ob_double_.segment(pos, num_joints_ * (buffer_length_ - 1)) = temp;
                    pos += num_joints_ * buffer_length_;

                    // joint velocity history
                    temp = ob_double_.segment(pos + num_joints_, num_joints_ * (buffer_length_ - 1));
                    ob_double_.segment(pos + num_joints_ * (buffer_length_ - 1), num_joints_) = gv.tail(num_joints_);
                    ob_double_.segment(pos, num_joints_ * (buffer_length_ - 1)) = temp;
                    pos += num_joints_ * buffer_length_;

                    // body velocity error history
                    temp.setZero(3 * (buffer_length_ - 1));
                    temp = ob_double_.segment(pos + 3, 3 * (buffer_length_ - 1));

                    Eigen::Vector3d body_vel_err = target_velocity_ - body_linear_vel_;
                    body_vel_err(2) = target_velocity_(2) - body_angular_vel_(2);
                    ob_double_.segment(pos + 3 * (buffer_length_ - 1), 3) = body_vel_err;
                    ob_double_.segment(pos, 3 * (buffer_length_ - 1)) = temp;
                    pos += 3 * buffer_length_;
                }
                step_++;

                ob_scaled_ = (ob_double_ - ob_mean_).cwiseQuotient(ob_std_);
            }

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

            Eigen::Vector3d target_velocity_,
                            body_linear_vel_,
                            body_angular_vel_;

            int num_joints_;

            std::vector<double> initial_joint_positions_,
                                initial_joint_velocities_;

            int buffer_length_ = 1,
                step = -1;

            double  body_height_, 
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
                std::string ee_frame_names[4] = {"front_left_ee", "front_right_ee", "rear_left_ee", "rear_right_ee"};
            };

            GaitParams  gait_params_,
                        default_gait_params_,
                        stance_gait_params_;

            std::set<size_t> foot_indices_;
    };

}