// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

/*
This environment implements the simple cartpole problem. It is meant to demonstrate the basic functionality of gym-ignition and how to set-up an MDP.
*/

#include "GymIgnitionEnv.hpp"

namespace gym_ignition {

class ENVIRONMENT : public GymIgnitionEnv {

    public:

        ENVIRONMENT(
            const std::string& resourceDir, 
            const YAML::Node& cfg, 
            int envIndex,
            bool render) :
            GymIgnitionEnv(resourceDir, cfg, envIndex, render),
            distribution_(-1.0, 1.0),
            uniform_dist_(-1.0, 1.0),
            rand_num_gen_(std::chrono::system_clock::now().time_since_epoch().count()) {

            // IMPORTANT! Set the simulation timestep and the RL controller timestep
            setSimulationTimeStep(cfg_["step_size"].template as<double>());
            setControlTimeStep(cfg_["control_dt"].template as<double>());

            // Insert world
            world_->insertWorld(get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/worlds/empty.world", "empty" + std::to_string(envIndex));

            // Insert ground plane
            world_->insertGround(get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/gym_ignition_models/ground_plane/ground_plane.sdf", "ground" + std::to_string(envIndex), /*enable contacts=*/true);

            // Open the GUI, if you enabled visualizing the training when launching the simulation
            if(visualizable_)
                world_->openGazeboGui(5);

            cartpole_ = world_->insertModel(
                get_cwd() + "install/gym_ignition_description/share/gym_ignition_description/gym_ignition_models/cartpole/cartpole.urdf",
                "cartpole" + std::to_string(envIndex),
                0., 0., 0.,
                0., 0., 0.);

            // use force control to cartpole joint
            cartpole_->setJointControlMode(scenario::core::JointControlMode::Force);

            // specify joint EFFORT targets for actuators
            rlControlCallbackPtr_ = [this]() { cartpole_->setJointEffortTargets(pTarget_); };

            actionDim_  = 1;    // cart force
            obsDim_     = 1 +   // x position
                          1 +   // pivot angle
                          1 +   // x velocity
                          1;    // pivot angular rate

            obScaled_.setZero(obsDim_);
            obDouble_.setZero(obsDim_);
            obMean_.setZero(obsDim_);
            obStd_.setZero(obsDim_);
            actionUnscaled_.setZero(actionDim_);
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            pTarget_.setZero(2); // 1. linear (prismatic) joint for the cart, 2. continuous (revolute) joint for the pole pivot

            obMean_ << 0., 0., 0., 0.;
            obStd_ << 3., 2*M_PI, 0.1, 10;
            actionMean_ << 0.;
            actionStd_ << 100.;

            cartpole_->printJointNames();
        }

        void init() final { }

        void reset() final {
            
            std::vector<double> initialJointPositions = {0., 0.};
            std::vector<double> initialJointVelocities = {0., 0.};
            cartpole_->resetJointStates(initialJointPositions, initialJointVelocities);

            // NOTE: Any time you call this method, it will advance the simulation 1 step so that you can get a new observation
            world_->pauseGazebo();

            updateObservation_();
            updateExtraInfo_();
        }

        float step(
            const Eigen::VectorXd& action) final {
            
            // Unscale the action from the RL controller
            actionUnscaled_ = action;
            actionUnscaled_ = actionUnscaled_.cwiseProduct(actionStd_) + actionMean_;

            for(int i = 0; i < int(controlDt_ / simulationDt_ + 1e-10); i++) {
                pTarget_[0] = actionUnscaled_[0]; // only apply a force to the cart and not to the pole's pivot
                
                world_->integrate(1, rlControlCallbackPtr_);
            }
            
            updateObservation_();

            return getReward_();
        }

        bool isTerminalState(
            float& terminalReward) final {
            terminalReward = 0.;
            
            if(std::abs(cartPosition_) >= 2.) {
                terminalReward = -rewardCoeffs_["terminal"];
                return true;
            }

            if(std::abs(poleAngle_) >= M_PI/4.) {
                terminalReward = -rewardCoeffs_["terminal"];
                return true;
            }

            return false;
        }

        void getExtraInfo(
            std::unordered_map<std::string, double>& extraInfo,
            gym_ignition::msg::Rgb& rgbImg,
            gym_ignition::msg::Depth& depthImg,
            gym_ignition::msg::Thermal& thermalImg) {

            updateExtraInfo_();
            extraInfo = extraInfo_;
        }

        void setSeed(
            int seed) final {

        }

        void observe(
            Eigen::VectorXd& ob) final {
            
            ob = obScaled_;
        }

    private:

        void updateExtraInfo_() {
            
            extraInfo_["cart_position"] = cartPosition_;
            extraInfo_["pole_angle"] = poleAngle_;
            extraInfo_["cart_velocity"] = cartVelocity_;
            extraInfo_["pole_velocity"] = poleVelocity_;
            extraInfo_["cart_force"] = pTarget_[0];
        }

        void updateObservation_() {

            cartPosition_ = cartpole_->generalizedCoordinates()[7];
            poleAngle_ = cartpole_->generalizedCoordinates()[8];
            cartVelocity_ = cartpole_->generalizedVelocities()[6];
            poleVelocity_ = cartpole_->generalizedVelocities()[7];

            int pos = 0;

            obDouble_[pos++] = cartPosition_;
            obDouble_[pos++] = poleAngle_;
            obDouble_[pos++] = cartVelocity_;
            obDouble_[pos++] = poleVelocity_;

            obScaled_ = (obDouble_ - obMean_).cwiseQuotient(obStd_);

            // Double check to make sure no NaNs in observation
            checkValid(obScaled_, "observation");
        }

        float getReward_() {
            
            auto cartForce = cartpole_->generalizedForces()[6];
            rewards_["force"] = -rewardCoeffs_["force"] * cartForce * cartForce;

            rewards_["angle"] = rewardCoeffs_["angle"] * std::exp(-100. * poleAngle_ * poleAngle_);

            return rewards_.sum();
        }

        std::mt19937 rand_num_gen_;
        std::normal_distribution<double> distribution_;
        std::uniform_real_distribution<double> uniform_dist_;

        std::unique_ptr<Model> cartpole_;

        Eigen::VectorXd obDouble_, obScaled_, actionUnscaled_, actionMean_, actionStd_, obMean_, obStd_, pTarget_;

        double cartPosition_, poleAngle_, cartVelocity_, poleVelocity_;

        std::unordered_map<std::string, double> extraInfo_;
};

}