// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

#ifndef SRC_GYMIGNITIONENV_HPP
#define SRC_GYMIGNITIONENV_HPP

#include <stdlib.h>
#include <filesystem> // for path separators and such
#include <fmt/core.h> // https://fmt.dev/latest/contents.html
#include <cstdint>
#include <set>
#include <random> // for random number generator
#include <chrono>
#include <string>
#include <thread>
#include <future>
#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry> // to get quaternions, rotation matrices and whatnot
#include "yaml-cpp/yaml.h"
#include <sdf/sdf.hh>

#include <scenario/gazebo/GazeboSimulator.h>
#include <scenario/gazebo/Joint.h>
#include <scenario/gazebo/Model.h>
#include <scenario/gazebo/World.h>
#include <scenario/controllers/Controller.h>
#include <scenario/controllers/ComputedTorqueFixedBase.h>
#include <scenario/gazebo/utils.h>
#include <scenario/gazebo/helpers.h>

#include <ignition/msgs.hh>
#include <ignition/transport.hh>
#include <ignition/gazebo/ServerConfig.hh>

#include "rclcpp/rclcpp.hpp"
#include "gym_ignition/srv/info.hpp"
#include "gym_ignition/srv/step.hpp"
#include "gym_ignition/srv/reset.hpp"
#include "gym_ignition/srv/observe.hpp"
#include "gym_ignition/srv/extra.hpp"
#include "gym_ignition/msg/rgb.hpp"
#include "gym_ignition/msg/depth.hpp"
#include "gym_ignition/msg/thermal.hpp"

namespace gym_ignition {

    /* Generic struct to hold position and orientation information */
    struct ModelPose {
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
    };

    struct ContactPoint {
        double depth;
        Eigen::Vector3d force;
        Eigen::Vector3d torque;
        Eigen::Vector3d normal;
        Eigen::Vector3d position;
    };

    /* Helper functions */
    // Get the current working directory as a string
    inline std::string get_cwd(void) {
        char buff[FILENAME_MAX];
        getcwd(buff, FILENAME_MAX);
        std::string current_working_dir(buff);
        return current_working_dir + std::filesystem::path::preferred_separator;
    }

    // TODO: Separate helper functions into separate Math.hpp header
    template<class T>
    inline void checkValid(Eigen::Matrix<T, Eigen::Dynamic, 1>& vec, const std::string vecName) {

        if (std::isnan(vec.norm())) {
            std::cout << "NaN in "  << vecName << std::endl;
            std::cout << vec.transpose() << std::endl;
            throw;
        }
    }

    template<class T>
    inline std::vector<T> eigenVec2StdVec(Eigen::Matrix<T, Eigen::Dynamic, 1> eigen) {
        std::vector<T> vec(eigen.data(), eigen.data() + eigen.size());
        return vec;
    }

    template<class T>
    inline Eigen::VectorXd stdVec2EigenVec(std::vector<T> vec) {
        return Eigen::VectorXd::Map(&vec[0], vec.size());
    }

    template<class T> 
    inline Eigen::Matrix<T, 3, 1> rotMat2Rpy(const Eigen::Matrix<T, 3, 3> rotMat) {

        return (Eigen::Matrix<T, 3, 1>() << 
            std::atan2(rotMat(2, 1), rotMat(2, 2)),
            std::atan2(-rotMat(2, 0), sqrt(rotMat(2, 1)*rotMat(2, 1) + rotMat(2, 2)*rotMat(2, 2))),
            std::atan2(rotMat(1, 0), rotMat(0, 0))).finished();
    }

    template<class T>
    inline Eigen::Matrix<T, 3, 1> quat2rpy(const T w, const T x, const T y, const T z){

        Eigen::Matrix<T, 3, 1> rpy;
        rpy[0] = static_cast<T>(std::atan2(2.0 * (x*y + w*z), w*w + x*x - y*y - z*z)); // roll
        rpy[1] = static_cast<T>(std::asin(-2.0 * (x*z - w*y))); // pitch
        rpy[2] = static_cast<T>(std::atan2(2.0*(y*z + w*x), w*w - x*x - y*y + z*z)); // yaw

        return rpy;
    }

    template<class T>
    inline Eigen::Quaternion<T> rpy2quat(const T R, const T P, const T Y) {
        // Abbreviations for the various angular functions
        double cy = cos(Y * 0.5);
        double sy = sin(Y * 0.5);
        double cp = cos(P * 0.5);
        double sp = sin(P * 0.5);
        double cr = cos(R * 0.5);
        double sr = sin(R * 0.5);

        return Eigen::Quaternion<T>({
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy});
    }

    template Eigen::Matrix<T, 3, 3> quat2RotMat(Eigen::Quaternion<T> quat) {

        return quat.normalized().toRotationMatrix();
    }

    template<class T>
    inline Eigen::Matrix<T, 3, 3> rpy2RotMat(const T R, const T P, const T Y) {

        return rpy2quat<T>(R, P, Y).toRotationMatrix();
    }

    /* Typedefs */
    typedef std::function<void(void)> controlClbkPtr;

    using EigenRowMajorMat=Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
    using EigenVec=Eigen::Matrix<float, -1, 1>;
    using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;
}

#define __IGN_MAKE_STR(x) #x
#define _IGN_MAKE_STR(x) __IGN_MAKE_STR(x)
#define IGN_MAKE_STR(x) _IGN_MAKE_STR(x)

#define READ_YAML(a, b, c) RSFATAL_IF(!c, "Node "<<IGN_MAKE_STR(c)<<" doesn't exist") \
                           b = c.as<a>();

#include "World.hpp"
#include "Reward.hpp"

namespace gym_ignition {
class GymIgnitionEnv : public rclcpp::Node {

    public:

        GymIgnitionEnv(
            const std::string& resourceDir,
            const YAML::Node& cfg,
            const int envIndex,
            const bool render) :
            Node("env" + std::to_string(envIndex)),
            envId_(envIndex),
            visualizable_(render),
            resourceDir_(std::move(resourceDir)),
            cfg_(cfg),
            world_(std::make_unique<World>(cfg_["step_size"].template as<double>(), cfg_["rtf"].template as<double>(), cfg_["steps_per_run"].template as<size_t>())) {

            // read the reward names and coefficients from the YAML config
            rewardCoeffs_.readFromYaml(cfg_["reward"]);
            rewards_ = rewardCoeffs_;

            // Create the services
            services_.stepSrv = this->create_service<gym_ignition::srv::Step>("/env" + std::to_string(envIndex) + "/step", std::bind(&GymIgnitionEnv::step_, this, std::placeholders::_1, std::placeholders::_2));

            services_.resetSrv = this->create_service<gym_ignition::srv::Reset>("/env" + std::to_string(envIndex) + "/reset", std::bind(&GymIgnitionEnv::reset_, this, std::placeholders::_1, std::placeholders::_2));

            services_.infoSrv = this->create_service<gym_ignition::srv::Info>("/env" + std::to_string(envIndex) + "/info", std::bind(&GymIgnitionEnv::info_, this, std::placeholders::_1, std::placeholders::_2));

            services_.observeSrv = this->create_service<gym_ignition::srv::Observe>("/env" + std::to_string(envIndex) + "/observe", std::bind(&GymIgnitionEnv::observe_, this, std::placeholders::_1, std::placeholders::_2));

            services_.extraSrv = this->create_service<gym_ignition::srv::Extra>("/env" + std::to_string(envIndex) + "/extra", std::bind(&GymIgnitionEnv::extra_, this, std::placeholders::_1, std::placeholders::_2));
        }

        virtual ~GymIgnitionEnv() = default;

        //// Implement the following methods as part of the MDP ////
        virtual void init() = 0;
        virtual void reset() = 0;
        virtual void setSeed(int seed) = 0;
        virtual void observe(Eigen::VectorXd& ob) = 0;
        virtual float step(const Eigen::VectorXd& action) = 0;
        virtual bool isTerminalState(float& terminalReward) = 0;
        ////////////////////////////////////////////////////////////

        //// Option methods to implement ////
        virtual void close() {};
        virtual void getExtraInfo(std::unordered_map<std::string, double>& extraInfo, gym_ignition::msg::Rgb& rgbImg, gym_ignition::msg::Depth& depthImg, gym_ignition::msg::Thermal& thermalImg) {};
        virtual void getCameraObs(gym_ignition::msg::Rgb& rgbImg, gym_ignition::msg::Depth& depthImg, gym_ignition::msg::Thermal& thermalImg) {};
        /////////////////////////////////////

        int getObsDim() { return obsDim_; }
        int getActionDim() { return actionDim_; }
        int getExtraInfoDim() { extraInfoDim_ = extraInfo_.size(); return extraInfoDim_; }

        void turnOnVisualization() { visualizeThisStep_ = true; }
        void turnOffVisualization() { visualizeThisStep_ = false; }
        void setControlTimeStep(const double dt) { 
            controlDt_ = dt;
            maxEpisodeSteps_ = int(cfg_["max_time"].template as<double>() / controlDt_);
        }
        double getControlTimeStep() { return controlDt_; }

        void setSimulationTimeStep(const double dt){
            world_->setSimulationTimeStep(dt);

            simulationDt_ = world_->getSimulationTimeStep();
        }

        double getSimulationTimeStep() { return simulationDt_; }

        void zeroRewards() { rewards_.setZero(); }

    protected:
        reward rewards_;
        reward rewardCoeffs_;

        std::string resourceDir_;
        YAML::Node cfg_;

        std::unique_ptr<World> world_;
        controlClbkPtr rlControlCallbackPtr_;

        int obsDim_ = 0, actionDim_ = 0, extraInfoDim_ = 0;

        double simulationDt_ = 0.001, controlDt_ = 0.001;
        int maxEpisodeSteps_ = 1, currentEpisodeStep_ = 0;
        bool visualizeThisStep_ = false;
        bool visualizable_ = false;

    private:

        void step_(
            const std::shared_ptr<gym_ignition::srv::Step::Request> request, 
            std::shared_ptr<gym_ignition::srv::Step::Response> response) {
            
            zeroRewards();

            action_.setZero(getActionDim());
            action_ = stdVec2EigenVec<double>(request->action);

            response->reward = step(action_);

            observation_.setZero(getObsDim());
            observe(observation_);
            
            auto terminalReward = 0.0f;
            response->is_done = isTerminalState(terminalReward);

            response->reward += terminalReward;

            getCameraObs(
                response->rgb,
                response->depth,
                response->thermal);

            response->observation = eigenVec2StdVec(observation_);
            
            currentEpisodeStep_++;
            if(response->is_done || (int(currentEpisodeStep_ % maxEpisodeSteps_) == 0)){
                reset();
                currentEpisodeStep_ = 0;
            }
            
            response->env_id = envId_;
        }

        void reset_(
            const std::shared_ptr<gym_ignition::srv::Reset::Request> request, 
            std::shared_ptr<gym_ignition::srv::Reset::Response> response) {
            
            reset();
            
            observation_.setZero(getObsDim());
            observe(observation_);

            getCameraObs(
                response->rgb,
                response->depth,
                response->thermal);
            
            response->observation = eigenVec2StdVec(observation_);
            response->env_id = envId_;
        }

        void info_(
            const std::shared_ptr<gym_ignition::srv::Info::Request> request, 
            std::shared_ptr<gym_ignition::srv::Info::Response> response) {
            
            if(request->sim_time_step > 1e-6)
                setSimulationTimeStep(request->sim_time_step);

            if(request->control_time_step > 1e-6)
                setControlTimeStep(request->control_time_step);

            response->num_obs = getObsDim();
            response->num_extras = getExtraInfoDim();
            response->num_acts = getActionDim();
            response->num_envs = cfg_["num_envs"].template as<int>();
            response->visualizable = visualizable_;
            response->env_id = envId_;
        }

        void observe_(
            const std::shared_ptr<gym_ignition::srv::Observe::Request> request,
            std::shared_ptr<gym_ignition::srv::Observe::Response> response) {
            
            observation_.setZero(getObsDim());
            observe(observation_);
            response->observation = eigenVec2StdVec(observation_);

            getCameraObs(
                response->rgb,
                response->depth,
                response->thermal);

            response->env_id = envId_;
        }

        void extra_(
            const std::shared_ptr<gym_ignition::srv::Extra::Request> request,
            std::shared_ptr<gym_ignition::srv::Extra::Response> response) {

            getExtraInfo(extraInfo_, response->rgb, response->depth, response->thermal);
            
            for(const auto& elem : extraInfo_) {
                response->extra_info_keys.push_back(elem.first);
                response->extra_info.push_back(elem.second);
            }

            for(const auto& elem : rewards_.rewardTerms_) {
                response->reward_keys.push_back(elem.first);
                response->reward.push_back(elem.second);
            }

            response->env_id = envId_;
        }

        struct srvStruct{
            rclcpp::Service<gym_ignition::srv::Step>::SharedPtr stepSrv; // callback: step_
            rclcpp::Service<gym_ignition::srv::Reset>::SharedPtr resetSrv; // callback: reset_
            rclcpp::Service<gym_ignition::srv::Info>::SharedPtr infoSrv; // callback: info_
            rclcpp::Service<gym_ignition::srv::Extra>::SharedPtr extraSrv; // callback: extra_
            rclcpp::Service<gym_ignition::srv::Observe>::SharedPtr observeSrv; // callback: observe_
        };

        srvStruct services_;
        int envId_;

        Eigen::VectorXd observation_;
        Eigen::VectorXd action_;
        std::unordered_map<std::string, double> extraInfo_;
};

}

#endif // SRC_GYMIGNITIONENV_HPP
