// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

// Class for working with the camera in Ignition Gazebo.
// User must provide a path to the camera xacro
// A camera object is instantiated when the user calles world_->insertCamera()

#include "Model.hpp"

namespace gym_ignition {

class Camera : public Model {

    public:

        explicit Camera(
            scenario::core::ModelPtr robotPtr,
            const std::string cameraName, /* e.g. kinect0, camera1, realsense2, ... */
            const YAML::Node& cfg,
            const std::string xacroPath,
            const std::string urdfPath, /* name of URDF, created in same directory as xacro */ 
            const std::string resourceDir, 
            const std::unordered_map<std::string, std::any>& kwargs = {} /* kwargs-like dictionary of additional arguments to pass to xacro command */) :
            Model(robotPtr, urdfPath) {

            hasRgb_ = cfg["camera"]["has_rgb"].template as<bool>();
            hasDepth_ = cfg["camera"]["has_depth"].template as<bool>();
            hasLogical_ = cfg["camera"]["has_logical"].template as<bool>();
            hasThermal_ = cfg["camera"]["has_thermal"].template as<bool>();

            if(!hasRgb_) {
                newRgbCamObsAvailable_ = true;
            } else {
                rgbImageWidth_ = cfg["camera"]["rgb"]["resolution"]["w"].template as<int>();
                rgbImageHeight_ = cfg["camera"]["rgb"]["resolution"]["h"].template as<int>();
                rgbImageMsg_.height = rgbImageHeight_;
                rgbImageMsg_.width = rgbImageWidth_;
                rgbImageMsg_.data.resize(rgbImageHeight_ * rgbImageWidth_ * 3, 0);
            }

            if(!hasDepth_) {
                newDepthCamObsAvailable_ = true;
            } else {
                depthImageWidth_ = cfg["camera"]["depth"]["resolution"]["w"].template as<int>();
                depthImageHeight_ = cfg["camera"]["depth"]["resolution"]["h"].template as<int>();
                depthImageMsg_.height = depthImageHeight_;
                depthImageMsg_.width = depthImageWidth_;
                depthImageMsg_.data.resize(depthImageHeight_ * depthImageWidth_, 0);
            }

            if(!hasLogical_)
                newLogicalCamObsAvailable_ = true;

            if(!hasThermal_) {
                newThermalCamObsAvailable_ = true;
            } else {
                thermalImageWidth_ = cfg["camera"]["depth"]["resolution"]["w"].template as<int>();
                thermalImageHeight_ = cfg["camera"]["depth"]["resolution"]["h"].template as<int>();
                thermalLinearResolution_ = cfg["camera"]["thermal"]["linear_resolution"].template as<double>();
                thermalImageMsg_.height = thermalImageHeight_;
                thermalImageMsg_.width = thermalImageWidth_;
                thermalImageMsg_.data.resize(thermalImageHeight_ * thermalImageWidth_, 0);
            }

            createSubscriptions_(hasRgb_, hasDepth_, hasLogical_, hasThermal_, cameraName);
        }

        ~Camera() { }

        int rgbImageWidth() {
            return rgbImageWidth_;
        }

        int rgbImageHeight() {
            return rgbImageHeight_;
        }

        int depthImageWidth() {
            return depthImageWidth_;
        }

        int depthImageHeight() {
            return depthImageHeight_;
        }

        int thermalImageWidth() {
            return thermalImageWidth_;
        }

        int thermalImageHeight() {
            return thermalImageHeight_;
        }

        bool hasRgb() {
            return hasRgb_;
        }

        bool hasDepth() {
            return hasDepth_;
        }

        bool hasLogical() {
            return hasLogical_;
        }

        bool hasThermal() {
            return hasThermal_;
        }

        void getRgbCameraObservation(gym_ignition::msg::Rgb& img) {

            if(newRgbCamObsAvailable_)
                newRgbCamObsAvailable_ = false;

            img = rgbImageMsg_;
        }

        void getDepthCameraObservation(gym_ignition::msg::Depth& img) {

            if(newDepthCamObsAvailable_)
                newDepthCamObsAvailable_ = false;

            // TODO: figure out how to deal with nan values
            img = depthImageMsg_;
        }

        void getThermalCameraObservation(gym_ignition::msg::Thermal& img) {

            if(newThermalCamObsAvailable_)
                newThermalCamObsAvailable_ = false;

            img = thermalImageMsg_;
        }

        std::unordered_map<std::string, ModelPose> getLogicalCameraObservation(const std::string modelName = {}) {

            if(!modelName.empty()){
                std::unordered_map<std::string, ModelPose> requestedModelPose;
                for(auto it : modelPoses_) {
                    if((it.first).find(modelName) == 0){
                        requestedModelPose[modelName] = it.second;
                        break;
                    }
                }

                if(requestedModelPose.empty()){
                    std::cerr << "[Camera.hpp] Model named `" << modelName << "` not found!" << std::endl;
                } else {
                    if(newLogicalCamObsAvailable_)
                        newLogicalCamObsAvailable_ = false;

                    return requestedModelPose;
                }
            }

            if(newLogicalCamObsAvailable_)
                newLogicalCamObsAvailable_ = false;

            return modelPoses_;
        }

        bool newCameraObservationAvailable(bool getNewCameraObs = false) {

            if(getNewCameraObs && !getNewCameraObs_) {
                getNewCameraObs_ = true;

                if(hasRgb_)
                    newRgbCamObsAvailable_ = false;

                if(hasDepth_)
                    newDepthCamObsAvailable_ = false;

                if(hasLogical_)
                    newLogicalCamObsAvailable_ = false;

                if(hasThermal_)
                    newThermalCamObsAvailable_ = false;
            }

            if(newRgbCamObsAvailable_ && newDepthCamObsAvailable_ && newLogicalCamObsAvailable_ && newThermalCamObsAvailable_) {
                getNewCameraObs_ = false;
                return true;
            }

            return false;
        }

        // bool waitForNewCameraObservation() {
        //     // Throw away the current camera observation and wait for a new one. Return true when one is ready.

        //     if(hasRgb_ && newRgbCamObsAvailable_)
        //         newRgbCamObsAvailable_ = false;

        //     if(hasDepth_ && newDepthCamObsAvailable_)
        //         newDepthCamObsAvailable_ = false;

        //     if(hasLogical_ && newLogicalCamObsAvailable_)
        //         newLogicalCamObsAvailable_ = false;

        //     if(hasThermal_ && newThermalCamObsAvailable_)
        //         newThermalCamObsAvailable_ = false;

        // }

        static void generateUrdfFromXacro(
            const std::string& cameraName,
            const std::string& xacroPath,
            const std::string& urdfPath,
            const std::string& resourceDir,
            const std::unordered_map<std::string, std::any>& kwargs) {
            // NOTE: The xacro should have at least 2 arguments - `config_file`, i.e. the path to the YAML config, and `camera_name`. You can specify additional arguments as strings (for the moment, only strings are supported as of C++17 for std::any) that will be passed to the xacro command upon generation of your urdf.
            
            std::string cmdStr = fmt::format("xacro {0}", xacroPath);
            if(!kwargs.empty()){
                for(auto it : kwargs) {
                    cmdStr = fmt::format(cmdStr + " " + it.first + ":={0}", std::any_cast<std::string>(it.second));
                }
            }
            cmdStr = fmt::format(cmdStr +  " config_file:={0} camera_name:={1} > {2}", resourceDir, cameraName, urdfPath);
            const char* command = cmdStr.c_str();

            std::cout << "[Camera.hpp] Creating URDF for `" << cameraName << "`" << std::endl;
            std::cout << "[Camera.hpp] " << cmdStr << std::endl;
            system(command);
        }

    private:

        void createSubscriptions_(
            bool hasRgb, 
            bool hasDepth, 
            bool hasLogical,
            bool hasThermal,
            const std::string& cameraName) {

            if(hasRgb) {
                if(!ignitionNode_.Subscribe("/" + cameraName + "/rgb", &Camera::rgbCameraCallback_, this)) {
                    std::cerr << "[Camera.hpp] Error subscribing to the RGB camera topic." << std::endl;
                }
            }

            if(hasDepth) {
                if(!ignitionNode_.Subscribe("/" + cameraName + "/depth", &Camera::depthCameraCallback_, this)) {
                    std::cerr << "[Camera.hpp] Error subscribing to the depth camera topic." << std::endl;
                }
            }

            if(hasLogical) {
                if(!ignitionNode_.Subscribe("/" + cameraName + "/logical", &Camera::logicalCameraCallback_, this)) {
                    std::cerr << "[Camera.hpp] Error subscribing to the logical camera topic." << std::endl;
                }
            }

            if(hasThermal) {
                if(!ignitionNode_.Subscribe("/" + cameraName + "/thermal", &Camera::thermalCameraCallback_, this)) {
                    std::cerr << "[Camera.hpp] Error subscribing to the thermal camera topic." << std::endl;
                }
            }

        }

        void rgbCameraCallback_(
            const ignition::msgs::Image& msg) {
            
            memcpy(rgbImageMsg_.data.data(), msg.data().c_str(), rgbImageHeight_ * rgbImageWidth_ * 3 * sizeof(uint8_t));

            newRgbCamObsAvailable_ = true;
        }

        void depthCameraCallback_(
            const ignition::msgs::Image& msg) {
            
            float f;
            memcpy(depthImageMsg_.data.data(), msg.data().c_str(), depthImageHeight_ * depthImageWidth_ * sizeof(f));
            
            newDepthCamObsAvailable_ = true;
        }

        void logicalCameraCallback_(
            const ignition::msgs::LogicalCameraImage& msg) {
            
            for(size_t i = 0; i < msg.model_size(); i++) {
                ModelPose pose = {};
                pose.position = Eigen::Vector3d( /* position */
                        msg.model(i).pose().position().x(), 
                        msg.model(i).pose().position().y(), 
                        msg.model(i).pose().position().z());
                pose.orientation = Eigen::Quaterniond( /* orientation */
                        msg.model(i).pose().orientation().w(),
                        msg.model(i).pose().orientation().x(), 
                        msg.model(i).pose().orientation().y(), 
                        msg.model(i).pose().orientation().z());
                modelPoses_[msg.model(i).name()] = pose;
            }

            newLogicalCamObsAvailable_ = true;
        }

        void thermalCameraCallback_(
            const ignition::msgs::Image& msg) {
            
            memcpy(thermalImageMsg_.data.data(), msg.data().c_str(), thermalImageHeight_ * thermalImageWidth_ * sizeof(uint16_t));

            newThermalCamObsAvailable_ = true;
        }

        ignition::transport::Node ignitionNode_;
        gym_ignition::msg::Rgb rgbImageMsg_;
        gym_ignition::msg::Depth depthImageMsg_;
        gym_ignition::msg::Thermal thermalImageMsg_;

        std::unordered_map<std::string, ModelPose> modelPoses_;

        bool hasRgb_ = false;
        bool hasDepth_ = false;
        bool hasLogical_ = false;
        bool hasThermal_ = false;

        bool newRgbCamObsAvailable_ = false;
        bool newDepthCamObsAvailable_ = false;
        bool newLogicalCamObsAvailable_ = false;
        bool newThermalCamObsAvailable_ = false;
        bool getNewCameraObs_ = false;

        int rgbImageWidth_ = 0;
        int rgbImageHeight_ = 0;
        int depthImageWidth_ = 0;
        int depthImageHeight_ = 0;
        int thermalImageWidth_ = 0;
        int thermalImageHeight_ = 0;
        int thermalLinearResolution_ = 0.01; // 10 mK
};

}