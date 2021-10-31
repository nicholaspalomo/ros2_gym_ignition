// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

#include "Model.hpp"

namespace gym_ignition{

// This class is used for creating an instance of a robot in simulation that has a camera fixed rigidly to one of the frames of the robot (e.g. to the end effector/hand/head) 
class RobotCamera {

    public:

        explicit RobotCamera(
            std::unique_ptr<Kinematics>& robotKinematics,
            std::unique_ptr<Camera>& robotCamera) :
            robotKinematics_(std::move(robotKinematics)),
            robotCamera_(std::move(robotCamera))
        {}

        int numJoints() {
            return robotKinematics_->numJoints();
        }

        int printJointNames() {
            robotKinematics_->printJointNames();
        }
    
        void resetJointStates(
            std::vector<double>& jointPositions,
            std::vector<double>& jointVelocities) {
            
            robotKinematics_->resetJointStates(jointPositions, jointVelocities);
        }

        void setJointControllerPeriod(
            const double controllerDt) { 
            
            robotKinematics_->setJointControllerPeriod(controllerDt);
        }

        void setJointControlMode(
            const scenario::core::JointControlMode mode){
            
            robotKinematics_->setJointControlMode(mode);
        }

        void setJointPidGains(
            const YAML::Node& pid_node) {
            
            robotKinematics_->setJointPidGains(pid_node);
        }

        void setJointPositionTargets(
            std::vector<double>& targets) {
            
            robotKinematics_->setJointPositionTargets(targets);
        }

        void setJointPositionTargets(
            Eigen::Ref<Eigen::VectorXd> positions) {
            
            robotKinematics_->setJointPositionTargets(positions);
        }

        void setJointEffortTargets(
            std::vector<double>& targets) {
            
            robotKinematics_->setJointEffortTargets(targets);
        }

        void setJointEffortTargets(
            Eigen::Ref<Eigen::VectorXd> forces) {
            
            robotKinematics_->setJointEffortTargets(forces);
        }

        void setJointVelocityTargets(
            std::vector<double>& targets) {
            
            robotKinematics_->setJointVelocityTargets(targets);
        }

        void setJointVelocityTargets(
            Eigen::Ref<Eigen::VectorXd> velocities) {

            robotKinematics_->setJointVelocityTargets(velocities);
        }

        void addFrameTarget(
            const std::string targetFrame,
            Eigen::Vector3d& targetPositionInBaseFrame,
            Eigen::Vector3d& targetRpyInBaseFrame) {
            
            robotKinematics_->addFrameTarget(targetFrame, targetPositionInBaseFrame, targetRpyInBaseFrame);
        }

        void addFrameTarget(
            const std::string targetFrame,
            Eigen::Vector3d& targetPositionInBaseFrame,
            Eigen::Quaterniond& targetQuatRotToBaseFrame) {
            
            robotKinematics_->addFrameTarget(targetFrame, targetPositionInBaseFrame, targetQuatRotToBaseFrame);
        }

        void addFrameTarget(
            const std::string targetFrame,
            Eigen::Vector3d& targetPositionInBaseFrame,
            Eigen::Matrix<double, 3, 3>& rotMatTarget2Base) {
            
            robotKinematics_->addFrameTarget(targetFrame, targetPositionInBaseFrame, rotMatTarget2Base);
        }

        Eigen::VectorXd solveIk(
            Eigen::VectorXd jointPositions = {}) {
            
            return robotKinematics_->solveIk(jointPositions);
        }

        Eigen::Vector3d getFramePositionInWorld(
            const std::string frameName) {
            
            return robotKinematics_->getFramePositionInWorld(frameName);
        }

        Eigen::Quaterniond getRotationFrame2World(
            const std::string frameName) {
            
            return robotKinematics_->getRotationFrame2World(frameName);
        }

        Eigen::VectorXd generalizedCoordinates() {

            return robotKinematics_->generalizedCoordinates();
        }

        Eigen::VectorXd generalizedVelocities() {

            return robotKinematics_->generalizedVelocities();
        }

        Eigen::VectorXd generalizedForces() {

            return robotKinematics_->generalizedForces();
        }

        Eigen::Matrix<double, 6, 1> getFrameVelocity(
            const std::string frameName,
            const std::string targetFrameName = {}) {
            
            return robotKinematics_->getFrameVelocity(frameName, targetFrameName);
        }

        Eigen::Matrix<double, 6, 1> getFrameForces(
            const std::string frameName,
            const std::string targetFrameName = {}) {

            return robotKinematics_->getFrameForces(frameName, targetFrameName);
        }

        Eigen::Vector3d getFramePosition_(
            const std::string frameName) {

            return robotKinematics_->getFramePosition(frameName);
        }

        Eigen::Vector3d getFramePosition(
            const std::string frameName,
            const std::string targetFrameName = {}) {

            return robotKinematics_->getFramePosition(frameName, targetFrameName);
        }

        std::vector<scenario::core::Contact> getLinkContacts(
            const std::string linkName) {
            
            return robotKinematics_->getLinkContacts(linkName);
        }

        std::vector<std::string> getLinkNames(
            const bool scoped = false) {
            
            return robotKinematics_->getLinkNames(scoped);
        }

    private:

        std::unique_ptr<Kinematics> robotKinematics_;
        std::unique_ptr<Camera> robotCamera_;
};

}