// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

#ifndef SRC_GYMIGNITIONMODEL_HPP
#define SRC_GYMIGNITIONMODEL_HPP

namespace gym_ignition{

class Model {

    public:

        explicit Model(
            scenario::core::ModelPtr robot,
            const std::string urdfPath,
            std::vector<std::string> joints = {}) :
            robot_(robot),
            urdfPath_(urdfPath),
            jointSerialization_(joints) {

            robotGazebo_ = std::static_pointer_cast<scenario::gazebo::Model>(robot_);

            // Maintain a history of the joint generalized forces. See:
            // https://github.com/robotology/gym-ignition/pull/141
            // https://github.com/robotology/gym-ignition/discussions/372
            for(auto jointName : robot_->jointNames())
                robot_->getJoint(jointName)->enableHistoryOfAppliedJointForces(true, 1);

            // Enable contact detection for all links in model
            enableContacts(true);
        }

        static std::vector<std::string> readJointSerializationFromYaml(
            const YAML::Node& node) {

            std::vector<std::string> joints;
            for(auto it = node.begin(); it != node.end(); it++)
                joints.push_back(it->as<std::string>());

            return joints;
        }

        void printJointNames() {

            auto jointNames = robot_->jointNames();
            for(auto jointName : jointNames)
                std::cout << jointName << std::endl;
        }

        std::vector<std::string> jointNames() {
            return robot_->jointNames();
        }

        void enableContacts(
            bool enable) { 
            robot_->enableContacts(/*enable=*/enable);
            robotGazebo_->enableSelfCollisions(enable); 
        }

        void resetJointStates(
            std::vector<double>& jointPositions,
            std::vector<double>& jointVelocities) {
                
            // Reset the joint positions and velocities to the initial values
            robotGazebo_->resetJointPositions(jointPositions, robot_->jointNames());
            robotGazebo_->resetJointVelocities(jointVelocities, robot_->jointNames());
        }

        void setJointControllerPeriod(
            const double controllerDt) { 

            robot_->setControllerPeriod(controllerDt);
        }

        void setJointControlMode(
            const scenario::core::JointControlMode mode){

            robot_->setJointControlMode(mode, robot_->jointNames());
        }

        void setJointPidGains(
            const YAML::Node& pid_node) {
            // Set the PID gains for the joint controllers

            for(auto jointName : robot_->jointNames()) {
                auto jointPtr = robot_->getJoint(jointName);
                scenario::core::PID pid_gains(
                    pid_node[jointName]["p"].template as<double>(),
                    pid_node[jointName]["i"].template as<double>(),
                    pid_node[jointName]["d"].template as<double>()
                );
                bool success = jointPtr->setPID(pid_gains);
                
                if(!success){ std::cout << "[Model.hpp] Failed to set PID gains for joint: " << jointName << std::endl; throw; }
            }
        }

        void setJointPositionTargets(
            std::vector<double>& targets) {

            bool success = robot_->setJointPositionTargets(targets, robot_->jointNames());
            if(!success){ std::cout << "[Model.hpp] Failed to set joint position targets!" << std::endl; throw; }
        }

        void setJointPositionTargets(
            Eigen::Ref<Eigen::VectorXd> positions) {

            std::vector<double> targets = eigenVec2StdVec<double>(positions);
            setJointPositionTargets(targets);
        }

        void setJointEffortTargets(
            std::vector<double>& targets) {

            bool success = robot_->setJointGeneralizedForceTargets(targets, robot_->jointNames());
            if(!success){ std::cout << "[Model.hpp] Failed to set joint effort targets!" << std::endl; throw; }
        }

        void setJointEffortTargets(
            Eigen::Ref<Eigen::VectorXd> forces) {

                std::vector<double> targets = eigenVec2StdVec<double>(forces);
                setJointEffortTargets(targets);
        }

        void setJointVelocityTargets(
            std::vector<double>& targets) {
            
            bool success = robot_->setJointVelocityTargets(targets, robot_->jointNames());
            if(!success){ std::cout << "[Model.hpp] Failed to set joint velocity targets!" << std::endl; throw; }
        }

        void setJointVelocityTargets(
            Eigen::Ref<Eigen::VectorXd> velocities) {

                std::vector<double> targets = eigenVec2StdVec<double>(velocities);
                setJointVelocityTargets(targets);
        }

        int generalizedCoordinateDim() {

            return 7 + robot_->nrOfJoints();
        }

        int generalizedVelocityDim() {

            return 6 + robot_->nrOfJoints();
        }

        Eigen::VectorXd generalizedCoordinates() {

            Eigen::VectorXd gc;
            gc.setZero(generalizedCoordinateDim());

            int pos = 0;
            for(auto basePosition : robot_->basePosition())
                gc[pos++] = basePosition;

            for(auto baseOrientation : robot_->baseOrientation())
                gc[pos++] = baseOrientation;

            for(auto jointPosition : robot_->jointPositions())
                gc[pos++] = jointPosition;

            return gc;
        }

        Eigen::VectorXd generalizedVelocities() {

            Eigen::VectorXd gv;
            gv.setZero(generalizedVelocityDim());
            
            int pos = 0;
            for(auto baseLinVel : robot_->baseWorldLinearVelocity())
                gv[pos++] = baseLinVel;

            for(auto baseAngVel : robot_->baseWorldAngularVelocity())
                gv[pos++] = baseAngVel;
                
            for(auto jointVelocity : robot_->jointVelocities())
                gv[pos++] = jointVelocity;

            return gv;
        }

        Eigen::VectorXd generalizedForces() {

            Eigen::VectorXd gf;
            gf.setZero(generalizedVelocityDim());
            auto baseLink = robot_->getLink(robot_->baseFrame());

            int pos = 0;
            for(auto generalizedForce : baseLink->contactWrench())
                gf[pos++] = generalizedForce;

            for(auto jointName : robot_->jointNames())
                gf[pos++] = robot_->getJoint(jointName)->historyOfAppliedJointForces()[0];

            return gf;
        }

        Eigen::Matrix<double, 6, 1> getFrameForces(
            const std::string frameName,
            const std::string targetFrameName = {}) {
            
            auto contactWrench = robot_->getLink(frameName)->contactWrench();
            if(targetFrameName.empty())
                return Eigen::Map<Eigen::Matrix<double, 6, 1>>(&contactWrench[0], 6);

            Eigen::Matrix<double, 3, 3> rotMatFrame2Target = getRotationFrame2Target(frameName, targetFrameName).toRotationMatrix();

            Eigen::Matrix<double, 6, 1> frameForces = Eigen::Map<Eigen::Matrix<double, 6, 1>>(&contactWrench[0], 6);

            frameForces.segment<3>(0) = rotMatFrame2Target * frameForces.segment<3>(0);
            frameForces.segment<3>(3) = rotMatFrame2Target * frameForces.segment<3>(3);

            return frameForces;
        }

        int numJoints() { 
            
            return robot_->nrOfJoints();
        }

        int numJointsIk() {

            return jointSerialization_.size() > 0 ? jointSerialization_.size() : numJoints();
        }

        Eigen::Vector3d getFramePositionInWorld(
            const std::string frameName) {

            return Eigen::Map<Eigen::Matrix<double, 3, 1>>(&(robot_->getLink(frameName)->position()[0]), 3);
        }

        Eigen::Quaterniond getRotationFrame2World(
            const std::string frameName) {

            // This method returns the quaternion from BODY to WORLD frame!
            auto frameOrientation = robot_->getLink(frameName)->orientation();
            Eigen::Quaterniond orientation(
                frameOrientation[0], /* w */
                frameOrientation[1], /* x */
                frameOrientation[2], /* y */
                frameOrientation[3]);/* z */

            return orientation.normalized(); // Note that in Eigen, the order of the coefficients is [w, x, y, z]!
        }

        Eigen::Matrix<double, 6, 1> getFrameVelocity(
            const std::string frameName,
            const std::string targetFrameName = {}) {

            auto link = robot_->getLink(frameName);
            
            Eigen::Matrix<double, 6, 1> frameVelocityInTargetFrame;
            if(targetFrameName.empty()) {
            // If no target frame specified, get the body velocity in the global coordinate system
                auto worldLinVel = link->worldLinearVelocity();
                auto worldAngVel = link->worldAngularVelocity();
                
                frameVelocityInTargetFrame.segment<3>(0) = Eigen::Map<Eigen::Matrix<double, 3, 1>>(&worldLinVel[0], 3);
                frameVelocityInTargetFrame.segment<3>(3) = Eigen::Map<Eigen::Matrix<double, 3, 1>>(&worldAngVel[0], 3);

            } else if(frameName.compare(targetFrameName) == 0) {
            // If target frame matches the given frame, get the body velocity in the body's coordinate frame
                auto bodyLinVel = link->bodyLinearVelocity();
                auto bodyAngVel = link->bodyAngularVelocity();

                frameVelocityInTargetFrame.segment<3>(0) = Eigen::Map<Eigen::Matrix<double, 3, 1>>(&bodyLinVel[0], 3);
                frameVelocityInTargetFrame.segment<3>(3) = Eigen::Map<Eigen::Matrix<double, 3, 1>>(&bodyAngVel[0], 3);

            } else {
            // Otherwise, get the body velocity specified in the target frame

                Eigen::Matrix<double, 3, 3> rotMatFrame2Target = getRotationFrame2Target(frameName, targetFrameName).toRotationMatrix();
                auto bodyLinVel = link->bodyLinearVelocity();
                auto bodyAngVel = link->bodyAngularVelocity();

                frameVelocityInTargetFrame.segment<3>(0) = rotMatFrame2Target * Eigen::Map<Eigen::Matrix<double, 3, 1>>(&bodyLinVel[0], 3);
                frameVelocityInTargetFrame.segment<3>(3) = rotMatFrame2Target * Eigen::Map<Eigen::Matrix<double, 3, 1>>(&bodyAngVel[0], 3);
            }

            return frameVelocityInTargetFrame;
        }

        Eigen::Vector3d getFramePosition(
            const std::string frameName,
            const std::string targetFrameName = {}) {
            
            if(targetFrameName.empty())
                return Eigen::Vector3d(robot_->basePosition().data());

            auto framePositionInWorldFrame = getFramePosition_(frameName);
            auto targetFramePositionInWorldFrame = getFramePosition_(targetFrameName);

            auto rotMatTargetFrame2World = getRotationFrame2World(targetFrameName).toRotationMatrix();

            return rotMatTargetFrame2World.transpose() * framePositionInWorldFrame - targetFramePositionInWorldFrame;
        }

        Eigen::Quaterniond getRotationFrame2Target(
            const std::string frameName,
            const std::string targetFrameName = {}) {

            if(targetFrameName.empty())
                return Eigen::Quaterniond(robot_->baseOrientation().data());

            auto rotMatFrame2World = getRotationFrame2World(frameName).toRotationMatrix();

            auto rotMatTargetFrame2World = getRotationFrame2World(targetFrameName).toRotationMatrix();

            Eigen::Quaterniond quatFrame2Target(rotMatTargetFrame2World.transpose() * rotMatFrame2World);

            return quatFrame2Target;
        }

        std::vector<scenario::core::Contact> getLinkContacts(
            const std::string linkName) {
            // Return a map of the bodies in contact with the given link and the pose of the contact point
            
            return robot_->getLink(linkName)->contacts();
        }

        void resetBasePose(
            const double x, const double y, const double z,
            const double R, const double P, const double Y) {
            
            std::array<double, 3> position = {x, y, z};
            Eigen::Quaterniond quat = rpy2quat(R, P, Y);
            std::array<double, 4> orientation = {quat.w(), quat.x(), quat.y(), quat.z()};
            
            if(!robotGazebo_->resetBasePose(position, orientation)) { std::cout << "[Model.hpp] Failed to reset base position!" << std::endl; throw; }

            std::array<double, 3> velocity = {0., 0., 0.};
            if(!robotGazebo_->resetBaseWorldLinearVelocity(velocity)) { std::cout << "[Model.hpp] Failed to reset base linear velocity!" << std::endl; throw; }

            if(!robotGazebo_->resetBaseWorldAngularVelocity(velocity)) { std::cout << "[Model.hpp] Failed to reset base angular velocity!" << std::endl; throw; }
        }

        std::vector<std::string> getLinkNames(
            const bool scoped = false) {

            return robot_->linkNames(scoped);
        }

        uint8_t getLinkIndexFromName(const std::string linkName, const bool scoped = false) {

            auto linkNames = getLinkNames(scoped);
            auto linkIt = std::find(linkNames.begin(), linkNames.end(), linkName);

            if(linkIt != linkNames.end()) {
                return linkIt - linkNames.begin();
            }else{
                return -1;
            }

        }

        bool inContact(
            const std::string linkName) {

            return robot_->getLink(linkName)->inContact();
        }

    protected:

        scenario::core::ModelPtr robot_;
        std::shared_ptr<scenario::gazebo::Model> robotGazebo_;

        std::vector<std::string> jointSerialization_;
        std::string urdfPath_;

        std::vector<ContactPoint> populateContactInfo_(
            std::vector<scenario::core::ContactPoint> contactScenario) {
            
            std::vector<ContactPoint> contacts;
            for(auto contact : contactScenario) {
                ContactPoint contactPoint;
                contactPoint.depth = contact.depth;
                contactPoint.force = Eigen::Vector3d({contact.force[0], contact.force[1], contact.force[2]});
                contactPoint.torque = Eigen::Vector3d({contact.torque[0], contact.torque[1], contact.torque[2]});
                contactPoint.normal = Eigen::Vector3d({contact.normal[0], contact.normal[1], contact.normal[2]});
                contactPoint.position = Eigen::Vector3d({contact.position[0], contact.position[1], contact.position[2]});

                contacts.push_back(contactPoint);
            }

            return contacts;
        }

        Eigen::Vector3d getFramePosition_(
            const std::string frameName) {

            auto framePosition = robot_->getLink(frameName)->position();
            Eigen::Vector3d position({framePosition[0], framePosition[1], framePosition[2]});

            return position;
        }
};

}

#endif // SRC_GYMIGNITIONMODEL_HPP