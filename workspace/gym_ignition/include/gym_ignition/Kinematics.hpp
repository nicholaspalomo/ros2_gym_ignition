// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

// This class implements generic inverse and forward kinematics operations for articulated systems whose URDF follow a tree-like structure (i.e. no loops in the linkages). It is meant to be used as a member of the Kinematics class.

#ifndef SRC_GYMIGNITION_KINEMATICS_HPP
#define SRC_GYMIGNITION_KINEMATICS_HPP

// iDynTree headers
#include <iDynTree/KinDynComputations.h>
#include <iDynTree/ModelIO/ModelLoader.h>
#include <iDynTree/InverseKinematics.h>
#include <iDynTree/Model/FreeFloatingState.h>
#include <iDynTree/Estimation/GravityCompensationHelpers.h>

#include "Model.hpp"

namespace gym_ignition{

class Kinematics : public Model {

    public:

        explicit Kinematics(
            scenario::core::ModelPtr robot,
            const std::string urdfPath,
            std::vector<std::string> joints = {},
            const std::string baseConstraint = "floating",
            const std::string baseFrame = "base_link",
            const std::vector<double> nominalJointPositions = {}) :
            Model(robot, urdfPath, joints),
            baseConstraint_(baseConstraint),
            baseFrame_(baseFrame),
            fk_(std::make_unique<iDynTree::KinDynComputations>()),
            ik_(std::make_unique<iDynTree::InverseKinematics>()),
            gravComp_(std::make_unique<iDynTree::GravityCompensationHelper>()),
            genBiasForce_(std::make_unique<iDynTree::FreeFloatingGeneralizedTorques>()) {

            bool success = false;
            if(!jointSerialization_.empty()) {
                success = modelLoader_.loadReducedModelFromFile(urdfPath_, jointSerialization_);

                // Populate the joint indexes based on the serialization (important for inverse kinematics)
                size_t k = 0;
                for(size_t i = 0; i < robot_->jointNames().size(); i++) {
                    for(size_t j = k; j < jointSerialization_.size(); j++) {
                        if((robot_->jointNames()[i]).find(jointSerialization_[k]) != std::string::npos) {
                            jointSerializationIndexes_.push_back(i);
                            k++;
                            break;
                        }
                    }
                }
                std::sort(jointSerializationIndexes_.begin(), jointSerializationIndexes_.end());

            } else {
                success = modelLoader_.loadModelFromFile(urdfPath_); 

                // Populate the joint indexes based on the serialization (important for inverse kinematics)
                for(size_t i = 0; i < robot_->jointNames().size(); i++) {
                    jointSerializationIndexes_.push_back(i);
                }
            }

            if(!success) { std::cerr << "[Kinematics.hpp] Failed to load model." << std::endl; }

            if(!fk_->loadRobotModel(modelLoader_.model())) { std::cerr << "[Kinematics.hpp] Failed to create forward kinematics object." << std::endl; }

            // Everything in Model.hpp assumes an inertial-fixed velocity representation, meaning that the base and link velocities are by default given in the inertial frame, unless otherwise specified.
            fk_->setFrameVelocityRepresentation(iDynTree::FrameVelocityRepresentation::INERTIAL_FIXED_REPRESENTATION);

            if(!ik_->setModel(modelLoader_.model())){ std::cerr << "[Kinematics.hpp] Failed to create inverse kinematics object." << std::endl; }

            // Parameterize targets as a constraint, as opposed to a cost
            ik_->setDefaultTargetResolutionMode(iDynTree::InverseKinematicsTreatTargetAsConstraintFull);

            // Set the default cost, constraint tolerances to something reasonable; otherwise IK will probably fail to reach a solution more often than not
            ik_->setCostTolerance(1e-3);
            ik_->setConstraintsTolerance(1e-3);

            // Set the rotation parameterization for the IK model
            ik_->setRotationParametrization(iDynTree::InverseKinematicsRotationParametrizationRollPitchYaw);

            // We want to add the base pose as a constraint if we're dealing with a fixed-based robot
            if(baseConstraint.compare("fixed") == 0) {
                basePoseOptimized_ = iDynTree::Transform::Identity();
                success = ik_->setFloatingBaseOnFrameNamed(baseFrame_);

                if(!success) { std::cerr << "[Kinematics.hpp] Failed to set fixed-base constraint for robot. " << std::endl; }

                ik_->addFrameConstraint(baseFrame_, basePoseOptimized_);

                std::cout << "[Kinematics.hpp] Fixed base robot" << std::endl;
            } else if(baseConstraint.compare("floating") == 0) {
                std::cout << "[Kinematics.hpp] Floating base robot" << std::endl;
            } else {
                std::cerr << "[Kinematics.hpp] Base constraint not recognized. Valid base constraints are `fixed` and `floating`." << std::endl;
            }

            // Set the gravity compensation object
            gravComp_->loadModel(modelLoader_.model(), baseFrame_);
            genBiasForce_->resize(modelLoader_.model());

            // Initialize previous IK solution to 0 joint angles
            previousIkSolution_.setZero(numJointsIk());
        }

        void addFrameTarget(
            const std::string targetFrame,
            Eigen::Vector3d& targetPositionInBaseFrame,
            Eigen::Vector3d& targetRpyInBaseFrame) {

            Eigen::Matrix<double, 3, 3> rotMatTarget2Base = rpy2RotMat<double>(targetRpyInBaseFrame[0], targetRpyInBaseFrame[1], targetRpyInBaseFrame[2]);
            addFrameTarget(targetFrame, targetPositionInBaseFrame, rotMatTarget2Base);
        }

        void addFrameTarget(
            const std::string targetFrame,
            Eigen::Vector3d& targetPositionInBaseFrame,
            Eigen::Quaterniond& targetQuatRotToBaseFrame) {
            
            Eigen::Matrix<double, 3, 3> rotMatTarget2Base = targetQuatRotToBaseFrame.toRotationMatrix();
            addFrameTarget(targetFrame, targetPositionInBaseFrame, rotMatTarget2Base);
        }

        void addFrameTarget(
            const std::string targetFrame,
            Eigen::Vector3d& targetPositionInBaseFrame,
            Eigen::Matrix<double, 3, 3>& rotMatTarget2Base) {
            
            // Initialize the IK solution to the current joint positions
            for(auto i : jointSerializationIndexes_){
                previousIkSolution_[i] = generalizedCoordinates().tail(numJoints())[i];
            }

            Eigen::Quaterniond baseQuaternion(
                generalizedCoordinates().segment(3, 4)[0],
                generalizedCoordinates().segment(3, 4)[1], 
                generalizedCoordinates().segment(3, 4)[2], 
                generalizedCoordinates().segment(3, 4)[3]);

            setCurrentRobotConfiguration(
                generalizedCoordinates().head(3),
                baseQuaternion,
                previousIkSolution_);

            iDynTree::Position position_base2Target(
                targetPositionInBaseFrame[0], 
                targetPositionInBaseFrame[1],
                targetPositionInBaseFrame[2]);

            iDynTree::Rotation rotation_target2Base(
                rotMatTarget2Base(0, 0), rotMatTarget2Base(0, 1), rotMatTarget2Base(0, 2),
                rotMatTarget2Base(1, 0), rotMatTarget2Base(1, 1), rotMatTarget2Base(1, 2),
                rotMatTarget2Base(2, 0), rotMatTarget2Base(2, 1), rotMatTarget2Base(2, 2));           

            iDynTree::Transform T_base2target = iDynTree::Transform(rotation_target2Base, position_base2Target); // rotation given as full rotation matrix

            if(firstCall_){
                firstCall_ = false;
                iDynTree::Transform transform = iDynTree::Transform::Identity();
                if(!ik_->addTarget(targetFrame, transform))
                    std::cout << "[Kinematics.hpp] Failed to set target frame transform for inverse kinematics model." << std::endl;
                // TODO: Enable setting the target constraint resolution separately for target frame. See documentation for `setTargetResolutionMode`.
            }

            if(!ik_->updateTarget(targetFrame, T_base2target))
                std::cout << "[Kinematics.hpp] Failed to update target frame transform for inverse kinematics model." << std::endl;

        }

        Eigen::VectorXd solveIk(
            Eigen::VectorXd jointPositions = {}) {

            // Warm start for next solver call
            if(jointPositions.size() != 0){
                iDynTree::VectorDynSize initialJointPositions(numJointsIk());
                for(auto ind : jointSerializationIndexes_)
                    initialJointPositions[ind] = jointPositions[ind];
                ik_->setFullJointsInitialCondition(&basePoseOptimized_, &(initialJointPositions));
            }else{
                iDynTree::VectorDynSize previousIkSolution(previousIkSolution_.data(), numJointsIk());
                ik_->setFullJointsInitialCondition(&basePoseOptimized_, &(previousIkSolution));
            }

            // Solve the IK
            bool success;
            success = ik_->solve();

            if(!success){
                std::cout << "[Kinematics.hpp] WARNING: IK failed to converge." << std::endl;
            }else{
                iDynTree::VectorDynSize jointPositionTargets(numJointsIk());
                ik_->getReducedSolution(basePoseOptimized_, jointPositionTargets);
                previousIkSolution_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(jointPositionTargets.data(), numJointsIk());
            }

            return previousIkSolution_;
        }

        Eigen::Vector3d getFramePositionFromFk(
            const std::string baseFrameName,
            const std::string targetFrameName,
            const Eigen::VectorXd jointPositions = {}) {

            auto T_base2target = getRelativeTransform_(baseFrameName, targetFrameName, jointPositions);

            iDynTree::Position pos_targetInBaseFrame = T_base2target.getPosition();

            return Eigen::Vector3d(pos_targetInBaseFrame.data());
        }

        void resetKinematics(
            Eigen::VectorXd jointPositions) {
            
            basePoseOptimized_ = iDynTree::Transform::Identity();
            for(auto i : jointSerializationIndexes_){
                previousIkSolution_[i] = generalizedCoordinates().tail(numJoints())[i];
            }
        }

        Eigen::Quaterniond getFrameOrientationFromFkQuat(
            const std::string baseFrameName,
            const std::string targetFrameName,
            const Eigen::VectorXd jointPositions = {}) {

            iDynTree::VectorDynSize currentJointPositions = currentSerializedJointPositions_();
            fk_->setJointPos(currentJointPositions);

            auto T_base2target = getRelativeTransform_(baseFrameName, targetFrameName, jointPositions);

            iDynTree::Vector4 rotation_targetInBaseFrame = T_base2target.getRotation().asQuaternion();

            return Eigen::Quaterniond(rotation_targetInBaseFrame.data());
        }

        Eigen::Matrix<double, 3, 3> getFrameOrientationFromFkRotMat(
            const std::string baseFrameName,
            const std::string targetFrameName,
            const Eigen::VectorXd jointPositions = {}) {
            
            return getFrameOrientationFromFkQuat(baseFrameName, targetFrameName, jointPositions).normalized().toRotationMatrix();
        }

        Eigen::Vector3d getFrameOrientationFromFkRpy(
            const std::string baseFrameName,
            const std::string targetFrameName,
            const Eigen::VectorXd jointPositions = {}) {

            Eigen::Quaterniond quat = getFrameOrientationFromFkQuat(baseFrameName, targetFrameName, jointPositions);

            return quat2rpy<double>(quat.w(), quat.x(), quat.y(), quat.z());
        }

        Eigen::VectorXd getJointTorquesFromInverseDynamics(
            Eigen::Matrix<double, 6, 1>& baseAcceleration, /* acceleration of the base frame */
            Eigen::VectorXd& jointAccelerations, /* joint accelerations */
            Eigen::VectorXd linkExternalForces = {}) { /* external forces on the links - note: the inverseDynamics method already takes care of gravity! */

            updateStateFromModel_();
            
            // Solve for the joint torques using inverse dynamics
            iDynTree::FreeFloatingGeneralizedTorques baseForceAndJointTorquesIdyn(modelLoader_.model());
            iDynTree::Vector6 baseAcc(baseAcceleration.data(), 6);
            iDynTree::LinkNetExternalWrenches linkExtForces(modelLoader_.model());
            iDynTree::VectorDynSize jointAcc(jointAccelerations.data(), numJointsIk());
            auto success = fk_->inverseDynamics(
                baseAcc,
                jointAcc,
                linkExtForces,
                baseForceAndJointTorquesIdyn);
            
            if(success)
                return Eigen::Map<Eigen::MatrixXd>(baseForceAndJointTorquesIdyn.jointTorques().data(), numJointsIk(), 1);

            std::cerr << "[Kinematics.hpp] Failed to calculate the inverse dynamics for the robot!" << std::endl;
        }

        void updateRobotStateFromModel() {
            updateStateFromModel_();
        }

        Eigen::MatrixXd getFreeFloatingMassMatrix() {

            iDynTree::MatrixDynSize M(numJointsIk()+6, numJointsIk()+6);
            fk_->getFreeFloatingMassMatrix(M);

            return iDynTree::toEigen(M);

        }

        Eigen::VectorXd getGravityCompensationTorques() {
            
            fk_->generalizedBiasForces(*genBiasForce_);

            iDynTree::JointDOFsDoubleArray& jointTorques = genBiasForce_->jointTorques();

            return Eigen::Map<Eigen::VectorXd>(jointTorques.data(), numJointsIk(), 1);

        }

        Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor> getFrameJacobian(
            const std::string jacobianFrameName) {
            
            auto J = getFrameJacobianIDynTree_(jacobianFrameName);

            return Eigen::Map<Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor>>(J.data(), 6, 6 + numJointsIk());
        }

        Eigen::Matrix<double, 6, 1> getFrameTwistFromFk(
            const std::string targetFrameName) {

            auto J_joints2target = getFrameJacobianIDynTree_(targetFrameName);

            int dofs = 6 + numJointsIk();
            iDynTree::VectorDynSize velocities(dofs);
            for(size_t i = 0; i < dofs; i++){
                if(i > 6) {
                    velocities[i] = generalizedVelocities()[jointSerializationIndexes_[i - 6]];
                } else {
                    velocities[i] = generalizedVelocities()[i];
                }

            }

            return getFrameTwistFromJacAndJointVel_(J_joints2target, velocities);
        }

        void updateFrameTransformConstraint(
            const std::string frameName,
            iDynTree::Position& position,
            iDynTree::Vector4& quaternion) {

            if(!ik_->isFrameConstraintActive(frameName)) {
                std::cout << "[Kinematics.hpp] Constraint on frame " << frameName << " not active." << std::endl;
            }

            iDynTree::Rotation orientation;
            orientation.fromQuaternion(quaternion);

            iDynTree::Transform transform = iDynTree::Transform(orientation, position); // rotation given as full rotation matrix

            if(!ik_->activateFrameConstraint(frameName, transform)) {
                std::cout << "[Kinematics.hpp] Failed to update constraint on frame " << frameName << std::endl;
            }
        }

        void setCurrentRobotConfiguration(
            Eigen::Vector3d basePosition,
            Eigen::Quaterniond baseQuaternion,
            Eigen::VectorXd jointConfiguration) {

            iDynTree::Position basePositionIdyn(basePosition[0], basePosition[1], basePosition[2]);
            
            iDynTree::Vector4 baseQuaternionIdyn;
            baseQuaternionIdyn[0] = baseQuaternion.w();
            baseQuaternionIdyn[1] = baseQuaternion.x();
            baseQuaternionIdyn[2] = baseQuaternion.y();
            baseQuaternionIdyn[3] = baseQuaternion.z();
            iDynTree::Rotation baseOrientationIdyn;
            baseOrientationIdyn.fromQuaternion(baseQuaternionIdyn);

            iDynTree::Transform baseTransform = iDynTree::Transform(baseOrientationIdyn, basePositionIdyn); // rotation given as full rotation matrix

            iDynTree::VectorDynSize jointConfigurationIdyn(jointConfiguration.data(), jointConfiguration.size());

            if(!ik_->setCurrentRobotConfiguration(baseTransform, jointConfigurationIdyn)) {
                std::cout << "[Kinematics.hpp] Failed to set the robot configuration." << std::endl;
            }

            if(baseConstraint_.compare("fixed") == 0) {
                updateFrameTransformConstraint(baseFrame_, basePositionIdyn, baseQuaternionIdyn);
            }
        }

    private:

        void updateStateFromModel_() {

            Eigen::VectorXd jointPositionsSerialized;
            jointPositionsSerialized.setZero(numJointsIk());
            Eigen::VectorXd jointVelocitiesSerialized;
            jointVelocitiesSerialized.setZero(numJointsIk());
            for(size_t i = 0; i < numJointsIk(); i++) {
                jointPositionsSerialized[i] = generalizedCoordinates().segment(7, numJoints())[jointSerializationIndexes_[i]];
                jointVelocitiesSerialized[i] = generalizedVelocities().segment(6, numJoints())[jointSerializationIndexes_[i]];
            }

            // Set the robot state
            Eigen::Quaterniond baseQuaternion(
                generalizedCoordinates()[3],
                generalizedCoordinates()[4],
                generalizedCoordinates()[5],
                generalizedCoordinates()[6]);

            Eigen::Matrix<double, 3, 3> rotMatBase2World = baseQuaternion.toRotationMatrix();

            iDynTree::Position basePosition(
                generalizedCoordinates()[0], 
                generalizedCoordinates()[1],
                generalizedCoordinates()[2]);

            iDynTree::Rotation rotMatWorld2Base(
                rotMatBase2World(0, 0), rotMatBase2World(1, 0), rotMatBase2World(2, 0),
                rotMatBase2World(0, 1), rotMatBase2World(1, 1), rotMatBase2World(2, 1),
                rotMatBase2World(0, 2), rotMatBase2World(1, 2), rotMatBase2World(2, 2));           

            iDynTree::Transform transform_world2base(rotMatWorld2Base, basePosition); // rotation given as full rotation matrix

            iDynTree::LinVelocity baseLinVel(generalizedVelocities()[0], generalizedVelocities()[1], generalizedVelocities()[2]);
            iDynTree::AngVelocity baseAngVel(generalizedVelocities()[3], generalizedVelocities()[4], generalizedVelocities()[5]);

            iDynTree::Twist baseVelocity(baseLinVel, baseAngVel);
            iDynTree::VectorDynSize jointPositions(jointPositionsSerialized.data(), numJointsIk());
            iDynTree::VectorDynSize jointVelocities(jointVelocitiesSerialized.data(), numJointsIk());
            iDynTree::Vector3 gravity;
            gravity.zero();
            gravity[2] = -9.80665;

            fk_->setRobotState(
                    transform_world2base,
                    jointPositions,
                    baseVelocity,
                    jointVelocities,
                    gravity);

        }

        iDynTree::MatrixDynSize getFrameJacobianIDynTree_(
            const std::string frameName) {

            int dofs = 6 + numJointsIk();
            iDynTree::MatrixDynSize J_joints2target(6, dofs);
            fk_->getFrameFreeFloatingJacobian(frameName, J_joints2target);
            
            return J_joints2target;
        }

        iDynTree::VectorDynSize currentSerializedJointPositions_() {

            iDynTree::VectorDynSize currentJointPositions(numJointsIk());
            // Get the subset of the generalized coordinates corresponding to the provided joint serialization
            for(size_t i = 0; i < numJointsIk(); i++) {
                currentJointPositions[i] = generalizedCoordinates().tail(numJoints())(jointSerializationIndexes_[i]);
            }

            return currentJointPositions;
        }

        Eigen::Matrix<double, 6, 1> getFrameTwistFromJacAndJointVel_(
            const iDynTree::MatrixDynSize& J,
            const iDynTree::VectorDynSize& v) {

            Eigen::Matrix<double, 6, 1> frameTwist;
            for(size_t i = 0; i < 6; i++) {
                for(size_t j = 0; j < (6 + numJointsIk()); j++) {
                    frameTwist(i) += J.getVal(i, j) * v[j];
                }
            }

            return frameTwist;
        }

        iDynTree::Transform getRelativeTransform_(
            const std::string baseFrameName,
            const std::string targetFrameName,
            const Eigen::VectorXd jointPositions = {}) {
            
            iDynTree::VectorDynSize jointPositionsIdyn(numJointsIk());
            jointPositionsIdyn.zero();
            if(jointPositions.size() == 0) {
                jointPositionsIdyn.fillBuffer(currentSerializedJointPositions_().data());
            }else{
                //  TODO: Find a better way to copy the data than using for-loops
                for(size_t i = 0; i < numJointsIk(); i++) {
                    jointPositionsIdyn[i] = jointPositions(jointSerializationIndexes_[i]);
                }
            }
            
            fk_->setJointPos(jointPositionsIdyn);

            return fk_->getRelativeTransform(baseFrameName, targetFrameName);
        }

        bool firstCall_ = true;
        const std::string baseConstraint_;
        const std::string baseFrame_;
        std::vector<size_t> jointSerializationIndexes_;

        Eigen::VectorXd previousIkSolution_;
        iDynTree::Transform basePoseOptimized_;

        iDynTree::ModelLoader modelLoader_;
        std::unique_ptr<iDynTree::KinDynComputations> fk_;
        std::unique_ptr<iDynTree::InverseKinematics> ik_;
        std::unique_ptr<iDynTree::GravityCompensationHelper> gravComp_;
        std::unique_ptr<iDynTree::FreeFloatingGeneralizedTorques> genBiasForce_;

};

}

#endif //SRC_GYMIGNITION_KINEMATICS_HPP 