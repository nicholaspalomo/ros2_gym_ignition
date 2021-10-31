// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

#ifndef SRC_GYMIGNITIONWORLD_HPP
#define SRC_GYMIGNITIONWORLD_HPP

#include "Kinematics.hpp"
#include "Camera.hpp"
#include "RobotCamera.hpp"

namespace gym_ignition{

class World {

    public:

        explicit World(
            double stepSize_,
            double rtf,
            size_t stepsPerRun) :
            gazebo_(std::make_unique<scenario::gazebo::GazeboSimulator>(stepSize_, rtf, stepsPerRun)) {

            simulationDt_ = stepSize_;
            rtf_ = rtf;
            stepsPerRun_ = stepsPerRun;
        }

        ~World() {
            gazebo_->close();
        }

        void insertWorld(
            const std::string modelPath,
            const std::string modelName) { 

            gazebo_->insertWorldFromSDF(modelPath, modelName); 

            // Initialize the simulator
            gazebo_->initialize();

            // Get pointer to world object
            world_ = gazebo_->getWorld(modelName);

            // Set the pysics engine
            world_->setPhysicsEngine(scenario::gazebo::PhysicsEngine::Dart);

            // Set gravity vector
            std::array<double, 3> gravity({0., 0., -9.80665});
            world_->setGravity(gravity);
        }

        void integrate1() {
            // Integrate the physics a single timestep
            gazebo_->run();
        }

        void integrate(
            double seconds) {
            
            // Integrate the simulation for specified number of seconds
            for(size_t i = 0; i < int(seconds / simulationDt_ + 1e-10); i++) {
                gazebo_->run();
            }
        }

        void integrate(
            int n_steps,
            controlClbkPtr callback) {

            // Integrate the simulation for n_steps number of time steps
            callback();
            for(size_t i = 0; i < n_steps; i++) {
                gazebo_->run();
            }
        }

        std::unique_ptr<Model> insertModel(
            const std::string modelPath,
            const std::string modelName,
            const double x, const double y, const double z,
            const double R, const double P, const double Y) {
            
            return insertObject(modelPath, modelName, x, y, z, R, P, Y);
        }

        std::unique_ptr<Model> insertObject(
            const std::string modelPath,
            const std::string modelName,
            const double x, const double y, const double z,
            const double R, const double P, const double Y) {

            auto pose = xyzRPY2ScenarioPose(x, y, z, R, P, Y);

            insertModel_(modelPath, modelName, pose);

            return std::make_unique<Model>(
                    world_->getModel(/*modelName=*/modelName),
                    modelPath);
        }

        std::unique_ptr<RobotCamera> insertRobotWithCamera(
            const std::string modelPath,
            const std::string modelName,
            const YAML::Node& cameraYamlCfg,
            const std::string xacroPath,
            const std::string urdfPath,
            const std::string resourceDir,
            const double x, const double y, const double z,
            const double R, const double P, const double Y, 
            std::vector<std::string> joints = {},
            const std::string baseConstraint = "floating",
            const std::string baseFrame = "base_link",
            const std::unordered_map<std::string, std::any>& kwargs = {}) {

            createCameraUrdf_(
                modelName,
                cameraYamlCfg,
                xacroPath,
                urdfPath,
                resourceDir,
                kwargs);

            auto robotKinematics = insertRobot(
                    modelPath,
                    modelName,
                    x, y, z,
                    R, P, Y,
                    joints,
                    baseConstraint,
                    baseFrame);

            auto robotCamera = std::make_unique<Camera>(
                    world_->getModel(/*modelName=*/modelName),
                    modelName,
                    cameraYamlCfg,
                    xacroPath,
                    urdfPath,
                    resourceDir,
                    kwargs);

            return std::make_unique<RobotCamera>(
                robotKinematics, 
                robotCamera);
        }

        std::unique_ptr<Kinematics> insertRobot(
            const std::string modelPath,
            const std::string modelName,
            const double x, const double y, const double z,
            const double R, const double P, const double Y, 
            std::vector<std::string> joints = {},
            const std::string baseConstraint = "floating",
            const std::string baseFrame = "base_link") {

            auto pose = xyzRPY2ScenarioPose(x, y, z, R, P, Y);

            insertModel_(modelPath, modelName, pose);

            return getModel(modelPath, modelName, joints, baseConstraint, baseFrame);
        }

        void insertWorldPlugin(
            const std::string& libName,
            const std::string& className,
            const std::string& context = {}) {

            if(!world_->insertWorldPlugin(libName, className, context)){ std::cout << "[World.hpp] Failed to insert plugin: " << libName << " " << className << std::endl; throw; }
        }

        void insertGround(
            const std::string modelPath,
            const std::string modelName,
            bool enable) {

            scenario::core::Pose pose(
                std::array<double, 3>({0., 0., 0.}),
                std::array<double, 4>({1.0, 0., 0., 0.}));

            insertModel_(modelPath, modelName, pose);

            ground_ = world_->getModel(/*modelName=*/modelName);
            ground_->enableContacts(/*enable=*/enable);
            groundGazebo_ = std::static_pointer_cast<scenario::gazebo::Model>(ground_);
            groundGazebo_->enableSelfCollisions(enable);
        }

        std::unique_ptr<Camera> insertCamera(
            const std::string modelName,
            const YAML::Node& cameraYamlCfg,
            const std::string xacroPath,
            const std::string urdfPath,
            const std::string resourceDir,
            const Eigen::Vector3d& position = Eigen::Vector3d({0., 0., 0.}),
            const Eigen::Vector3d& rpy = Eigen::Vector3d({0., 0., 0.}),
            const std::unordered_map<std::string, std::any>& kwargs = {}) {

            createCameraUrdf_(
                modelName,
                cameraYamlCfg,
                xacroPath,
                urdfPath,
                resourceDir,
                kwargs);

            Eigen::Quaterniond quat = rpy2quat(rpy[0], rpy[1], rpy[2]);
            scenario::core::Pose pose(
                std::array<double, 3>({position[0], position[1], position[2]}),
                std::array<double, 4>({quat.w(), quat.x(), quat.y(), quat.z()}));

            insertModel_(urdfPath, modelName, pose);

            return std::make_unique<Camera>(
                world_->getModel(/*modelName=*/modelName),
                modelName,
                cameraYamlCfg,
                xacroPath,
                urdfPath,
                resourceDir,
                kwargs);

        }

        std::unique_ptr<Kinematics> insertRobot(
            const std::string modelPath,
            const std::string modelName,
            int envIndex,
            int numEnvs, std::vector<std::string> joints = {},
            const std::string baseConstraint = "floating",
            const std::string baseFrame = "base_link") {

            int numAgents = int(sqrt(double(numEnvs)));
            double spawnRadius = 2.0;
            auto robotPtr = insertRobot(
                modelPath,
                modelName,
                -spawnRadius*((numAgents-1) - (numAgents-1)*(envIndex%numAgents)), // space the robots in the x-direction
                -spawnRadius*((numAgents-1) - (numAgents-1)*(int(std::floor(envIndex/numAgents))%numAgents)), // space the robots in the y-direction
                0., 0., 0., 0., // z, R, P, Y
                joints,
                baseConstraint,
                baseFrame
            );
            
            return robotPtr;
        }

        std::unique_ptr<Kinematics> insertRobot(
            const std::string modelPath,
            const std::string modelName,
            std::vector<std::string> joints = {},
            const std::string baseConstraint = "floating",
            const std::string baseFrame = "base_link") {

            auto robotPtr = insertRobot(modelPath, modelName, 0, 0, 0, 0, 0, 0, joints, baseConstraint, baseFrame); 
            
            return robotPtr;
        }

        std::unique_ptr<Kinematics> getModel(
            const std::string modelPath,
            const std::string modelName,
            std::vector<std::string> joints = {},
            const std::string baseConstraint = "floating",
            const std::string baseFrame = "base_link") { 

            return std::make_unique<Kinematics>(
                world_->getModel(/*modelName=*/modelName),
                modelPath,
                joints,
                baseConstraint,
                baseFrame); 
        }

        void setSimulationTimeStep(
            const double dt) {

            sdf::Root root;
            sdf::ElementPtr sdfElement;
            std::cout << "[World.hpp] Using default empty world" << std::endl;
            auto errors = root.LoadSdfString(scenario::gazebo::utils::getEmptyWorld());
            assert(errors.empty()); // TODO
            for (size_t worldIdx = 0; worldIdx < root.WorldCount(); ++worldIdx) {
                if (!scenario::gazebo::utils::updateSDFPhysics(root,
                                            dt,
                                            rtf_,
                                            /*realTimeUpdateRate=*/-1,
                                            worldIdx)) {
                    std::cout << "Failed to set physics profile" << std::endl;
                    throw "[World.hpp] Fatal error.";
                }
            }
        }

        double getSimulationTimeStep() {
            
            return simulationDt_;
        }

        void openGazeboGui(
            const int wait) {

            gazebo_->gui();
            std::this_thread::sleep_for(std::chrono::seconds(wait));
        }

        void pauseGazebo() {

            gazebo_->run(/*paused=*/true);
        }

        double currentSimTime() {

            return world_->time();
        }

        void removeModel(
            const std::string modelName) {

            if(!world_->removeModel(modelName)){
                std::cerr << "Failed to remove model from simulation." << std::endl;
            }
            pauseGazebo();
        }

    private:

        scenario::core::Pose xyzRPY2ScenarioPose(const double x, const double y, const double z, const double R, const double P, const double Y) {

            auto rotMat = rpy2RotMat<double>(R, P, Y);
            Eigen::Quaterniond quat(rotMat);
            
            return scenario::core::Pose(
                std::array<double, 3>({x, y, z}),
                std::array<double, 4>({quat.w(), quat.x(), quat.y(), quat.z()}));
        }

        void insertModel_(
            const std::string modelPath,
            const std::string modelName,
            scenario::core::Pose pose = scenario::core::Pose::Identity()){

            auto success = world_->insertModel(modelPath, pose, modelName);
            if(!success){ std::cout << "[World.hpp] Failed to insert model: " << modelName << std::endl; throw; }
            pauseGazebo();
        }

        void createCameraUrdf_(
            const std::string modelName,
            const YAML::Node& cameraYamlCfg,
            const std::string xacroPath,
            const std::string urdfPath,
            const std::string resourceDir,
            const std::unordered_map<std::string, std::any>& kwargs = {}) {

            if((cameraYamlCfg["camera"]["has_rgb"].template as<bool>() || cameraYamlCfg["camera"]["has_depth"].template as<bool>()) && firstRgbdCamera_){
                firstRgbdCamera_ = false;
                insertWorldPlugin("ignition-gazebo-sensors-system", "ignition::gazebo::systems::Sensors");
            }

            if(cameraYamlCfg["camera"]["has_logical"].template as<bool>()){
                firstLogicalCamera_ = false;
                insertWorldPlugin("ignition-gazebo-logical-camera-system", "ignition::gazebo::systems::LogicalCamera");
            }

            // Before spawning the camera, we need to make sure that we first generate its URDF according to the provided function parameters
            Camera::generateUrdfFromXacro(
                modelName, 
                xacroPath, 
                urdfPath, 
                resourceDir, 
                kwargs);

        }

        double simulationDt_ = 0.001;
        double rtf_;
        int stepsPerRun_;

        std::unique_ptr<scenario::gazebo::GazeboSimulator> gazebo_;
        std::shared_ptr<scenario::gazebo::World> world_;
        scenario::core::ModelPtr ground_;
        std::shared_ptr<scenario::gazebo::Model> groundGazebo_;

        bool firstRgbdCamera_ = true;
        bool firstLogicalCamera_ = true;

};

}

#endif // SRC_GYMIGNITIONWORLD_HPP