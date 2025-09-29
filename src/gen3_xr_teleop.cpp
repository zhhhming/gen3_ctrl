/**
 * Gen3 XR Teleoperation Controller
 * Real-time control of Kinova Gen3 robot using XR controller input
 * 
 * Dependencies:
 * - Gen3RobotController (hardware interface)
 * - TRAC-IK (inverse kinematics)
 * - Python XRoboToolkit SDK (via pybind11)
 */

#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <deque>
#include <signal.h>
#include <fstream>
#include <algorithm>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Python embedding
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// TRAC-IK
#include <trac_ik/trac_ik.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

// Eigen for math operations
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Robot controller
#include "Gen3RobotController.h"

namespace py = pybind11;

// Global flag for clean shutdown
std::atomic<bool> g_shutdown_requested(false);

void signal_handler(int sig) {
    std::cout << "\nShutdown signal received" << std::endl;
    g_shutdown_requested = true;
}

class XRTeleopController {
public:
    XRTeleopController(const std::string& robot_urdf_path,
                       const std::string& robot_ip = "192.168.1.10",
                       int tcp_port = 10000,
                       int udp_port = 10001,
                       const std::string& username = "admin",
                       const std::string& password = "admin")
        : robot_urdf_path_(robot_urdf_path),
          robot_ip_(robot_ip),
          tcp_port_(tcp_port),
          udp_port_(udp_port),
          username_(username),
          password_(password),
          is_active_(false),
          shutdown_requested_(false),
          num_joints_(7),
          scale_factor_(1.0),
          ik_rate_hz_(50),
          control_rate_hz_(1000)
    {
        // Initialize thread synchronization
        target_joints_.resize(num_joints_, 0.0f);
        current_joints_.resize(num_joints_, 0.0f);
        target_gripper_ = 0.0f;
        
        // Initialize coordinate transforms
        initializeTransforms();
    }
    
    ~XRTeleopController() {
        shutdown();
    }
    
    bool initialize() {
        std::cout << "Initializing XR Teleoperation Controller..." << std::endl;
        
        // 1. Initialize Python interpreter and XR SDK
        if (!initializePython()) {
            std::cerr << "Failed to initialize Python/XR SDK" << std::endl;
            return false;
        }
        
        // 2. Initialize robot controller
        if (!initializeRobot()) {
            std::cerr << "Failed to initialize robot controller" << std::endl;
            return false;
        }
        
        // 3. Initialize TRAC-IK
        if (!initializeTracIK()) {
            std::cerr << "Failed to initialize TRAC-IK" << std::endl;
            return false;
        }
        
        std::cout << "Controller initialized successfully!" << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "Starting teleoperation..." << std::endl;
        
        // Start IK thread
        std::thread ik_thread(&XRTeleopController::ikThread, this);
        
        // Start control thread with high priority
        std::thread control_thread(&XRTeleopController::controlThread, this);
        
        // Set control thread priority (Linux specific)
#ifdef __linux__
        sched_param sch_params;
        sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
        if (pthread_setschedparam(control_thread.native_handle(), SCHED_FIFO, &sch_params)) {
            std::cerr << "Warning: Failed to set control thread priority" << std::endl;
        }
#endif
        
        // Main thread waits for shutdown
        while (!shutdown_requested_ && !g_shutdown_requested) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Signal threads to stop
        shutdown_requested_ = true;
        
        // Wait for threads to finish
        if (ik_thread.joinable()) {
            ik_thread.join();
        }
        if (control_thread.joinable()) {
            control_thread.join();
        }
        
        std::cout << "Teleoperation stopped" << std::endl;
    }
    
    void shutdown() {
        shutdown_requested_ = true;
        
        // Clean up robot
        if (robot_controller_) {
            robot_controller_->exitLowLevelMode();
            robot_controller_->stopRobot();
            robot_controller_->shutdown();
            robot_controller_.reset();
        }
        
        // Clean up Python
        if (Py_IsInitialized()) {
            xr_client_ = py::none();
            py::finalize_interpreter();
        }
    }
    
private:
    // Configuration
    std::string robot_urdf_path_;
    std::string robot_ip_;
    int tcp_port_;
    int udp_port_;
    std::string username_;
    std::string password_;
    
    // Control parameters
    double scale_factor_;
    int ik_rate_hz_;
    int control_rate_hz_;
    int num_joints_;
    
    // Robot controller
    std::unique_ptr<Gen3RobotController> robot_controller_;
    
    // TRAC-IK
    std::unique_ptr<TRAC_IK::TRAC_IK> tracik_solver_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    KDL::Chain kdl_chain_;
    
    // Python XR SDK
    py::object xr_client_;
    
    // Thread control
    std::atomic<bool> shutdown_requested_;
    
    // Shared state (protected by mutex)
    std::mutex state_mutex_;
    std::vector<float> target_joints_;
    std::vector<float> current_joints_;
    float target_gripper_;
    
    // Control state
    std::atomic<bool> is_active_;
    
    // Reference frames
    bool ref_ee_valid_ = false;
    KDL::Frame ref_ee_frame_;
    bool ref_controller_valid_ = false;
    Eigen::Vector3d ref_controller_pos_;
    Eigen::Quaterniond ref_controller_quat_;
    
    // Coordinate transforms
    Eigen::Matrix3d R_headset_world_;
    Eigen::Matrix3d R_z_90_cw_;
    
    // Filter state
    std::vector<float> filtered_joint_state_;
    bool filter_initialized_ = false;
    const float filter_alpha_ = 0.2f;
    
    bool initializePython() {
        try {
            // Initialize Python interpreter
            py::initialize_interpreter();
            
            // Import XRoboToolkit SDK
            py::module sys = py::module::import("sys");
            sys.attr("path").attr("append")("/home/ming/miniconda3/envs/xrrobotics/lib/python3.10/site-packages");
            
            // Create XR client
            py::module xrt = py::module::import("xrobotoolkit_sdk");
            xrt.attr("init")();
            
            // Store reference to SDK for later use
            xr_client_ = xrt;
            
            std::cout << "Python XR SDK initialized" << std::endl;
            return true;
            
        } catch (py::error_already_set& e) {
            std::cerr << "Python error: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool initializeRobot() {
        try {
            // Create robot controller
            robot_controller_ = std::make_unique<Gen3RobotController>(
                robot_ip_, tcp_port_, udp_port_, username_, password_
            );
            
            // Initialize connection
            if (!robot_controller_->initialize()) {
                return false;
            }
            
            // Clear any faults
            robot_controller_->clearFaults();
            
            // Enter low-level mode
            if (!robot_controller_->enterLowLevelMode()) {
                return false;
            }
            
            // Get initial joint positions
            auto positions = normalizeAngles(robot_controller_->getJointPositions());
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                current_joints_ = positions;
                target_joints_ = positions;  // Initialize targets to current
            }
            
            initializeFilterState(positions);

            std::cout << "Robot controller initialized" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Robot initialization error: " << e.what() << std::endl;
            return false;
        }
    }
    float normalizeAngle(float angle) const {
        float normalized = std::fmod(angle + 180.0f, 360.0f);
        if (normalized < 0.0f) {
            normalized += 360.0f;
        }
        return normalized - 180.0f;
    }

    std::vector<float> normalizeAngles(const std::vector<float>& angles) const {
        std::vector<float> normalized;
        normalized.reserve(angles.size());
        for (float angle : angles) {
            normalized.push_back(normalizeAngle(angle));
        }
        return normalized;
    }

    float unwrapAngle(float target, float reference) const {
        double unwrapped = static_cast<double>(reference) +
                           std::remainder(static_cast<double>(target) - static_cast<double>(reference), 360.0);
        return static_cast<float>(unwrapped);
    }

    void initializeFilterState(const std::vector<float>& initial_positions) {
        filtered_joint_state_ = initial_positions;
        filter_initialized_ = true;
    }
    
    bool initializeTracIK() {
        try {
            // Read URDF
            std::ifstream urdf_file(robot_urdf_path_);
            if (!urdf_file.is_open()) {
                std::cerr << "Cannot open URDF: " << robot_urdf_path_ << std::endl;
                return false;
            }
            
            std::string urdf_string((std::istreambuf_iterator<char>(urdf_file)),
                                    std::istreambuf_iterator<char>());
            
            // Initialize TRAC-IK solver
            tracik_solver_ = std::make_unique<TRAC_IK::TRAC_IK>(
                "base_link",           // base frame
                "bracelet_link",       // tip frame
                urdf_string,           // URDF
                0.005,                 // timeout (5ms)
                0.001,                 // epsilon (1mm)
                TRAC_IK::Distance      // solve type
            );
            
            // Get KDL chain
            KDL::Tree kdl_tree;
            if (!kdl_parser::treeFromString(urdf_string, kdl_tree)) {
                std::cerr << "Failed to parse URDF to KDL tree" << std::endl;
                return false;
            }
            
            if (!kdl_tree.getChain("base_link", "bracelet_link", kdl_chain_)) {
                std::cerr << "Failed to extract KDL chain" << std::endl;
                return false;
            }
            
            // Initialize FK solver
            fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_chain_);
            
            std::cout << "TRAC-IK initialized with " << kdl_chain_.getNrOfJoints() << " joints" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "TRAC-IK initialization error: " << e.what() << std::endl;
            return false;
        }
    }
    
    void initializeTransforms() {
        // Headset to world transform
        R_headset_world_ << 0, 0, -1,
                           -1, 0, 0,
                            0, 1, 0;
        
        // 90 degree CW rotation around Z
        R_z_90_cw_ << 0, 1, 0,
                     -1, 0, 0,
                      0, 0, 1;
    }
    
    py::array_t<double> getXRPose(const std::string& device) {
        if (device == "right_controller") {
            return py::cast<py::array_t<double>>(xr_client_.attr("get_right_controller_pose")());
        } else if (device == "left_controller") {
            return py::cast<py::array_t<double>>(xr_client_.attr("get_left_controller_pose")());
        } else {
            return py::cast<py::array_t<double>>(xr_client_.attr("get_headset_pose")());
        }
    }
    
    float getXRValue(const std::string& name) {
        if (name == "right_grip") {
            return py::cast<float>(xr_client_.attr("get_right_grip")());
        } else if (name == "right_trigger") {
            return py::cast<float>(xr_client_.attr("get_right_trigger")());
        } else if (name == "left_grip") {
            return py::cast<float>(xr_client_.attr("get_left_grip")());
        } else if (name == "left_trigger") {
            return py::cast<float>(xr_client_.attr("get_left_trigger")());
        }
        return 0.0f;
    }
    
    void processControllerPose(const py::array_t<double>& xr_pose,
                              Eigen::Vector3d& delta_pos,
                              Eigen::Vector3d& delta_rot) {
        auto r = xr_pose.unchecked<1>();
        
        // Extract position and quaternion
        Eigen::Vector3d controller_pos(r(0), r(1), r(2));
        Eigen::Quaterniond controller_quat(r(6), r(3), r(4), r(5));  // w,x,y,z
        
        // Transform to world coordinates
        controller_pos = R_headset_world_ * controller_pos;
        Eigen::Quaterniond R_quat(R_headset_world_);
        controller_quat = R_quat * controller_quat * R_quat.conjugate();
        
        // Calculate deltas
        if (!ref_controller_valid_) {
            ref_controller_pos_ = controller_pos;
            ref_controller_quat_ = controller_quat;
            ref_controller_valid_ = true;
            delta_pos.setZero();
            delta_rot.setZero();
        } else {
            delta_pos = (controller_pos - ref_controller_pos_) * scale_factor_;
            
            // Quaternion difference as angle-axis
            Eigen::Quaterniond quat_diff = controller_quat * ref_controller_quat_.conjugate();
            Eigen::AngleAxisd angle_axis(quat_diff);
            delta_rot = angle_axis.angle() * angle_axis.axis();
        }
        
        // Apply 90 degree rotation
        delta_pos = R_z_90_cw_ * delta_pos;
        delta_rot = R_z_90_cw_ * delta_rot;
    }
    
    KDL::Frame eigenToKDL(const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat) {
        KDL::Frame frame;
        frame.p = KDL::Vector(pos.x(), pos.y(), pos.z());
        frame.M = KDL::Rotation::Quaternion(quat.x(), quat.y(), quat.z(), quat.w());
        return frame;
    }
    
    void kdlToEigen(const KDL::Frame& frame, Eigen::Vector3d& pos, Eigen::Quaterniond& quat) {
        pos = Eigen::Vector3d(frame.p.x(), frame.p.y(), frame.p.z());
        double x, y, z, w;
        frame.M.GetQuaternion(x, y, z, w);
        quat = Eigen::Quaterniond(w, x, y, z);
    }
    
    void ikThread() {
        std::cout << "IK thread started" << std::endl;
        auto dt = std::chrono::duration<double>(1.0 / ik_rate_hz_);
        
        while (!shutdown_requested_ && !g_shutdown_requested) {
            auto loop_start = std::chrono::steady_clock::now();
            
            try {
                // Get XR inputs
                float grip_value = 0.f, trigger_value = 0.f;
                {
                    py::gil_scoped_acquire gil;
                    grip_value = getXRValue("right_grip");
                    trigger_value = getXRValue("right_trigger");
                }
                                
                // Update gripper target
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    target_gripper_ = std::max(0.0f, std::min(1.0f, trigger_value));
                }
                
                // Check activation
                bool new_active = (grip_value > 0.9f);
                
                if (new_active != is_active_) {
                    if (new_active) {
                        std::cout << "Control activated" << std::endl;
                        
                        // Reset references on activation
                        ref_ee_valid_ = false;
                        ref_controller_valid_ = false;
                        
                    } else {
                        std::cout << "Control deactivated" << std::endl;
                    }
                    is_active_ = new_active;
                }
                
                if (is_active_) {
                    // Get current joint positions for FK
                    KDL::JntArray current_joints_kdl(num_joints_);
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        for (int i = 0; i < num_joints_; ++i) {
                            current_joints_kdl(i) = current_joints_[i] * M_PI / 180.0;  // deg to rad
                        }
                    }
                    
                    // Initialize reference frame if needed
                    if (!ref_ee_valid_) {
                        fk_solver_->JntToCart(current_joints_kdl, ref_ee_frame_);
                        ref_ee_valid_ = true;
                    }
                    
                    // Get controller pose and calculate deltas
                    py::array_t<double> xr_pose = getXRPose("right_controller");
                    Eigen::Vector3d delta_pos, delta_rot;
                    processControllerPose(xr_pose, delta_pos, delta_rot);
                    
                    // Apply deltas to reference frame
                    Eigen::Vector3d ref_pos;
                    Eigen::Quaterniond ref_quat;
                    kdlToEigen(ref_ee_frame_, ref_pos, ref_quat);
                    
                    // Update position
                    Eigen::Vector3d target_pos = ref_pos + delta_pos;
                    
                    // Update orientation
                    double angle = delta_rot.norm();
                    Eigen::Quaterniond target_quat = ref_quat;
                    if (angle > 1e-6) {
                        Eigen::Vector3d axis = delta_rot / angle;
                        Eigen::AngleAxisd delta_rotation(angle, axis);
                        target_quat = delta_rotation * ref_quat;
                    }
                    
                    // Convert to KDL frame
                    KDL::Frame target_frame = eigenToKDL(target_pos, target_quat);
                    
                    // Solve IK
                    KDL::JntArray ik_solution(num_joints_);
                    int ret = tracik_solver_->CartToJnt(current_joints_kdl, target_frame, ik_solution);
                    
                    if (ret >= 0) {
                        // Convert solution to degrees and update target
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        for (int i = 0; i < num_joints_; ++i) {
                            target_joints_[i] = normalizeAngle(ik_solution(i) * 180.0 / M_PI);
                        }
                    } else {
                        // IK failed, keep previous target
                        static int fail_count = 0;
                        if (++fail_count % 50 == 0) {  // Print every second at 50Hz
                            std::cerr << "IK solution not found" << std::endl;
                        }
                    }
                }
                
            } catch (const std::exception& e) {
                std::cerr << "IK thread error: " << e.what() << std::endl;
            }
            
            // Maintain loop rate
            auto loop_end = std::chrono::steady_clock::now();
            auto loop_duration = loop_end - loop_start;
            if (loop_duration < dt) {
                std::this_thread::sleep_for(dt - loop_duration);
            }
        }
        
        std::cout << "IK thread stopped" << std::endl;
    }
    
    std::vector<float> filterJointPositions(const std::vector<float>& target_positions) {
        if (!filter_initialized_) {
            initializeFilterState(target_positions);
            return target_positions;
        }

        std::vector<float> filtered(num_joints_, 0.0f);
        for (int i = 0; i < num_joints_; ++i) {
            float unwrapped_target = unwrapAngle(target_positions[i], filtered_joint_state_[i]);
            float filtered_angle = filter_alpha_ * unwrapped_target + (1.0f - filter_alpha_) * filtered_joint_state_[i];
            filtered[i] = filtered_angle;
            filtered_joint_state_[i] = filtered_angle;
        }
        

        return filtered;
    }
    
    void controlThread() {
        std::cout << "Control thread started at " << control_rate_hz_ << "Hz" << std::endl;
        auto dt = std::chrono::duration<double>(1.0 / control_rate_hz_);
        
        // Performance monitoring
        std::deque<double> loop_times;
        const size_t max_samples = 1000;
        auto last_report = std::chrono::steady_clock::now();
        
        while (!shutdown_requested_ && !g_shutdown_requested) {
            auto loop_start = std::chrono::steady_clock::now();
            
            try {
                // Get target positions
                std::vector<float> target_joints;
                float target_gripper;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    target_joints = target_joints_;
                    target_gripper = target_gripper_;
                }
                
                // Apply filtering to joint targets
                std::vector<float> filtered_joints = filterJointPositions(target_joints);
                
                // Send joint positions
                robot_controller_->setJointPositions(filtered_joints);
                
                // Send gripper command (no filtering)
                robot_controller_->setGripperPosition(target_gripper, 1.0f);
                
                // Send commands and refresh feedback
                if (!robot_controller_->sendCommandAndRefresh()) {
                    std::cerr << "Failed to send command" << std::endl;
                }
                
                // Update current joint positions
                auto current = normalizeAngles(robot_controller_->getJointPositions());
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    current_joints_ = current;
                }
                
                // Performance monitoring
                auto loop_end = std::chrono::steady_clock::now();
                auto loop_duration = std::chrono::duration<double>(loop_end - loop_start).count();
                loop_times.push_back(loop_duration * 1000.0);  // Convert to ms
                
                if (loop_times.size() > max_samples) {
                    loop_times.pop_front();
                }
                
                // Report statistics every 2 seconds
                if (loop_end - last_report > std::chrono::seconds(2)) {
                    double avg = 0, max = 0;
                    for (double t : loop_times) {
                        avg += t;
                        max = std::max(max, t);
                    }
                    avg /= loop_times.size();
                    
                    std::cout << "Control loop: avg=" << avg << "ms, max=" << max 
                             << "ms, rate=" << (1000.0/avg) << "Hz" << std::endl;
                    
                    last_report = loop_end;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Control thread error: " << e.what() << std::endl;
            }
            
            // Maintain loop rate
            auto loop_end = std::chrono::steady_clock::now();
            auto loop_duration = loop_end - loop_start;
            if (loop_duration < dt) {
                std::this_thread::sleep_for(dt - loop_duration);
            } else {
                static int overrun_count = 0;
                if (++overrun_count % 100 == 0) {
                    std::cerr << "Control loop overrun: " 
                             << std::chrono::duration<double, std::milli>(loop_duration).count() 
                             << "ms" << std::endl;
                }
            }
        }
        
        std::cout << "Control thread stopped" << std::endl;
    }
};

int main(int argc, char** argv) {
    // Install signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Configuration
    std::string urdf_path = "/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-7DOF.urdf";
    std::string robot_ip = "192.168.1.10";
    
    // Parse command line arguments
    if (argc > 1) {
        robot_ip = argv[1];
    }
    if (argc > 2) {
        urdf_path = argv[2];
    }
    
    std::cout << "==================================" << std::endl;
    std::cout << "Gen3 XR Teleoperation Controller" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Robot IP: " << robot_ip << std::endl;
    std::cout << "URDF: " << urdf_path << std::endl;
    std::cout << std::endl;
    
    try {
        // Create controller
        XRTeleopController controller(urdf_path, robot_ip);
        
        // Initialize
        if (!controller.initialize()) {
            std::cerr << "Failed to initialize controller" << std::endl;
            return 1;
        }
        
        // Run main control loop
        controller.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Program terminated successfully" << std::endl;
    return 0;
}