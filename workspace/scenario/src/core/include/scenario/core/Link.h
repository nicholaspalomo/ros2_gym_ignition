/*
 * Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT)
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the
 * GNU Lesser General Public License v2.1 or any later version.
 */

#ifndef SCENARIO_CORE_LINK_H
#define SCENARIO_CORE_LINK_H

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace scenario::core {
    class Link;
    class Model;
    struct Pose;
    struct Contact;
    struct ContactPoint;
} // namespace scenario::core

class scenario::core::Link
{
public:
    Link() = default;
    virtual ~Link() = default;

    /**
     * Check if the link is valid.
     *
     * @return True if the link is valid, false otherwise.
     */
    virtual bool valid() const = 0;

    /**
     * Get the name of the link.
     *
     * @param scoped If true, the scoped name of the link is returned.
     * @return The name of the link.
     */
    virtual std::string name(const bool scoped = false) const = 0;

    /**
     * Get the mass of the link.
     *
     * @return The mass of the link.
     */
    virtual double mass() const = 0;

    /**
     * Get the position of the link.
     *
     * The returned position is the position of the link frame, as it was
     * defined in the model file, in world coordinates.
     *
     * @return The cartesian position of the link frame in world coordinates.
     */
    virtual std::array<double, 3> position() const = 0;

    /**
     * Get the orientation of the link.
     *
     * The orientation is returned as a quaternion, which defines the
     * rotation between the world frame and the link frame.
     *
     * @return The wxyz quaternion defining the orientation if the link wrt the
     * world frame.
     */
    virtual std::array<double, 4> orientation() const = 0;

    /**
     * Get the linear mixed velocity of the link.
     *
     * @todo Add link to the velocity representation documentation page.
     *
     * @return The linear mixed velocity of the link.
     */
    virtual std::array<double, 3> worldLinearVelocity() const = 0;

    /**
     * Get the angular mixed velocity of the link.
     *
     * @todo Add link to the velocity representation documentation page.
     *
     * @return The angular mixed velocity of the link.
     */
    virtual std::array<double, 3> worldAngularVelocity() const = 0;

    /**
     * Get the linear body velocity of the link.
     *
     * @todo Add link to the velocity representation documentation page.
     *
     * @return The linear body velocity of the link.
     */
    virtual std::array<double, 3> bodyLinearVelocity() const = 0;

    /**
     * Get the angular body velocity of the link.
     *
     * @todo Add link to the velocity representation documentation page.
     *
     * @return The angular body velocity of the link.
     */
    virtual std::array<double, 3> bodyAngularVelocity() const = 0;

    /**
     * Get the linear mixed acceleration of the link.
     *
     * @todo Add link to the velocity representation documentation page.
     *
     * @return The linear mixed acceleration of the link.
     */
    virtual std::array<double, 3> worldLinearAcceleration() const = 0;

    /**
     * Get the angular mixed acceleration of the link.
     *
     * @todo Add link to the velocity representation documentation page.
     *
     * @return The angular mixed acceleration of the link.
     */
    virtual std::array<double, 3> worldAngularAcceleration() const = 0;

    /**
     * Get the linear body acceleration of the link.
     *
     * @todo Add link to the velocity representation documentation page.
     *
     * @return The linear body acceleration of the link.
     */
    virtual std::array<double, 3> bodyLinearAcceleration() const = 0;

    /**
     * Get the angular body acceleration of the link.
     *
     * @todo Add link to the velocity representation documentation page.
     *
     * @return The angular body acceleration of the link.
     */
    virtual std::array<double, 3> bodyAngularAcceleration() const = 0;

    /**
     * Check if the contact detection is enabled.
     *
     * @return True if the contact detection is enabled, false otherwise.
     */
    virtual bool contactsEnabled() const = 0;

    /**
     * Enable the contact detection.
     *
     * @param enable True to enable the contact detection, false to disable.
     * @return True for success, false otherwise.
     */
    virtual bool enableContactDetection(const bool enable) = 0;

    /**
     * Check if the link has active contacts.
     *
     * @return True if the link has at least one contact and contacts are
     * enabled, false otherwise.
     */
    virtual bool inContact() const = 0;

    /**
     * Get the active contacts of the link.
     *
     * @return The vector of active contacts.
     */
    virtual std::vector<Contact> contacts() const = 0;

    /**
     * Get the total wrench generated by the active contacts.
     *
     * All the contact wrenches are composed to an equivalent wrench
     * applied to the origin of the link frame and expressed in world
     * coordinates.
     *
     * @return The total wrench of the active contacts.
     */
    virtual std::array<double, 6> contactWrench() const = 0;
};

struct scenario::core::Pose
{
    Pose() = default;
    Pose(std::array<double, 3> p, std::array<double, 4> o)
        : position(p)
        , orientation(o)
    {}
    Pose(std::pair<std::array<double, 3>, std::array<double, 4>> pose)
        : position(pose.first)
        , orientation(pose.second)
    {}

    static Pose Identity() { return {}; }

    bool operator==(const Pose& other) const
    {
        return this->position == other.position
               && this->orientation == other.orientation;
    }

    bool operator!=(const Pose& other) const { return !(*this == other); }

    std::array<double, 3> position = {0, 0, 0};
    std::array<double, 4> orientation = {1, 0, 0, 0};
};

struct scenario::core::ContactPoint
{
    double depth;
    std::array<double, 3> force;
    std::array<double, 3> torque;
    std::array<double, 3> normal;
    std::array<double, 3> position;
};

struct scenario::core::Contact
{
    std::string bodyA;
    std::string bodyB;
    std::vector<ContactPoint> points;
};

#endif // SCENARIO_CORE_LINK_H
