#ifndef dynamics_hpp
#define dynamics_hpp
#include "spacecraft.hpp"

struct Dynamics {

  // members
  const Spacecraft spacecraft;

  // constructor
  Dynamics (const Spacecraft & spacecraft_) : spacecraft(spacecraft_) {};

  // desctructor
  ~Dynamics (void) {};

};
