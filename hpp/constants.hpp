#ifndef constants_hpp
#define constants_hpp

// astronamical unit [m]
const double R_EARTH(149597870700);

// orbital velocity  [m/s]
const double V_EARTH(29784.6905);

// centripetal acceleration [m/s^2]
const double A_EARTH(V_EARTH*V_EARTH/R_EARTH);

// sea-level gravitational acceleration [m/s^2]
const double G_0(9.80665);

// gravitational parametre of Earth [m^3/s^2]
const double MU_EARTH(398600441800000.0);

#endif