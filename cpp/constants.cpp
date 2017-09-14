#include <boost/python.hpp>
#include "../hpp/constants.hpp"

boost::python::scope().attr("R_EARTH") = R_EARTH;
boost::python::scope().attr("V_EARTH") = V_EARTH;
boost::python::scope().attr("A_EARTH") = A_EARTH;
boost::python::scope().attr("G_0") = G_0;
boost::python::scope().attr("MU_EARTH") = MU_EARTH;
