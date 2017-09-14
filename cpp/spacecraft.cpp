#include <boost/python.hpp>
#include "../hpp/spacecraft.hpp"

using namespace boost::python;

BOOST_PYTHON_MODULE (actai) {
  class_<Spacecraft>("Spacecraft", init<double, double, double>())
    .def_readonly("mass", &Spacecraft::mass)
    .def_readonly("thrust", &Spacecraft::thrust)
    .def_readonly("isp", &Spacecraft::isp)
    .def_readonly("veff", &Spacecraft::veff);
};
