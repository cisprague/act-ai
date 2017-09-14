#ifndef spacecraft_hpp
#define spacecraft_hpp

struct Spacecraft {

	// constants
	const double mass, thrust, isp, veff;

	// constructor
	Spacecraft (const double & mass, const double & thrust, const double & isp) : mass(mass_), thrust(thrust_), isp(isp_), veff(G_0*isp) {};

	// destructor
	~Spacecraft (void) {};

};
