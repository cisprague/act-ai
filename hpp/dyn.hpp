#ifndef dyn_hpp
#define dyn_hpp

namespace dynamical_system {

	// base class
	struct system {

		// constants
		const int sdim, cdim;

		// constructor
		system (
			const int & sdim,
			const int & cdim
		) {};

		// destructor
		~system (void) {};
	}
}